import ezdxf
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os

# 注意关于AutoCAD DXF布局和视口的说明:
# 1. 每个布局(Layout)是一个纸空间，可以包含多个视口(Viewport)
# 2. 视口用于在纸空间中显示模型空间的内容
# 3. 在ezdxf中，每个图形内容对应一个块(Block)
# 4. 为了让每个楼层的内容分开显示，我们为每个楼层创建单独的块
# 5. 视口的属性设置决定了它如何显示块内容
# 6. 布局中status=1的视口会被AutoCAD识别为活动视口

# 全局常量
SCALE = 0.02  # 原来是0.01，再加大一倍以提高可见性

def create_floor_layout(doc, floor_name, ms_block, layout_name=None):
    """创建楼层布局"""
    # 如果未指定布局名称，使用楼层名称
    if layout_name is None:
        layout_name = f"楼层 {floor_name}"
    
    # 创建新布局
    layout = doc.layouts.new(layout_name)
    
    # 获取布局的模型空间（paperspace）
    # 注意：在新版本的ezdxf中，不需要使用layout.block
    
    # 创建视口
    viewport = layout.add_viewport(
        center=(150, 150),  # 视口中心位置，调大便于看清
        size=(300, 300),    # 视口大小，调大便于看清
        view_center_point=(0, 0),  # 视图中心点
        view_height=500     # 视图高度，调大以显示更多内容
    )
    
    # 增加一个引用块
    try:
        layout.add_blockref(ms_block.name, (0, 0))
        print(f"  ✓ 成功添加块引用 {ms_block.name} 到布局 {layout_name}")
    except Exception as e:
        print(f"  ⚠️ 添加块引用失败: {e}")
    
    return layout

def export_to_dxf(design_file: str, heating_design_file: str = None, output_file=None) -> str:
    """
    将AR设计文件和地暖设计文件导出为DXF格式
    
    Args:
        design_file: AR设计JSON文件路径
        heating_design_file: 地暖设计JSON文件路径，如果不指定则使用相同文件名但扩展名为.dxf
        output_file: 输出DXF文件路径，如果未指定则使用默认路径
        
    Returns:
        生成的DXF文件路径
    """
    # 如果未指定输出文件，使用默认路径
    if output_file is None:
        # 从输入文件名生成输出文件名
        base_name = os.path.basename(heating_design_file or design_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join('output', f'{name_without_ext}.dxf')

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("\n=== DXF导出工具 ===\n")
    print(f"🔷 开始导出DXF文件...")
    print(f"  - 设计文件: {design_file}")
    if heating_design_file:
        print(f"  - 地暖设计文件: {heating_design_file}")
    print()
    
    # 加载设计文件
    with open(design_file, 'r', encoding='utf-8') as f:
        design_data = json.load(f)
    
    # 加载地暖设计文件（如果有）
    heating_data = None
    if heating_design_file and os.path.exists(heating_design_file):
        try:
            with open(heating_design_file, 'r', encoding='utf-8') as f:
                heating_data = json.load(f)
            print(f"✅ 成功加载地暖设计文件: {heating_design_file}")
            
            # 检查是否是多楼层地暖文件
            if "Floors" in heating_data:
                floors = heating_data.get("Floors", [])
                print(f"  - 多楼层地暖文件，包含 {len(floors)} 个楼层")
                for floor in floors:
                    level_name = floor.get("LevelName", "未知")
                    pipes_count = len(floor.get("Pipes", []))
                    collectors_count = len(floor.get("Collectors", []))
                    print(f"    - 楼层 {level_name}: {pipes_count} 根管道, {collectors_count} 个集水器")
            else:
                pipes_count = len(heating_data.get("Pipes", []))
                collectors_count = len(heating_data.get("Collectors", []))
                print(f"  - 单楼层地暖文件: {pipes_count} 根管道, {collectors_count} 个集水器")
            print()
        except Exception as e:
            print(f"⚠️ 加载地暖设计文件失败: {str(e)}")
            heating_data = None
    
    # 创建楼层数据映射
    floor_data_map = {}
    
    # 检测设计文件的数据结构
    # 判断是否是新版数据结构（Floor列表）
    if "Floor" in design_data:
        print(f"  ✓ 检测到新版数据结构，使用Floor列表")
        floors = design_data.get("Floor", [])
        for floor in floors:
            floor_name = floor.get("Num", "")
            if floor_name:
                print(f"  - 发现楼层 {floor_name}")
                floor_data_map[floor_name] = {
                    'design': floor,
                    'heating': None  # 先设为空，后面添加
                }
    # 旧版数据结构（floors）
    elif "floors" in design_data:
        print(f"  ✓ 检测到旧版数据结构，使用floors列表")
        floors = design_data.get("floors", [])
        for floor in floors:
            floor_name = floor.get("floorNum", "")
            if floor_name:
                floor_data_map[floor_name] = {
                    'design': floor,
                    'heating': None  # 先设为空，后面添加
                }
    else:
        print(f"  ⚠️ 未能识别设计文件数据结构，无法提取楼层信息")
    
    # 从地暖数据获取楼层信息
    if heating_data:
        # 检查是否是多楼层文件
        if "Floors" in heating_data:
            # 多楼层文件
            floors_data = heating_data.get("Floors", [])
        else:
            # 单楼层文件
            floors_data = [heating_data]
        
        # 遍历每个楼层地暖数据
        for floor_data in floors_data:
            level_name = floor_data.get("LevelName", "")
            if level_name in floor_data_map:
                floor_data_map[level_name]['heating'] = floor_data
            else:
                # 如果在设计文件中没找到对应楼层，也创建一个新条目
                floor_data_map[level_name] = {
                    'design': None,
                    'heating': floor_data
                }
    
    # 创建一个新的DXF文档
    doc = ezdxf.new('R2010')  # 使用更兼容的版本
    
    # 创建所需的图层
    doc.layers.new(name='WALLS', dxfattribs={'color': 1})      # 蓝色墙体
    doc.layers.new(name='DOORS', dxfattribs={'color': 2})      # 绿色门
    doc.layers.new(name='WINDOWS', dxfattribs={'color': 3})    # 红色窗户
    doc.layers.new(name='ROOMS', dxfattribs={'color': 4})      # 洋红色房间
    doc.layers.new(name='TEXT', dxfattribs={'color': 7})       # 白色文本
    doc.layers.new(name='HEATING_PIPES', dxfattribs={'color': 5})  # 青色地暖管道
    doc.layers.new(name='COLLECTORS', dxfattribs={'color': 6})  # 紫色集水器
    
    # 设置布局
    # ModelSpace是整个设计的总视图
    msp = doc.modelspace()
    
    # 打印楼层数
    floors = list(floor_data_map.keys())
    floors.sort()  # 排序楼层
    print(f"✅ 找到 {len(floors)} 个楼层\n")
    
    # 计算每个楼层的偏移量
    floor_positions = {}
    for i, floor_name in enumerate(floors):
        floor_positions[floor_name] = (i * 1000, 0)  # 横向偏移1000单位
    
    # 为每个楼层创建独立的模型空间内容块
    floor_blocks = {}
    
    # 处理每个楼层
    for floor_name in floors:
        floor_data = floor_data_map[floor_name]
        
        # 创建该楼层的模型空间块
        block_name = f"FLOOR_{floor_name}"
        floor_block = doc.blocks.new(name=block_name)
        floor_blocks[floor_name] = floor_block
        
        print(f"🔷 处理楼层 [{floor_name}] 的图形:")
        
        # 检查建筑数据是否存在
        if floor_data['design'] is not None:
            print(f"  - 找到建筑设计数据，准备绘制...")
            # 绘制建筑元素
            draw_building_elements(floor_block, floor_data['design'])
        else:
            print(f"  ✗ 没有建筑设计数据")
        
        # 绘制地暖元素
        if floor_data['heating'] is not None:
            # 计算该楼层的偏移量
            floor_offset = floor_positions[floor_name]
            
            # 绘制地暖元素（不需要偏移，因为每个楼层都在自己的块中）
            draw_heating_with_offset(floor_block, floor_data['heating'], SCALE, (0, 0))
            print(f"  ✓ 绘制地暖元素")
        else:
            print(f"  ✗ 没有地暖设计数据")

        # 添加楼层标识
        try:
            floor_block.add_text(
                f"楼层 {floor_name}",
                dxfattribs={
                    'layer': 'TEXT',
                    'height': 200 * SCALE,
                    'color': 7,  # 白色
                    'insert': (2500 * SCALE, 4500 * SCALE)
                }
            )
            print(f"  ✓ 添加楼层标识")
        except Exception as e:
            print(f"  ⚠️ 添加楼层标识失败: {e}")
            
        print()  # 添加空行分隔
    
    # 在主模型空间放置各楼层块的实例，应用偏移
    for floor_name, floor_offset in floor_positions.items():
        if floor_name in floor_blocks:
            # 插入该楼层的块
            block_reference = msp.add_blockref(
                floor_blocks[floor_name].name,
                insert=floor_offset,
                dxfattribs={'layer': 'WALLS'}
            )
    
    # 为每个楼层创建独立的布局（图纸空间）
    for floor_name in floors:
        print(f"🔷 正在创建楼层 [{floor_name}] 的设计图...")
        
        # 创建该楼层的布局，并设置其内容为对应的块
        layout = create_floor_layout(
            doc,
            floor_name,
            floor_blocks[floor_name],
            layout_name=f"楼层 {floor_name}"
        )
    
    # 保存DXF文件
    doc.saveas(output_file)
    print(f"\n✅ DXF文件已成功导出至: {output_file}")
    
    return output_file

def draw_building_elements(space, floor_data):
    """
    绘制建筑元素，例如墙体、门、窗户和房间
    
    Args:
        space: 要绘制到的空间（模型空间或块）
        floor_data: 楼层数据
    """
    building_elements_drawn = 0  # 计数器，追踪绘制的建筑元素数量
    
    # 检查数据结构，适配不同格式
    # 新格式：Construction包含各种元素
    if "Construction" in floor_data:
        construction = floor_data.get("Construction", {})
        
        # 绘制墙体
        walls = construction.get("Wall", [])
        if walls:
            print(f"  - 绘制墙体: {len(walls)} 个墙体")
            for wall in walls:
                if "Curve" in wall:
                    curve = wall.get("Curve", {})
                    start_point = curve.get("StartPoint", {})
                    end_point = curve.get("EndPoint", {})
                    
                    if start_point and end_point:
                        try:
                            # 绘制墙体中心线
                            line = space.add_line(
                                (start_point.get("x", 0) * SCALE, start_point.get("y", 0) * SCALE),
                                (end_point.get("x", 0) * SCALE, end_point.get("y", 0) * SCALE),
                                dxfattribs={'layer': 'WALLS', 'lineweight': 40}
                            )
                            
                            # 如果有FirstLine和SecondLine，绘制墙体边线
                            if "FirstLine" in wall and "SecondLine" in wall:
                                first_line = wall.get("FirstLine", {})
                                second_line = wall.get("SecondLine", {})
                                
                                if first_line and "StartPoint" in first_line and "EndPoint" in first_line:
                                    space.add_line(
                                        (first_line["StartPoint"].get("x", 0) * SCALE, first_line["StartPoint"].get("y", 0) * SCALE),
                                        (first_line["EndPoint"].get("x", 0) * SCALE, first_line["EndPoint"].get("y", 0) * SCALE),
                                        dxfattribs={'layer': 'WALLS', 'lineweight': 25}
                                    )
                                
                                if second_line and "StartPoint" in second_line and "EndPoint" in second_line:
                                    space.add_line(
                                        (second_line["StartPoint"].get("x", 0) * SCALE, second_line["StartPoint"].get("y", 0) * SCALE),
                                        (second_line["EndPoint"].get("x", 0) * SCALE, second_line["EndPoint"].get("y", 0) * SCALE),
                                        dxfattribs={'layer': 'WALLS', 'lineweight': 25}
                                    )
                            
                            building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制墙体失败: {e}")
        
        # 绘制门
        doors = construction.get("Door", []) or construction.get("DoorAndWindow", [])
        if doors:
            print(f"  - 绘制门: {len(doors)} 个门")
            for door in doors:
                if "BaseLine" in door:
                    base_line = door.get("BaseLine", {})
                    start_point = base_line.get("StartPoint", {})
                    end_point = base_line.get("EndPoint", {})
                    
                    if start_point and end_point:
                        try:
                            # 绘制门的基线
                            line = space.add_line(
                                (start_point.get("x", 0) * SCALE, start_point.get("y", 0) * SCALE),
                                (end_point.get("x", 0) * SCALE, end_point.get("y", 0) * SCALE),
                                dxfattribs={'layer': 'DOORS', 'lineweight': 35, 'color': 2}
                            )
                            
                            # 如果有门的位置和尺寸信息，绘制门的轮廓
                            if "Location" in door and "Size" in door:
                                location = door.get("Location", {})
                                size = door.get("Size", {})
                                width = size.get("Width", 0)
                                
                                # 绘制简单的门符号（90度线）
                                if width > 0:
                                    center_x = location.get("x", 0)
                                    center_y = location.get("y", 0)
                                    half_width = width / 2
                                    
                                    # 在门的位置绘制一个圆，表示门的位置
                                    space.add_circle(
                                        (center_x * SCALE, center_y * SCALE),
                                        radius=20 * SCALE,
                                        dxfattribs={'layer': 'DOORS', 'color': 2}
                                    )
                            
                            building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制门失败: {e}")
        
        # 绘制房间
        rooms = construction.get("Room", [])
        if rooms:
            print(f"  - 绘制房间: {len(rooms)} 个房间")
            for room in rooms:
                if "Boundary" in room:
                    boundaries = room.get("Boundary", [])
                    if boundaries:
                        try:
                            # 收集房间边界点
                            points = []
                            for boundary in boundaries:
                                start_point = boundary.get("StartPoint", {})
                                if start_point:
                                    points.append((start_point.get("x", 0) * SCALE, start_point.get("y", 0) * SCALE))
                            
                            # 如果至少有3个点，绘制房间边界
                            if len(points) >= 3:
                                # 绘制房间边界多段线
                                polyline = space.add_lwpolyline(
                                    points,
                                    dxfattribs={'layer': 'ROOMS', 'lineweight': 20, 'color': 4}
                                )
                                
                                # 添加房间名称文本
                                if "Name" in room and "AnnotationPoint" in room:
                                    name = room.get("Name", "")
                                    annotation_point = room.get("AnnotationPoint", {})
                                    
                                    space.add_text(
                                        name,
                                        dxfattribs={
                                            'layer': 'TEXT',
                                            'height': 100 * SCALE,
                                            'color': 7,
                                            'insert': (annotation_point.get("x", 0) * SCALE, annotation_point.get("y", 0) * SCALE)
                                        }
                                    )
                                
                                building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制房间失败: {e}")
                            
        # 绘制楼梯（如果有）
        elevators = construction.get("Elevators", [])
        if elevators:
            print(f"  - 绘制电梯: {len(elevators)} 个电梯")
            for elevator in elevators:
                if "Elevator" in elevator:
                    elevator_lines = elevator.get("Elevator", [])
                    for line in elevator_lines:
                        start_point = line.get("StartPoint", {})
                        end_point = line.get("EndPoint", {})
                        
                        if start_point and end_point:
                            try:
                                # 绘制电梯线
                                space.add_line(
                                    (start_point.get("x", 0) * SCALE, start_point.get("y", 0) * SCALE),
                                    (end_point.get("x", 0) * SCALE, end_point.get("y", 0) * SCALE),
                                    dxfattribs={'layer': 'WALLS', 'lineweight': 35, 'color': 5}
                                )
                                building_elements_drawn += 1
                            except Exception as e:
                                print(f"  ⚠️ 绘制电梯线失败: {e}")
        
    # 旧格式：直接包含各种元素
    else:
        # 绘制墙体
        if "walls" in floor_data:
            print(f"  - 绘制墙体: {len(floor_data['walls'])} 个墙体")
            for wall in floor_data["walls"]:
                if "contourPoints" in wall:
                    points = wall["contourPoints"]
                    if len(points) >= 2:
                        # 创建多段线表示墙体
                        try:
                            polyline = space.add_lwpolyline(
                                [(p["x"] * SCALE, p["y"] * SCALE) for p in points],
                                dxfattribs={'layer': 'WALLS', 'lineweight': 50}  # 增加线宽
                            )
                            building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制墙体失败: {e}")
        
        # 绘制门
        if "doors" in floor_data:
            print(f"  - 绘制门: {len(floor_data['doors'])} 个门")
            for door in floor_data["doors"]:
                if "contourPoints" in door:
                    points = door["contourPoints"]
                    if len(points) >= 2:
                        # 创建多段线表示门
                        try:
                            polyline = space.add_lwpolyline(
                                [(p["x"] * SCALE, p["y"] * SCALE) for p in points],
                                dxfattribs={'layer': 'DOORS', 'lineweight': 50}  # 增加线宽
                            )
                            building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制门失败: {e}")
        
        # 绘制窗户
        if "windows" in floor_data:
            print(f"  - 绘制窗户: {len(floor_data['windows'])} 个窗户")
            for window in floor_data["windows"]:
                if "contourPoints" in window:
                    points = window["contourPoints"]
                    if len(points) >= 2:
                        # 创建多段线表示窗户
                        try:
                            polyline = space.add_lwpolyline(
                                [(p["x"] * SCALE, p["y"] * SCALE) for p in points],
                                dxfattribs={'layer': 'WINDOWS', 'lineweight': 50}  # 增加线宽
                            )
                            building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制窗户失败: {e}")
        
        # 绘制房间
        if "rooms" in floor_data:
            print(f"  - 绘制房间: {len(floor_data['rooms'])} 个房间")
            for room in floor_data["rooms"]:
                if "contourPoints" in room:
                    points = room["contourPoints"]
                    if len(points) >= 3:  # 房间需要至少3个点形成封闭区域
                        # 创建多段线表示房间
                        try:
                            polyline = space.add_lwpolyline(
                                [(p["x"] * SCALE, p["y"] * SCALE) for p in points],
                                dxfattribs={'layer': 'ROOMS', 'lineweight': 40}  # 增加线宽
                            )
                            
                            # 绘制房间边界填充
                            hatch = space.add_hatch(color=4)  # 添加填充，使房间更明显
                            hatch.paths.add_polyline_path(
                                [(p["x"] * SCALE, p["y"] * SCALE) for p in points],
                                is_closed=True
                            )
                            
                            # 如果房间有名称，添加文本标签
                            if "name" in room:
                                # 计算房间中心点
                                center_x = sum(p["x"] for p in points) / len(points)
                                center_y = sum(p["y"] for p in points) / len(points)
                                
                                # 添加房间名称文本
                                space.add_text(
                                    room["name"],
                                    dxfattribs={
                                        'layer': 'TEXT',
                                        'height': 300 * SCALE,  # 增大文本高度
                                        'insert': (center_x * SCALE, center_y * SCALE)
                                    }
                                )
                            building_elements_drawn += 1
                        except Exception as e:
                            print(f"  ⚠️ 绘制房间失败: {e}")
    
    print(f"  ✓ 共绘制了 {building_elements_drawn} 个建筑元素")

def draw_heating_elements(space, heating_data, scale):
    """绘制所有地暖元素"""
    # 检查是否是多楼层文件
    if "Floors" in heating_data:
        # 多楼层文件
        floors_data = heating_data.get("Floors", [])
    else:
        # 单楼层文件
        floors_data = [heating_data]
    
    # 遍历每个楼层数据
    for floor_data in floors_data:
        draw_heating_elements_for_floor(space, floor_data, scale)

def draw_heating_elements_for_floor(space, floor_data, scale):
    """绘制单个楼层的地暖元素"""
    # 绘制管道
    pipes = floor_data.get("Pipes", [])
    for pipe in pipes:
        # 获取管道点序列
        points = pipe.get("Points", [])
        if len(points) < 2:
            continue
        
        # 创建多段线
        polyline = space.add_lwpolyline(
            [(p["X"] * scale, p["Y"] * scale) for p in points],
            dxfattribs={'layer': 'HEATING_PIPES'}
        )
    
    # 绘制集水器
    collectors = floor_data.get("Collectors", [])
    for collector in collectors:
        # 获取集水器位置
        position = collector.get("Position", {})
        if position:
            x, y = position.get("X", 0), position.get("Y", 0)
            # 在集水器位置画一个圆
            space.add_circle(
                (x * scale, y * scale),
                radius=0.1,  # 适当的半径
                dxfattribs={'layer': 'COLLECTORS'}
            )
            # 添加集水器标签
            space.add_text(
                f"集水器 {collector.get('Id', '')}",
                dxfattribs={
                    'layer': 'TEXT',
                    'height': 0.15,
                    'insert': (x * scale, (y + 100) * scale)  # 稍微偏移一点
                }
            )

def draw_floor_with_offset(space, floor_data, scale, offset):
    """
    绘制单个楼层的建筑元素，应用位置偏移
    
    Args:
        space: 要绘制到的空间（模型空间或块）
        floor_data: 楼层数据
        scale: 坐标缩放因子
        offset: (x, y) 偏移量元组
    """
    if 'Construction' not in floor_data:
        return
        
    construction = floor_data['Construction']
    offset_x, offset_y = offset
    
    # 绘制房间
    for room in construction.get('Room', []):
        # 绘制房间边界
        for boundary in room.get('Boundary', []):
            start = boundary.get('StartPoint', {})
            end = boundary.get('EndPoint', {})
            if start and end:
                space.add_line(
                    (start['x'] * scale + offset_x, start['y'] * scale + offset_y),
                    (end['x'] * scale + offset_x, end['y'] * scale + offset_y),
                    dxfattribs={'layer': 'ROOMS'}
                )
        
        # 添加房间名称文本
        if 'AnnotationPoint' in room:  # 使用注释点作为文本位置
            point = room['AnnotationPoint']
            space.add_text(
                room.get('Name', ''),
                dxfattribs={
                    'layer': 'TEXT',
                    'height': 0.2,  # 文本高度也需要缩放
                    'insert': (point['x'] * scale + offset_x, point['y'] * scale + offset_y)
                }
            )
    
    # 绘制门
    for door in construction.get('DoorAndWindow', []):
        if door.get('Type') == '门':  # 只处理门
            base_line = door.get('BaseLine', {})
            if base_line:
                start = base_line.get('StartPoint', {})
                end = base_line.get('EndPoint', {})
                if start and end:
                    space.add_line(
                        (start['x'] * scale + offset_x, start['y'] * scale + offset_y),
                        (end['x'] * scale + offset_x, end['y'] * scale + offset_y),
                        dxfattribs={'layer': 'DOORS'}
                    )

def draw_heating_with_offset(space, heating_data, scale, offset):
    """
    绘制单个楼层的地暖元素，应用位置偏移
    
    Args:
        space: 要绘制到的空间（模型空间或块）
        heating_data: 地暖数据
        scale: 坐标缩放因子
        offset: (x, y) 偏移量元组
    """
    offset_x, offset_y = offset
    
    # 开始绘制地暖元素
    print(f"◆ 绘制地暖元素 (offset: {offset_x}, {offset_y})")
    
    # 添加原点标记
    try:
        # 在原点添加一个十字标记
        cross_size = 100 * scale
        space.add_line((0, 0), (cross_size, 0), dxfattribs={'color': 2, 'lineweight': 35})
        space.add_line((0, 0), (0, cross_size), dxfattribs={'color': 2, 'lineweight': 35})
        space.add_text(
            "原点(0,0)",
            dxfattribs={
                'layer': 'TEXT',
                'height': 50 * scale,
                'color': 7,
                'insert': (cross_size/2, cross_size/2)
            }
        )
        print(f"  ✓ 添加原点标记")
    except Exception as e:
        print(f"  ⚠️ 添加原点标记失败: {e}")
    
    # 绘制管道
    pipes = heating_data.get("Pipes", [])
    print(f"  - 管道数量: {len(pipes)}")
    
    for pipe in pipes:
        # 获取管道点序列
        points = pipe.get("Points", [])
        if len(points) < 2:
            continue
        
        # 创建多段线（应用偏移）
        try:
            polyline = space.add_lwpolyline(
                [(p["X"] * scale + offset_x, p["Y"] * scale + offset_y) for p in points],
                dxfattribs={'layer': 'HEATING_PIPES', 'lineweight': 25}
            )
            print(f"  ✓ 成功绘制管道，共 {len(points)} 个点")
        except Exception as e:
            print(f"  ✗ 绘制管道失败: {e}")
    
    # 检查是否有CollectorCoils数据
    collector_coils = heating_data.get("CollectorCoils", [])
    if collector_coils:
        print(f"  - 使用CollectorCoils数据: {len(collector_coils)} 个集水器线圈")
        extracted_collectors = []
        
        for collector_coil in collector_coils:
            collector_name = collector_coil.get("CollectorName", "未知")
            print(f"  - 集水器 {collector_name} 的线圈")
            
            # 将集水器添加到提取列表，尝试从线圈数据中提取位置
            collector_position = None
            
            # 检查集水器线圈数据结构
            if isinstance(collector_coil, dict):
                print(f"  - 集水器线圈数据键: {list(collector_coil.keys())}")
                
                # 检查Loops字段
                loops = collector_coil.get("Loops", None)
                if loops is not None:
                    if isinstance(loops, list):
                        print(f"    - Loops是列表: {len(loops)} 个元素")
                        # 处理loops列表
                        for loop in loops:
                            # 这里处理loops列表的每一项
                            pass
                    elif isinstance(loops, dict):
                        print(f"    - Loops是字典: {list(loops.keys())}")
                        # 这里处理loops字典
                        pass
                    else:
                        print(f"    - Loops是其他类型: {type(loops)}")
                
                # 检查CoilLoops字段
                coil_loops = collector_coil.get("CoilLoops", None)
                if coil_loops is not None:
                    print(f"    - 检查字段 'CoilLoops': {type(coil_loops)}")
                    if isinstance(coil_loops, list):
                        print(f"    - 处理CoilLoops: {len(coil_loops)} 项")
                        
                        # 获取第一个CoilLoop作为示例
                        if coil_loops:
                            first_coil_loop = coil_loops[0]
                            if isinstance(first_coil_loop, dict):
                                print(f"    - CoilLoop 示例数据键: {list(first_coil_loop.keys())}")
                                
                                # 尝试从CoilLoop中获取集水器位置 (例如，可能从起点位置推断)
                                if not collector_position and "Path" in first_coil_loop:
                                    path = first_coil_loop["Path"]
                                    if isinstance(path, list) and path:
                                        print(f"    - 找到Path字段: {type(path)}")
                                        print(f"    - Path包含 {len(path)} 个点")
                                        
                                        # 尝试从第一个Path点获取集水器位置
                                        first_path_item = path[0]
                                        print(f"    - 第一个Path点: {first_path_item}")
                                        
                                        if isinstance(first_path_item, dict):
                                            if "StartPoint" in first_path_item:
                                                start_point = first_path_item["StartPoint"]
                                                if isinstance(start_point, dict) and "x" in start_point and "y" in start_point:
                                                    # 使用起点作为集水器位置
                                                    collector_position = {
                                                        "X": float(start_point["x"]),
                                                        "Y": float(start_point["y"])
                                                    }
                                                    print(f"    ✓ 从Path起点提取集水器位置: ({collector_position['X']}, {collector_position['Y']})")
                                        
                                        # 绘制Path中的线段，使用更大的线宽
                                        for path_item in path:
                                            if isinstance(path_item, dict):
                                                if "StartPoint" in path_item and "EndPoint" in path_item:
                                                    start_point = path_item["StartPoint"]
                                                    end_point = path_item["EndPoint"]
                                                    
                                                    if (isinstance(start_point, dict) and "x" in start_point and "y" in start_point and
                                                        isinstance(end_point, dict) and "x" in end_point and "y" in end_point):
                                                        
                                                        try:
                                                            x1, y1 = float(start_point["x"]), float(start_point["y"])
                                                            x2, y2 = float(end_point["x"]), float(end_point["y"])
                                                            
                                                            # 创建线段（应用偏移和缩放），增加线宽
                                                            space.add_line(
                                                                (x1 * scale + offset_x, y1 * scale + offset_y),
                                                                (x2 * scale + offset_x, y2 * scale + offset_y),
                                                                dxfattribs={'layer': 'HEATING_PIPES', 'lineweight': 30}
                                                            )
                                                            print(f"    ✓ 成功绘制线段从 ({x1},{y1}) 到 ({x2},{y2})")
                                                        except Exception as e:
                                                            print(f"    ✗ 绘制线段失败: {e}")
            
            # 如果找到了集水器位置，加入提取列表
            if collector_position:
                extracted_collectors.append({
                    "Id": collector_name,
                    "Position": collector_position
                })
                
        # 使用提取的集水器信息替代原始集水器列表
        collectors = extracted_collectors
        print(f"  - 从CollectorCoils中提取了 {len(collectors)} 个集水器")
    else:
        # 使用原始集水器信息
        collectors = heating_data.get("Collectors", [])
        print(f"  - 使用原始Collectors数据: {len(collectors)} 个集水器")
    
    # 绘制集水器，使用更大、更明显的图形
    print(f"  - 绘制 {len(collectors)} 个集水器")
    for collector in collectors:
        # 获取集水器位置
        position = collector.get("Position", {})
        if position:
            try:
                x, y = position.get("X", 0), position.get("Y", 0)
                # 在集水器位置画一个圆（应用偏移），增大半径
                space.add_circle(
                    (x * scale + offset_x, y * scale + offset_y),
                    radius=1.0,  # 增大半径，更容易看见
                    dxfattribs={'layer': 'COLLECTORS', 'lineweight': 35}
                )
                # 添加集水器标签（应用偏移），增大文字
                space.add_text(
                    f"集水器 {collector.get('Id', '')}",
                    dxfattribs={
                        'layer': 'TEXT',
                        'height': 1.5,  # 增大文字高度
                        'color': 6,     # 设置颜色
                        'insert': (x * scale + offset_x, (y + 3) * scale + offset_y)  # 调整偏移
                    }
                )
                print(f"    ✓ 成功绘制集水器 {collector.get('Id', '')}")
            except Exception as e:
                print(f"    ✗ 绘制集水器失败: {e}")

def get_available_json_files():
    """获取data目录下所有可用的AR设计JSON文件"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("pypipeline/data")  # 尝试备选路径
    return sorted([f for f in data_dir.glob("ARDesign*.json")])

def main():
    """测试DXF导出功能"""
    print("\n=== DXF导出工具 ===")
    try:
        output_file = export_to_dxf("data/ARDesign02.json", "output/HeatingDesign_All_Floors.json")
        print(f"\n✅ DXF文件已成功导出至: {output_file}")
    except Exception as e:
        print(f"\n❌ 导出过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 