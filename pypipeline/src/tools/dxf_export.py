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

def export_to_dxf(design_file: str, heating_design_file: str = None) -> str:
    """
    将AR设计文件和地暖设计文件导出为DXF格式
    
    Args:
        design_file: AR设计JSON文件路径
        heating_design_file: 地暖设计JSON文件路径，如果不指定则使用相同文件名但扩展名为.dxf
        
    Returns:
        生成的DXF文件路径
    """
    print("\n🔷 开始导出DXF文件...")
    print(f"  - 设计文件: {design_file}")
    if heating_design_file:
        print(f"  - 地暖设计文件: {heating_design_file}")
    else:
        print("  - 没有提供地暖设计文件")
    
    # 1. 加载AR设计JSON数据
    with open(design_file, 'r', encoding='utf-8') as f:
        design_data = json.load(f)
    
    # 如果提供了地暖设计文件，也加载它
    heating_data = None
    if heating_design_file and os.path.exists(heating_design_file):
        try:
            with open(heating_design_file, 'r', encoding='utf-8') as f:
                heating_data = json.load(f)
            print(f"\n✅ 成功加载地暖设计文件: {heating_design_file}")
        except Exception as e:
            print(f"\n⚠️ 加载地暖设计文件失败: {str(e)}")
    
    # 2. 确定输出文件路径
    if heating_design_file:
        # 使用地暖设计文件的名称作为基础，但扩展名改为.dxf
        output_file = str(Path(heating_design_file).with_suffix('.dxf'))
    else:
        # 使用原始设计文件名称
        output_file = str(Path(design_file).with_suffix('.dxf'))
    
    # 3. 创建新的DXF文档
    doc = ezdxf.new('R2010')  # 使用AutoCAD 2010格式
    
    # 4. 为不同类型的实体创建图层
    doc.layers.new('WALLS', dxfattribs={'color': 1})       # 蓝色
    doc.layers.new('DOORS', dxfattribs={'color': 2})       # 黄色
    doc.layers.new('ROOMS', dxfattribs={'color': 3})       # 绿色
    doc.layers.new('TEXT', dxfattribs={'color': 7})        # 白色
    doc.layers.new('HEATING_PIPES', dxfattribs={'color': 6})  # 紫红色 - 地暖管道
    doc.layers.new('COLLECTORS', dxfattribs={'color': 4})     # 青色 - 集水器
    
    # 坐标缩放因子
    scale = 0.001  # 将毫米转换为米
    
    # 5. 创建楼层数据映射
    floor_data_map = {}
    
    # 从设计数据获取楼层信息
    for floor in design_data.get('Floor', []):
        floor_name = floor.get('Name', '')
        if not floor_name or 'Construction' not in floor:
            continue
        
        floor_data_map[floor_name] = {
            'design': floor,
            'heating': None  # 先初始化为None，之后会填充
        }
    
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
    
    # 6. 为每个楼层创建单独的布局并绘制内容
    # 先在模型空间(Model Space)创建一个简单的索引
    msp = doc.modelspace()
    # msp.add_text(
    #     "本文件包含多个楼层设计图，请在布局(Layout)中查看各楼层",
    #     dxfattribs={
    #         'height': 0.5,
    #         'insert': (0, 0)
    #     }
    # )
    
    # 检查是否有楼层数据
    if not floor_data_map:
        print("\n⚠️ 没有找到有效的楼层数据")
        # 在模型空间至少添加一些内容
        draw_building_elements(msp, design_data, scale)
        if heating_data:
            draw_heating_elements(msp, heating_data, scale)
    else:
        print(f"\n✅ 找到 {len(floor_data_map)} 个楼层")
        
        # 为每个楼层创建独立的模型空间内容块
        # 在不同的位置放置不同楼层的内容，确保它们不会重叠
        floor_positions = {}
        
        # 计算每个楼层内容的位置偏移，确保不同楼层在模型空间中不会重叠
        x_offset = 0
        for floor_idx, (floor_name, data) in enumerate(floor_data_map.items()):
            # 每个楼层水平偏移1000单位（可根据需要调整）
            floor_offset = (x_offset * 1000, 0)
            floor_positions[floor_name] = floor_offset
            
            # 创建一个楼层名称标记，便于在模型空间中识别
            msp.add_text(
                f"楼层: {floor_name}",
                dxfattribs={
                    'height': 1.0,
                    'insert': (floor_offset[0], floor_offset[1] - 5)
                }
            )
            
            # 在指定位置绘制该楼层的所有内容
            if data['design']:
                draw_floor_with_offset(msp, data['design'], scale, floor_offset)
            
            # 绘制地暖元素（如果有）
            if data['heating']:
                draw_heating_with_offset(msp, data['heating'], scale, floor_offset)
            
            x_offset += 1
        
        # 为每个楼层创建布局并设置视口
        for floor_name, data in floor_data_map.items():
            print(f"\n🔷 正在创建楼层 [{floor_name}] 的设计图...")
            
            # 创建新的布局
            layout = doc.layouts.new(f"楼层-{floor_name}")
            layout.page_setup(size=(420, 297), margins=(10, 10, 10, 10), units='mm')  # A3横向
            
            # 添加标题信息
            title_y = 287  # 页面顶部
            layout.add_text(
                f"楼层: {floor_name}",
                dxfattribs={
                    'height': 5,
                    'insert': (210, title_y),
                    'halign': 1,  # 1 = CENTER (水平居中)
                    'style': 'Standard'
                }
            )
            
            # 创建视口(Viewport)来显示该楼层的内容
            viewport = layout.add_viewport(
                center=(210, 150),  # 页面中心
                size=(380, 250),    # 视口大小
                view_center_point=floor_positions[floor_name],  # 视图中心点设置为该楼层的偏移位置
                view_height=50      # 视图高度
            )
            
            # 配置视口：显示区域取决于内容的边界
            try:
                # 设置视口属性
                viewport.dxf.status = 1  # 设置为活动视口
                
                # 注意：有些属性在某些ezdxf版本中可能不支持，使用try/except来处理
                try:
                    viewport.dxf.view_target_point = (floor_positions[floor_name][0], floor_positions[floor_name][1], 0)
                except Exception as e:
                    print(f"⚠️ 设置视口目标点失败: {e}")
                
                try:
                    # 视图方向可能不被支持
                    viewport.dxf.view_direction = (0, 0, 1)  # 视图方向 (从上方看)
                except Exception as e:
                    print(f"⚠️ 设置视口方向失败: {e}")
                
                # 使用默认视图高度
                try:
                    viewport.dxf.view_height = 50
                except Exception as e:
                    print(f"⚠️ 设置默认视口高度失败: {e}")
                    
                try:
                    viewport.dxf.view_center_point = floor_positions[floor_name]
                except Exception as e:
                    print(f"⚠️ 设置默认视口中心点失败: {e}")
                
                try:
                    viewport.dxf.view_mode = 0  # 视图模式 (不包括网格等)
                except Exception as e:
                    print(f"⚠️ 设置视口模式失败: {e}")
                    
            except Exception as e:
                print(f"⚠️ 配置视口失败: {e}，但继续执行其余操作")
            
            # 添加楼层标题（在视口下方）
            layout.add_text(
                f"楼层平面图: {floor_name}",
                dxfattribs={
                    'height': 4,
                    'insert': (210, 30),
                    'halign': 1,  # 1 = CENTER (水平居中)
                    'style': 'Standard'
                }
            )
    
    # 7. 保存DXF文件
    doc.saveas(output_file)
    
    return output_file

def draw_building_elements(space, design_data, scale):
    """绘制建筑基本元素"""
    for floor in design_data.get('Floor', []):
        if 'Construction' not in floor:
            continue
        draw_floor_elements(space, floor, scale)

def draw_floor_elements(space, floor_data, scale):
    """绘制单个楼层的建筑元素"""
    if 'Construction' not in floor_data:
        return
        
    construction = floor_data['Construction']
    
    # 绘制房间
    for room in construction.get('Room', []):
        # 绘制房间边界
        for boundary in room.get('Boundary', []):
            start = boundary.get('StartPoint', {})
            end = boundary.get('EndPoint', {})
            if start and end:
                space.add_line(
                    (start['x'] * scale, start['y'] * scale),
                    (end['x'] * scale, end['y'] * scale),
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
                    'insert': (point['x'] * scale, point['y'] * scale)
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
                        (start['x'] * scale, start['y'] * scale),
                        (end['x'] * scale, end['y'] * scale),
                        dxfattribs={'layer': 'DOORS'}
                    )

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
    
    # 绘制管道
    pipes = heating_data.get("Pipes", [])
    for pipe in pipes:
        # 获取管道点序列
        points = pipe.get("Points", [])
        if len(points) < 2:
            continue
        
        # 创建多段线（应用偏移）
        polyline = space.add_lwpolyline(
            [(p["X"] * scale + offset_x, p["Y"] * scale + offset_y) for p in points],
            dxfattribs={'layer': 'HEATING_PIPES'}
        )
    
    # 绘制集水器
    collectors = heating_data.get("Collectors", [])
    for collector in collectors:
        # 获取集水器位置
        position = collector.get("Position", {})
        if position:
            x, y = position.get("X", 0), position.get("Y", 0)
            # 在集水器位置画一个圆（应用偏移）
            space.add_circle(
                (x * scale + offset_x, y * scale + offset_y),
                radius=0.1,  # 适当的半径
                dxfattribs={'layer': 'COLLECTORS'}
            )
            # 添加集水器标签（应用偏移）
            space.add_text(
                f"集水器 {collector.get('Id', '')}",
                dxfattribs={
                    'layer': 'TEXT',
                    'height': 0.15,
                    'insert': (x * scale + offset_x, (y + 100) * scale + offset_y)  # 稍微偏移一点
                }
            )

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
        # 导出DXF文件
        output_file = export_to_dxf("data/ARDesign02.json", "output/HeatingDesign_All_Floors.json")
        print(f"\n✅ DXF文件已成功导出至: {output_file}")
    except Exception as e:
        print(f"\n❌ 导出过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 