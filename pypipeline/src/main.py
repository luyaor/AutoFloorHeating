import numpy as np
from core import partition
from pathlib import Path
from tools import dxf_export
from tools import visualization_data
from pipeline import cactus_solver
from pipeline import convert_to_heating_design
import json
import os

def get_available_json_files(file_type="design"):
    """Get list of available JSON files in the example directory
    
    Args:
        file_type: 文件类型，可选值为 "design"（设计文件）或 "input"（输入数据文件）
    Returns:
        Selected file path
    """
    example_dir = Path("data")
    if file_type == "design":
        # AR设计文件
        return sorted([f.name for f in example_dir.glob("ARDesign*.json")])
    else:
        # 输入数据文件
        return sorted([f.name for f in example_dir.glob("inputData*.json")])

def select_input_file(file_type="design"):
    """
    Interactive selection of input file
    
    Args:
        file_type: 文件类型，可选值为 "design"（设计文件）或 "input"（输入数据文件）
    Returns:
        Selected file path
    """
    available_files = get_available_json_files(file_type)
    if not available_files:
        raise FileNotFoundError(f"No {file_type} JSON files found in data directory")
        
    print(f"\n🔷 可用的{file_type}文件:")
    for fname in available_files:
        print(f"  @{fname}")
    
    default_file = "ARDesign01.json" if file_type == "design" else "inputData01.json"
    
    while True:
        choice = input(f"\n🔷 请选择{file_type}文件 [@{default_file}]: ").strip()
        if not choice:
            return os.path.join("data", default_file)
            
        if choice.startswith('@'):
            filename = choice[1:]  # Remove @ prefix
            if filename in available_files:
                return os.path.join("data", filename)
        print("❌ 无效的选择，请重试")

def display_input_info(design_data, input_data):
    """
    显示输入数据的详细信息
    
    Args:
        design_data: 设计数据字典
        input_data: 输入参数数据字典
    """
    print("\n📊 建筑信息:")
    print(f"  建筑名称: {design_data.get('WebParam', {}).get('Name', '未知')}")
    print(f"  建筑地址: {design_data.get('WebParam', {}).get('Address', '未知')}")
    
    # 打印输入数据的基本信息
    print("\n📊 输入参数信息:")
    web_data = input_data.get('WebData', {})
    assist_data = input_data.get('AssistData', {})
    
    # 打印集水器信息
    print("\n🔹 集水器信息:")
    for floor in assist_data.get('Floor', []):
        if 'Construction' in floor and floor['Construction']:
            collectors = floor['Construction'].get('AssistCollector', [])
            if collectors:
                print(f"\n  楼层 {floor['Name']} (共{len(collectors)}个集水器):")
                for idx, collector in enumerate(collectors, 1):
                    location = collector['Location']
                    print(f"    {idx}. 位置: ({location['x']:.2f}, {location['y']:.2f}, {location['z']:.2f})")
                    if 'Borders' in collector:
                        borders = collector['Borders']
                        print(f"       边界点数: {len(borders)}个")
                        # 打印边界框的大小
                        if borders:
                            x_coords = []
                            y_coords = []
                            for border in borders:
                                x_coords.extend([border['StartPoint']['x'], border['EndPoint']['x']])
                                y_coords.extend([border['StartPoint']['y'], border['EndPoint']['y']])
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            print(f"       边界框大小: {width:.2f}×{height:.2f}mm")
    
    # 打印基本参数
    print("\n🔹 基本参数:")
    print(f"  不平衡率: {web_data.get('ImbalanceRatio', '未知')}%")
    print(f"  连接管间距: {web_data.get('JointPipeSpan', '未知')}mm")
    print(f"  密集区墙距: {web_data.get('DenseAreaWallSpan', '未知')}mm")
    print(f"  密集区管距: {web_data.get('DenseAreaSpanLess', '未知')}mm")
    
    # 打印环路间距设置
    loop_spans = web_data.get('LoopSpanSet', [])
    if loop_spans:
        print("\n🔹 环路间距设置:")
        for span in loop_spans:
            print(f"  - {span['TypeName']}:")
            print(f"    最小间距: {span['MinSpan']}mm")
            print(f"    最大间距: {span['MaxSpan']}mm")
            print(f"    曲率: {span['Curvity']}")
    
    # 打印障碍物间距设置
    obs_spans = web_data.get('ObsSpanSet', [])
    if obs_spans:
        print("\n🔹 障碍物间距设置:")
        for span in obs_spans:
            print(f"  - {span['ObsName']}:")
            print(f"    最小间距: {span['MinSpan']}mm")
            print(f"    最大间距: {span['MaxSpan']}mm")
    
    # 打印入户管间距设置
    delivery_spans = web_data.get('DeliverySpanSet', [])
    if delivery_spans:
        print("\n🔹 入户管间距设置:")
        for span in delivery_spans:
            print(f"  - {span['ObsName']}:")
            print(f"    最小间距: {span['MinSpan']}mm")
            print(f"    最大间距: {span['MaxSpan']}mm")
    
    # 打印管道间距设置
    pipe_spans = web_data.get('PipeSpanSet', [])
    if pipe_spans:
        print("\n🔹 管道间距设置:")
        for span in pipe_spans[:3]:  # 只显示前3个示例
            print(f"  - {span['LevelDesc']}-{span['FuncName']}-{','.join(span['Directions'])}:")
            print(f"    外墙数: {span['ExterWalls']}")
            print(f"    管距: {span['PipeSpan']}mm")
        if len(pipe_spans) > 3:
            print(f"    ... 等共{len(pipe_spans)}条设置")
    
    # 打印弹性间距设置
    elastic_spans = web_data.get('ElasticSpanSet', [])
    if elastic_spans:
        print("\n🔹 弹性间距设置:")
        for span in elastic_spans:
            print(f"  - {span['FuncName']}:")
            print(f"    优先间距: {span['PriorSpan']}mm")
            print(f"    最小间距: {span['MinSpan']}mm")
            print(f"    最大间距: {span['MaxSpan']}mm")
    
    # 打印功能房间设置
    func_rooms = web_data.get('FuncRooms', [])
    if func_rooms:
        print("\n🔹 功能房间设置:")
        for room in func_rooms:
            print(f"  - {room['FuncName']}:")
            print(f"    包含: {', '.join(room['RoomNames'])}")
            
    # 显示楼层信息
    for floor in design_data.get("Floor", []):
        print(f"\n📊 楼层: {floor['Name']}")
        print(f"  层高: {floor['LevelHeight']}mm")
        
        if 'Construction' not in floor or not floor['Construction']:
            continue
            
        # 打印房间信息
        rooms = floor["Construction"].get("Room", [])
        print(f"\n📊 房间信息 (共{len(rooms)}个):")
        for room in rooms:
            print(f"  - {room['Name']:<10} (面积: {room['Area']}㎡, 类型: {room['NameType']})")
            
        # 打印门的信息
        doors = [d for d in floor["Construction"].get("DoorAndWindow", []) if d.get("Type") == "门"]
        print(f"\n📊 门的信息 (共{len(doors)}个):")
        for door in doors:
            print(f"  - {door['Name']:<10} (类型: {door.get('DoorType', '普通')}, 尺寸: {door['Size']['Width']}×{door['Size']['Height']}mm)")
        
        # 打印集水器信息
        collectors = floor["Construction"].get("AssistCollector", [])
        if collectors:
            print(f"\n📊 集水器信息 (共{len(collectors)}个):")
            for collector in collectors:
                location = collector["Location"]
                print(f"  - 位置: ({location['x']:.2f}, {location['y']:.2f}, {location['z']:.2f})")

def is_point_in_polygon(point, polygon):
    """
    判断点是否在多边形内部
    使用射线法 (Ray Casting Algorithm)
    
    Args:
        point: (x, y) 坐标元组
        polygon: 多边形顶点坐标列表 [(x1,y1), (x2,y2), ...]
    
    Returns:
        bool: 点是否在多边形内
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def calculate_point_to_edge_projection(point, edge_start, edge_end):
    """
    计算点到线段的投影点
    
    Args:
        point: (x, y) 坐标元组
        edge_start: 线段起点 (x, y) 坐标元组
        edge_end: 线段终点 (x, y) 坐标元组
        
    Returns:
        tuple: 投影点坐标 (x, y), 到线段的距离
    """
    x, y = point
    x1, y1 = edge_start
    x2, y2 = edge_end
    
    # 计算线段长度的平方
    line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
    
    # 如果线段长度为0，返回起点和点到起点的距离
    if line_length_sq == 0:
        return edge_start, ((x - x1)**2 + (y - y1)**2)**0.5
    
    # 计算投影比例 t
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
    
    # 计算投影点坐标
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    # 计算点到投影点的距离
    distance = ((x - proj_x)**2 + (y - proj_y)**2)**0.5
    
    return (proj_x, proj_y), distance

def find_nearest_edge_projection(point, polygon):
    """
    找到点到多边形所有边的最近投影点
    
    Args:
        point: (x, y) 坐标元组
        polygon: 多边形顶点坐标列表 [(x1,y1), (x2,y2), ...]
        
    Returns:
        tuple: (投影点坐标 (x, y), 最小距离, 边的索引)
    """
    min_distance = float('inf')
    nearest_projection = None
    nearest_edge_index = -1
    
    for i in range(len(polygon)):
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % len(polygon)]
        
        projection, distance = calculate_point_to_edge_projection(point, edge_start, edge_end)
        
        if distance < min_distance:
            min_distance = distance
            nearest_projection = projection
            nearest_edge_index = i
    
    return nearest_projection, min_distance, nearest_edge_index

def area_partition(key, floor_data, points, num_x, num_y, collectors):
    # 将points从列表转换为元组列表以便于后续处理
    points_tuple = [(p[0], p[1]) for p in points]
    
    # 1. 只保留当前图形区域内的集水器
    filtered_collectors = []
    for collector in collectors:
        # 将集水器坐标转换为米单位
        # collector_point = (collector['Location']['x']/100, collector['Location']['y']/100)
        collector_point = (collector['Location']['x'], collector['Location']['y'])
        
        # 检查集水器是否在当前多边形区域内
        if is_point_in_polygon(collector_point, points_tuple):
            # 2. 计算到最近边的投影
            projection, distance, edge_index = find_nearest_edge_projection(collector_point, points_tuple)
            
            # 添加集水器及其投影信息
            collector_data = {
                'location': {
                    'x': collector['Location']['x']/100,  # 转换为米
                    'y': collector['Location']['y']/100,
                    'z': collector['Location']['z']/100
                },
                'borders': [
                    {
                        'start': {
                            'x': border['StartPoint']['x']/100,
                            'y': border['StartPoint']['y']/100
                        },
                        'end': {
                            'x': border['EndPoint']['x']/100,
                            'y': border['EndPoint']['y']/100
                        }
                    }
                    for border in collector.get('Borders', [])
                ] if 'Borders' in collector else [],
                'projection': {
                    'point': {
                        'x': projection[0],
                        'y': projection[1]
                    },
                    'distance': distance,
                    'edge_index': edge_index
                }
            }
            filtered_collectors.append(collector_data)
    
    # 2. 如果当前范围内没有集水器，则跳过这个方法
    if not filtered_collectors:
        print(f"\n👮 当前区域 {key} 没有集水器，跳过处理...")
        return None, None, None, None
    
    # 保存分区输入数据
    partition_input = {
        'points': points,
        'num_x': num_x,
        'num_y': num_y,
        'floor_name': floor_data['Name'],
        'collectors': filtered_collectors
    }


    output_dir = Path('output')
    partition_input_file = output_dir / f'floor_{floor_data["Name"]}_{key}_partition_input.json'
    with open(partition_input_file, 'w', encoding='utf-8') as f:
        json.dump(partition_input, f, indent=2, ensure_ascii=False)
    print(f"\n💾 分区输入数据已保存至: {partition_input_file}")

    print("\n🔷 开始执行空间分区...")

    # (TODO) hardcode.....need improve
    #----------
    # partition_input_file = output_dir / "1_polygon_group_1_partition_input.json"
    #----------

    partition_input = load_partition_input(partition_input_file)

    inputp = partition_input['points']
    inputp = [(round(pt[0], 2), round(pt[1], 2)) for pt in inputp]

    collector = partition_input['collectors'][0]["projection"]["point"]
    collector_pt = (collector['x'], collector['y'])
    final_polygons, allp, new_region_info, wall_path, destination_pt = partition.partition_work(partition_input['points'], 
                                                                                          num_x=partition_input['num_x'], 
                                                                                          num_y=partition_input['num_y'],
                                                                                          collector=collector_pt)
    
    # (TODO) hardcode.....need improve
    #----------
    # start_point = allp.index(inputp[0])
    # new_region_info = [(x[0], x[1] + 1) for x in new_region_info]
    # st_in_area_cnt = 0
    # for x in new_region_info:
    #     if start_point in x[0]:
    #         st_in_area_cnt += 1
    #         x = (x[0], 0)
    # assert (st_in_area_cnt == 1)
    #----------


    print("\n📊 分区结果:")
    print(f"  - 分区数量: {len(final_polygons)}")
    print(f"  - 分区点数: {len(allp)}")
    print(f"  - 区域信息: {len(new_region_info)}个区域")
    print(f"  - 起点位置: {destination_pt}")
    

    print("\n✅ 分区计算完成...")

    # # 绘制分区结果
    # partition.plot_polygons(final_polygons, nat_lines=nat_lines, 
    #                      title="Space Partition Result", global_points=allp)
    # 准备输入数据
    # seg_pts = [(x[0]/100, x[1]/100) for x in allp]  # 从原始数据转换并缩放
    seg_pts = [(x[0], x[1]) for x in allp]
    regions = [(r[0], r[1]) for r in new_region_info]  # 从原始数据转换
    # Filter out regions where r[1] == -1
    # regions = [(r[0], r[1]) for r in regions if r[1] != -1]

    return seg_pts, regions, wall_path, destination_pt

def get_floor_collectors(floor_data, input_data):
    """
    获取指定楼层的集水器列表
    
    Args:
        floor_data: 楼层数据
        input_data: 输入参数数据
        
    Returns:
        tuple: (是否有集水器(bool), 集水器列表(list))
    """
    floor_name = floor_data['Name']
    
    # 在input_data中查找当前楼层的集水器信息
    for floor_info in input_data['AssistData']['Floor']:
        if floor_info['Name'] == floor_name:
            if ('Construction' in floor_info and 
                floor_info['Construction'] and 
                'AssistCollector' in floor_info['Construction'] and 
                floor_info['Construction']['AssistCollector']):
                return True, floor_info['Construction']['AssistCollector']
            break
    
    return False, []

def process_pipeline(key, floor_data, seg_pts, regions, wall_path, start_point):
    # 保存中间数据
    intermediate_data = {
        'floor_name': floor_data['Name'],
        'seg_pts': seg_pts,
        'regions': regions,  
        'wall_path': wall_path,
        'destination_pt': start_point,
        'pipe_interval': 250
    }

    output_dir = Path('output')
    output_file = output_dir / f'{floor_data["Name"]}_{key}_intermediate.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 中间数据已保存至: {output_file}")

    # output_file = output_dir / 'cases/case8_intermediate.json'
    # output_file = output_dir / '1_polygon_group_1_intermediate.json'
    pipe_pt_seq = cactus_solver.solve_pipeline(output_file)
    return pipe_pt_seq

def generate_design_files(all_pipe_data, design_data, input_data):
    """
    生成最终的地暖设计文件
    
    Args:
        all_pipe_data: 包含所有楼层管道数据的列表
        design_data: 原始设计数据
        input_data: 输入参数数据
        
    Returns:
        Path: 生成的设计文件路径，多楼层时返回合并文件路径，单楼层时返回楼层文件路径
    """
    if not all_pipe_data:
        print("\n👮 没有找到有效的管道布线数据，未生成设计文件")
        return None
        
    print("\n🔷 开始生成最终设计文件...")
    output_dir = Path('output')
    
    # 保存最后一个生成的文件路径
    last_file_path = None
    
    # 为每个楼层单独生成设计文件
    for floor_info in all_pipe_data:
        # 收集当前楼层所有区域的管道布线数据
        floor_pipe_pt_seq = []
        for area_info in floor_info['pipe_data']:
            floor_pipe_pt_seq.extend(area_info['pipe_pt_seq'])
        
        # 为当前楼层生成设计数据
        floor_design_data = convert_to_heating_design.convert_pipe_pt_seq_to_heating_design(
            floor_pipe_pt_seq,
            level_name=floor_info['floor_name'],
            level_no=floor_info['level_no'],
            level_desc=floor_info['level_desc'],
            house_name=design_data.get('WebParam', {}).get('Id', ""),  # 从设计文件中获取Id作为house_name，如果获取不到则使用空字符串
            curvity=100,
            input_data=input_data
        )
        
        # 为每个楼层保存单独的设计文件
        floor_out_file = output_dir / f"HeatingDesign_{floor_info['floor_name']}.json"
        convert_to_heating_design.save_design_to_json(floor_design_data, floor_out_file)
        print(f"\n✅ {floor_info['floor_name']}楼层的地暖设计数据已保存到：{floor_out_file}")
        
        # 更新最后生成的文件路径
        last_file_path = floor_out_file
    
    # 如果需要，还可以生成一个合并版本的文件（可选）
    if len(all_pipe_data) > 1:
        # 创建包含所有楼层数据的列表
        all_floors_data = []
        for floor_info in all_pipe_data:
            # 收集当前楼层所有区域的管道布线数据
            floor_pipe_pt_seq = []
            for area_info in floor_info['pipe_data']:
                floor_pipe_pt_seq.extend(area_info['pipe_pt_seq'])
            
            floor_design_data = convert_to_heating_design.convert_pipe_pt_seq_to_heating_design(
                floor_pipe_pt_seq,
                level_name=floor_info['floor_name'],
                level_no=floor_info['level_no'],
                level_desc=floor_info['level_desc'],
                house_name=design_data.get('WebParam', {}).get('Id', ""),
                curvity=100,
                input_data=input_data
            )
            all_floors_data.append(floor_design_data)
        
        # 保存合并版本的文件
        merged_out_file = output_dir / "HeatingDesign_All_Floors.json"
        # 这里我们创建一个包含所有楼层数据的字典
        merged_design_data = {
            "BuildingName": design_data.get('ARGeneralInfo', {}).get('BuildingName', ""),
            "Floors": all_floors_data
        }
        convert_to_heating_design.save_design_to_json(merged_design_data, merged_out_file)
        print(f"\n✅ 合并版本的多楼层地暖设计数据已保存到：{merged_out_file}")
        
        # 多楼层时，优先返回合并文件路径
        return merged_out_file
        
    # 单楼层时，返回最后一个生成的文件路径
    return last_file_path

def load_solver_params(json_file):
    """从JSON文件加载求解器参数"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_partition_input(json_file):
    """从JSON文件加载分区输入数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_level_no(floor_name):
    """
    从楼层名称中提取楼层编号
    
    Args:
        floor_name: 楼层名称，如"1"、"2F"等
        
    Returns:
        int: 楼层编号，默认为1
    """
    level_no = 1  # 默认楼层编号
    try:
        # 尝试从楼层名称中提取数字
        if floor_name.endswith('F'):
            level_no = int(floor_name.strip('F'))
        else:
            # 尝试直接将楼层名称转换为整数
            level_no = int(floor_name)
    except ValueError:
        # 如果转换失败，使用默认值1
        level_no = 1
    
    return level_no

def run_pipeline(num_x: int = 3, num_y: int = 3):
    """
    运行管道布线的完整流程
    
    Args:
        num_x: 网格x方向划分数
        num_y: 网格y方向划分数
    """
    # 0. 处理输入数据
    print("🔷 正在处理输入数据...")
    
    # 选择设计文件
    design_json_path = select_input_file("design")
    print(f"\n✅ 成功读取设计文件: {design_json_path}")
    
    
    # 选择输入数据文件
    input_json_path = select_input_file("input")
    print(f"\n✅ 成功读取输入数据文件: {input_json_path}")
    
    # 加载设计JSON数据显示详细信息
    with open(design_json_path, 'r', encoding='utf-8') as f:
        design_data = json.load(f)
    
    # 加载输入数据JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
        
    # 显示输入数据信息
    display_input_info(design_data, input_data)
    
    print("\n🔷 按任意键继续处理数据...")
    input()
    
    # 创建用于收集所有管道布线数据的结构
    all_pipe_data = []
    
    # data = visualization_data.load_json_data(design_json_path)
    # 遍历每个楼层, 绘制原始图像, 提取多边形信息, 执行分区, 执行管道布线
    for floor_data in design_data["Floor"]:
        # 检查当前楼层是否有集水器
        has_collector, collectors = get_floor_collectors(floor_data, input_data)
        
        if not has_collector:
            print(f"\n👮 楼层 {floor_data['Name']} 没有集水器，跳过处理...")
            continue
            
        print(f"\n📊 开始处理楼层: {floor_data['Name']}")
        print(f"✅ 检测到 {len(collectors)} 个集水器，继续处理...")
        
        processed_data, polygons = visualization_data.process_ar_design(floor_data)
        # print("\n✅ 原始图像绘制完成，按任意键继续...")
        # # 绘制原始数据
        # input()
        # visualization_data.plot_comparison(processed_data, polygons, collectors=collectors)
        # continue

        print("\n📊 提取的多边形信息:")
        
        # 收集当前楼层的所有管道布线数据
        floor_pipe_data = []
        
        for key, points in polygons.items():
            print(f"\n📊 当前处理楼层: {floor_data['Name']}")
            if not key.startswith("polygon"):
                continue

            # points = [(x[0]/100, x[1]/100) for x in points]

            print(f"🔷 当前处理多边编号: {key}")

            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)

            # 1. 执行分区
            seg_pts, regions, wall_path, start_point = area_partition(key, floor_data, points, num_x, num_y, collectors)
            
            # 如果没有集水器或分区处理失败，跳过当前多边形
            if seg_pts is None:
                print(f"\n👮 跳过当前多边形 {key} 的管道布线...")
                continue
                
            print(f"🔷 分区结果: {regions}")


            # 2. 执行管道布线
            print("\n🔷 开始执行管道布线...")

            try:
                pipe_pt_seq = process_pipeline(key, floor_data, seg_pts, regions, wall_path, start_point)
            except Exception as e:
                print(f"\n❌ 管道布线失败: {e}")
                import traceback
                print("\n🔴 错误堆栈信息:")
                print(traceback.format_exc())
                pipe_pt_seq = [[np.array([0, 0]), np.array([100, 100])]]
                # continue

            # 可视化管道布线结果
            # from plot_pipe_data import plot_pipe_pt_seq
            # plot_pipe_pt_seq(pipe_pt_seq)
            
            # 收集当前区域的管道布线数据
            floor_pipe_data.append({
                'area_key': key,
                'pipe_pt_seq': pipe_pt_seq
            })
            # break
        
        # 收集当前楼层的数据
        if floor_pipe_data:
            # 提取楼层信息
            all_pipe_data.append({
                'floor_data': floor_data,
                'floor_name': floor_data['Name'],  # 保持原始楼层名称不变
                'level_no': get_level_no(floor_data['Name']),
                'level_desc': floor_data['Name'],
                'pipe_data': floor_pipe_data
            })
            
        print("\n✅ 楼层处理完成!")
        # break
    
    # 所有楼层和区域处理完毕，生成最终的设计文件
    heating_design_file = generate_design_files(all_pipe_data, design_data, input_data)
    
    # 导出DXF文件
    if heating_design_file:
        print("\n🔷 正在导出DXF文件...")
        dxf_file = dxf_export.export_to_dxf(design_json_path, input_json_path, heating_design_file)
        print(f"✅ DXF文件已导出至: {dxf_file}")
    else:
        print("\n⚠️ 未生成设计文件，跳过DXF导出")

    print("\n✅ 管道布线完成!")

def main():
    print(f"\n{'='*50}")
    print("🔷 管道布线系统")
    print('='*50)
    
    run_pipeline(num_x=3, num_y=3)

if __name__ == "__main__":
    main() 