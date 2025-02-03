from cactus_solver import solve_pipeline
from core import partition
from pathlib import Path
from tools import dxf_export
from tools import visualization_data
import convert_to_heating_design
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
    
    default_file = "ARDesign02.json" if file_type == "design" else "inputData02.json"
    
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
        for span in pipe_spans:  # 只显示前3个示例
            print(f"  - {span['LevelDesc']}-{span['FuncName']}-{','.join(span['Directions'])}:")
            print(f"    外墙数: {span['ExterWalls']}")
            print(f"    管距: {span['PipeSpan']}mm")
        # if len(pipe_spans) > 3:
        #     print(f"    ... 等共{len(pipe_spans)}条设置")
    
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
    
    if False:
        # 导出DXF文件
        print("\n🔷 正在导出DXF文件...")
        dxf_file = dxf_export.export_to_dxf(design_json_path)
        print(f"✅ DXF文件已导出至: {dxf_file}")
    
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
    
    # data = visualization_data.load_json_data(design_json_path)
    # 遍历每个楼层, 绘制原始图像, 提取多边形信息, 执行分区, 执行管道布线
    for floor_data in design_data["Floor"]:
        # 检查当前楼层是否有集水器
        floor_name = floor_data['Name']
        has_collector = False
        collectors = []
        
        # 在input_data中查找当前楼层的集水器信息
        for floor_info in input_data['AssistData']['Floor']:
            if floor_info['Name'] == floor_name:
                if ('Construction' in floor_info and 
                    floor_info['Construction'] and 
                    'AssistCollector' in floor_info['Construction'] and 
                    floor_info['Construction']['AssistCollector']):
                    has_collector = True
                    collectors = floor_info['Construction']['AssistCollector']
                break
        
        if not has_collector:
            print(f"\n⚠️ 楼层 {floor_name} 没有集水器，跳过处理...")
            continue
            
        print(f"\n📊 开始处理楼层: {floor_name}")
        print(f"✅ 检测到 {len(collectors)} 个集水器，继续处理...")
        
        processed_data, polygons = visualization_data.process_ar_design(floor_data)
        # print("\n✅ 原始图像绘制完成，按任意键继续...")
        # # 绘制原始数据
        # input()
        # visualization_data.plot_comparison(processed_data, polygons, collectors=collectors)

        print("\n📊 提取的多边形信息:")
        for key, points in polygons.items():
            print(f"\n📊 当前处理楼层: {floor_data['Name']}")
            if key.startswith("polygon"):
                points = [(x[0]/100, x[1]/100) for x in points]

                # 保存分区输入数据
                partition_input = {
                    'points': points,
                    'num_x': num_x,
                    'num_y': num_y,
                    'floor_name': floor_data['Name'],
                    'collectors': [
                        {
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
                            ] if 'Borders' in collector else []
                        }
                        for collector in collectors
                    ]
                }

                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)

                partition_input_file = output_dir / 'partition_input.json'
                with open(partition_input_file, 'w', encoding='utf-8') as f:
                    json.dump(partition_input, f, indent=2, ensure_ascii=False)
                print(f"\n💾 分区输入数据已保存至: {partition_input_file}")

                # 1. 执行分区
                print("\n🔷 开始执行空间分区...")

                partition_input = load_partition_input(partition_input_file)

                final_polygons, nat_lines, allp, new_region_info, wall_path = partition.partition_work(partition_input['points'], 
                                                                                                      num_x=partition_input['num_x'], 
                                                                                                      num_y=partition_input['num_y'])

                print("\n📊 分区结果:")
                print(f"  - 分区数量: {len(final_polygons)}")
                print(f"  - 分区点数: {len(allp)}")
                print(f"  - 区域信息: {len(new_region_info)}个区域")

                print("\n✅ 分区计算完成...")

                # # 绘制分区结果
                # partition.plot_polygons(final_polygons, nat_lines=nat_lines, 
                #                      title="Space Partition Result", global_points=allp)


                # 2. 执行管道布线
                print("\n🔷 开始执行管道布线...")
                
                # 准备输入数据
                seg_pts = [(x[0]/100, x[1]/100) for x in allp]  # 从原始数据转换并缩放
                regions = [(r[0], r[1]) for r in new_region_info]  # 从原始数据转换

                # 保存中间数据
                intermediate_data = {
                    'floor_name': floor_data['Name'],
                    'seg_pts': seg_pts,
                    'regions': regions,  
                    'wall_path': wall_path
                }
                
                output_file = output_dir / 'intermediate_data.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

                print(f"\n💾 中间数据已保存至: {output_file}")
                
                output_file = output_dir / 'cases/case8_intermediate.json'
                pipe_pt_seq = solve_pipeline(output_file)
                # print(pipe_pt_seq)
                out_file = output_dir / "HeatingDesign_output.json"
                design_data = convert_to_heating_design.convert_pipe_pt_seq_to_heating_design(pipe_pt_seq, 
                                                        level_name="1F",
                                                        level_no=1,
                                                        level_desc="首层",
                                                        house_name="c1c37dc1a40f4302b6552a23cd1fd557",
                                                        curvity=100,
                                                        input_data=input_data)
                convert_to_heating_design.save_design_to_json(design_data, out_file)
                print(f"转换后的地暖设计数据已保存到：{out_file}")
                
                # 打印地暖设计数据详情
                print("\n📊 地暖设计数据详情 (HeatingCoil):")
                print(f"  楼层名称(LevelName): {design_data['LevelName']}")
                print(f"  楼号(LevelNo): {design_data['LevelNo']}")
                print(f"  楼层描述(LevelDesc): {design_data['LevelDesc']}")
                print(f"  户型编号(HouseName): {design_data['HouseName']}")
                
                # 打印伸缩缝信息
                expansions = design_data['Expansions']
                print(f"\n🔹 伸缩缝集合 (Expansions): {len(expansions)}条")
                
                # 打印分集水器回路信息
                collector_coils = design_data['CollectorCoils']
                print(f"\n🔹 分集水器回路集合 (CollectorCoils): {len(collector_coils)}个")
                for i, collector in enumerate(collector_coils, 1):
                    print(f"\n  分集水器 {i}:")
                    print(f"    - 编号(CollectorName): {collector['CollectorName']}")
                    print(f"    - 回路数量(Loops): {collector['Loops']}")
                    
                    # 打印户型盘管信息
                    coil_loops = collector['CoilLoops']
                    print(f"    - 户型盘管(CoilLoops): {len(coil_loops)}条")
                    for j, loop in enumerate(coil_loops, 1):
                        print(f"\n      回路 {j}:")
                        print(f"        - 总长度(Length): {loop['Length']:.2f}m")
                        print(f"        - 曲率(Curvity): {loop['Curvity']}")
                        
                        # 打印回路区域信息
                        areas = loop['Areas']
                        print(f"        - 回路区域(Areas): {len(areas)}个")
                        
                        # 打印回路路由信息
                        path = loop['Path']
                        print(f"        - 路由点数(Path): {len(path)}个")
                    
                    # 打印入户管道信息
                    deliverys = collector['Deliverys']
                    print(f"    - 入户管道(Deliverys): {len(deliverys)}条")
                
        print("\n✅ 管道布线完成!")

        break


def load_solver_params(json_file):
    """从JSON文件加载求解器参数"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_partition_input(json_file):
    """从JSON文件加载分区输入数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print(f"\n{'='*50}")
    print("🔷 管道布线系统")
    print('='*50)
    
    run_pipeline(num_x=3, num_y=3)

if __name__ == "__main__":
    main() 