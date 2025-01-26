import partition
import os
from cactus import CacRegion, CactusSolverDebug, arr
import visualization_data
import json
from pathlib import Path
import pickle


def get_available_json_files():
    """Get list of available JSON files in the example directory"""
    example_dir = Path("data")
    return sorted([f.name for f in example_dir.glob("*.json")])


def select_input_file():
    """
    Interactive selection of input file
    Returns:
        Selected file path
    """
    available_files = get_available_json_files()
    if not available_files:
        raise FileNotFoundError("No JSON files found in example directory")
        
    print("\n🔷 可用的输入文件:")
    for fname in available_files:
        print(f"  @{fname}")
    
    default_file = "ARDesign02.json"
    
    while True:
        choice = input(f"\n🔷 请选择输入文件 [@{default_file}]: ").strip()
        if not choice:
            return os.path.join("data", default_file)
            
        if choice.startswith('@'):
            filename = choice[1:]  # Remove @ prefix
            if filename in available_files:
                return os.path.join("data", filename)
        print("❌ 无效的选择，请重试")


def run_pipeline(input_file: str = None, num_x: int = 3, num_y: int = 3):
    """
    运行管道布线的完整流程
    
    Args:
        input_file: 输入JSON文件路径
        num_x: 网格x方向划分数
        num_y: 网格y方向划分数
    """
    # 0. 处理输入数据
    print("🔷 正在处理输入数据...")
    json_path = select_input_file()
    print(f"\n✅ 成功读取文件: {json_path}")
    
    # 加载原始JSON数据显示详细信息
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n📊 建筑信息:")
    print(f"  建筑名称: {data.get('WebParam', {}).get('Name', '未知')}")
    print(f"  建筑地址: {data.get('WebParam', {}).get('Address', '未知')}")
    
    for floor in data.get("Floor", []):
        print(f"\n📊 楼层: {floor['Name']}")
        print(f"  层高: {floor['LevelHeight']}mm")
        
        # 打印房间信息
        rooms = floor["Construction"]["Room"]
        print(f"\n📊 房间信息 (共{len(rooms)}个):")
        for room in rooms:
            print(f"  - {room['Name']:<10} (面积: {room['Area']}㎡, 类型: {room['NameType']})")
            
        # 打印门的信息
        doors = [d for d in floor["Construction"].get("DoorAndWindow", []) if d.get("Type") == "门"]
        print(f"\n📊 门的信息 (共{len(doors)}个):")
        for door in doors:
            print(f"  - {door['Name']:<10} (类型: {door.get('DoorType', '普通')}, 尺寸: {door['Size']['Width']}×{door['Size']['Height']}mm)")
    
    print("\n🔷 按任意键继续处理数据...")
    input()
    
    processed_data, polygons = visualization_data.process_ar_design(json_path)
    
    print("\n📊 提取的多边形信息:")
    print("\n✅ 原始图像绘制完成，按任意键继续...")
    
    # 绘制原始数据
    # input()
    # visualization_data.plot_comparison(processed_data, polygons, [])

    for key, points in polygons.items():
        print(f"\n📊 当前处理楼层: {data['Floor'][0]['Name']}")
        if key.startswith("polygon"):
            points = [(x[0]/100, x[1]/100) for x in points]
            
            # 1. 执行分区
            print("\n🔷 开始执行空间分区...")
            
            final_polygons, nat_lines, allp, new_region_info, wall_path = partition.partition_work(points, num_x=num_x, num_y=num_y)
            
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
            
            print("🔷 正在加载布线模型...")
            import cactus
            # 使用新的管道布局求解器
            # import cactus_data, case8
            print("🔷 正在准备数据...")
            
            # 准备输入数据
            seg_pts = [(x[0]/100, x[1]/100) for x in allp]  # 从原始数据转换并缩放
            regions = [(r[0], r[1]) for r in new_region_info]  # 从原始数据转换
            
            # 打印调试信息
            print("\n🔍 Debug - First region data:", new_region_info[0] if new_region_info else None)
            
            # 保存中间数据
            intermediate_data = {
                'floor_name': data['Floor'][0]['Name'],
                'seg_pts': seg_pts,
                'regions': regions,  
                'wall_path': wall_path
            }
            
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / 'intermediate_data.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 中间数据已保存至: {output_file}")

            loaded_params = load_solver_params(output_file)
            print(loaded_params)
            seg_pts = loaded_params['seg_pts']
            regions = loaded_params['regions']
            wall_path = loaded_params['wall_path']
            
            print("🔷 开始计算管道布线方案...")
            solver = cactus.CactusSolver(glb_h=1000, 
                                         glb_w=1000, 
                                         cmap={-1: "black",8: "grey",1:"blue",2:"yellow",3:"red",4: "cyan"}, 
                                         seg_pts=[arr(x[0] / 100 - 130, x[1] / 100) for x in seg_pts], 
                                         wall_pt_path=wall_path, 
                                         cac_region_fake=[CacRegion(x[0][::1], x[1]) for x in regions], 
                                         destination_pt=0, 
                                         suggested_m0_pipe_interval=100)
            solver.process(CactusSolverDebug(m1=False))
    
    print("\n✅ 管道布线完成!")


def load_solver_params(json_file):
    """从JSON文件加载求解器参数"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    print(f"\n{'='*50}")
    print("🔷 管道布线系统")
    print('='*50)
    
    run_pipeline(None)

if __name__ == "__main__":
    main() 