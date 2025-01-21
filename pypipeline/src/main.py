import partition
import os
import visualization_data
import json
from pathlib import Path


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
        
    print("\n可用的输入文件:")
    for fname in available_files:
        print(f"@{fname}")
    
    default_file = "ARDesign-min.json"
    
    while True:
        choice = input(f"\n请选择输入文件 [@{default_file}]: ").strip()
        if not choice:
            return os.path.join("data", default_file)
            
        if choice.startswith('@'):
            filename = choice[1:]  # Remove @ prefix
            if filename in available_files:
                return os.path.join("data", filename)
        print("无效的选择，请重试")


def run_pipeline(input_file: str = None, num_x: int = 3, num_y: int = 3):
    """
    运行管道布线的完整流程
    
    Args:
        input_file: 输入JSON文件路径
        num_x: 网格x方向划分数
        num_y: 网格y方向划分数
    """
    # 0. 处理输入数据
    print("Processing input data...")
    json_path = select_input_file()
    print(f"\n✓ 成功读取文件: {json_path}")
    
    # 加载原始JSON数据显示详细信息
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n建筑信息:")
    print(f"  建筑名称: {data.get('WebParam', {}).get('Name', '未知')}")
    print(f"  建筑地址: {data.get('WebParam', {}).get('Address', '未知')}")
    
    for floor in data.get("Floor", []):
        print(f"\n楼层: {floor['Name']}")
        print(f"  层高: {floor['LevelHeight']}mm")
        
        # 打印房间信息
        rooms = floor["Construction"]["Room"]
        print(f"\n房间信息 (共{len(rooms)}个):")
        for room in rooms:
            print(f"  - {room['Name']:<10} (面积: {room['Area']}㎡, 类型: {room['NameType']})")
            
        # 打印门的信息
        doors = [d for d in floor["Construction"].get("DoorAndWindow", []) if d.get("Type") == "门"]
        print(f"\n门的信息 (共{len(doors)}个):")
        for door in doors:
            print(f"  - {door['Name']:<10} (类型: {door.get('DoorType', '普通')}, 尺寸: {door['Size']['Width']}×{door['Size']['Height']}mm)")
    
    print("\n按任意键继续处理数据...")
    input()
    
    processed_data, polygons = visualization_data.process_ar_design(json_path)
    
    # Print the merged polygons points
    
    print("\n提取的多边形信息:")
    for key, points in polygons.items():
        if key.startswith("polygon"):
            print(f"\n{key}:")
            print("Points = [")
            for x, y in points:
                print(f"    ({x:.2f}, {y:.2f}),")
            print("]")
            
            # Verify counter-clockwise order
            area = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
            print(f"Area (should be positive for CCW): {area/2:.2f}")

    print("\n✓ 原始图像绘制完成，按任意键继续...")
    input()
    # 绘制原始数据
    visualization_data.plot_comparison(processed_data, polygons, [])
    
    for key, points in polygons.items():
        if key.startswith("polygon"):
            points = [(x[0]/100, x[1]/100) for x in points]
            
            # 1. 执行分区
            print("\n开始执行空间分区...")
            input()
            print("\n空间分区计算完成")
            print("\n原始坐标点:")
            for i, (x, y) in enumerate(points):
                print(f"  点{i+1}: ({x:.2f}, {y:.2f})")
            
            final_polygons, nat_lines, allp, new_region_info, wall_path = partition.partition_work(points, num_x=num_x, num_y=num_y)
            
            print("\n分区结果:")
            print(f"  - 分区数量: {len(final_polygons)}")
            print(f"  - 分区点数: {len(allp)}")
            print(f"  - 区域信息: {len(new_region_info)}个区域")
            
            print("\n✓ 分区计算完成，按任意键查看分区图...")
            input()
            
            partition.plot_polygons(final_polygons, nat_lines=nat_lines, 
                                 title="Space Partition Result", global_points=allp)
            
            print("\n✓ 分区图显示完成，按任意键继续管道布线...")
            input()

            # 2. 执行管道布线
            print("\n开始执行管道布线...")
            print("  正在准备数据...")
            # 准备输入数据
            seg_pts = [(x[0]/100, x[1]/100) for x in allp]  # 从原始数据转换并缩放
            regions = [(r[0], r[1]) for r in new_region_info]  # 从原始数据转换
            wall_path = wall_path
            
            print("  正在加载布线模型...")
            # 使用新的管道布局求解器
            import cactus_data, case8
            # cactus_data.SEG_PTS = seg_pts
            # cactus_data.CAC_REGIONS_FAKE = regions
            # cactus_data.WALL_PT_PATH = wall_path
            cactus_data.SEG_PTS = case8.SEG_PTS
            cactus_data.CAC_REGIONS_FAKE = case8.CAC_REGIONS_FAKE
            cactus_data.WALL_PT_PATH = case8.WALL_PT_PATH
            
            print("  开始计算管道布线方案...")
            import cactus
            # cactus4.test_gen_all_color_m1()
            cactus.test_g3_all_color()
    
    print("\n✓ 管道布线完成!")


def main():
    print(f"\n{'='*50}")
    print("管道布线系统")
    print('='*50)
    
    run_pipeline(None)

if __name__ == "__main__":
    main() 