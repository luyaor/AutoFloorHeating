import json
import os
from pathlib import Path
import partition

def main():
    """
    用于调试分区功能的独立脚本
    从partition_input.json读取输入数据并执行分区
    """
    print("\n=== 分区调试工具 ===")
    
    # 查找输入文件
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("pypipeline/data")
    
    input_file = data_dir / "partition_input.json"
    if not input_file.exists():
        print(f"\n❌ 错误: 未找到输入文件 {input_file}")
        return
    
    # 读取输入数据
    print(f"\n📂 正在读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = data['points']
    num_x = data['num_x']
    num_y = data['num_y']
    floor_name = data['floor_name']
    
    print(f"\n📊 输入数据信息:")
    print(f"  - 楼层: {floor_name}")
    print(f"  - 点数: {len(points)}")
    print(f"  - 分区参数: {num_x}×{num_y}")
    
    # 执行分区
    print("\n🔷 开始执行分区...")
    final_polygons, nat_lines, allp, new_region_info, wall_path = partition.partition_work(
        points, num_x=num_x, num_y=num_y
    )
    
    print("\n📊 分区结果:")
    print(f"  - 分区数量: {len(final_polygons)}")
    print(f"  - 分区点数: {len(allp)}")
    print(f"  - 区域信息: {len(new_region_info)}个区域")
    
    # 保存分区结果
    output = {
        'final_polygons': final_polygons,
        'natural_lines': nat_lines,
        'all_points': allp,
        'region_info': new_region_info,
        'wall_path': wall_path
    }
    
    output_file = input_file.with_name('partition_output.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 分区结果已保存至: {output_file}")
    
    # 可选：绘制结果
    draw = input("\n是否绘制分区结果? [y/N]: ").lower().strip() == 'y'
    if draw:
        partition.plot_polygons(final_polygons, nat_lines=nat_lines, 
                              title=f"Space Partition Result - {floor_name}", 
                              global_points=allp)

if __name__ == "__main__":
    main() 