import partition
import os
import visualization_data


def run_pipeline(case_id: int, num_x: int = 3, num_y: int = 4):
    """
    运行管道布线的完整流程
    
    Args:
        case_id: 测试用例编号 (0-5)
        num_x: 网格x方向划分数
        num_y: 网格y方向划分数
    """
    print(f"Processing case {case_id}...")
    
    # 0. 处理输入数据
    print("Processing input data...")
    json_path = os.path.join("data", "ARDesign-min.json")
    processed_data, polygons = visualization_data.process_ar_design(json_path)
    # 绘制原始数据
    visualization_data.plot_comparison(processed_data, polygons, [])
    
    for key, points in polygons.items():
        if key.startswith("polygon"):
            points = [(x[0]/100, x[1]/100) for x in points]
            # 1. 执行分区
            print("Performing partition...")
            print(points)
            final_polygons, nat_lines, allp, new_region_info,wall_path =partition.partition_work(points, num_x=num_x, num_y=num_y)
            # partition.plot_polygons(final_polygons, nat_lines=nat_lines, title="Final Merged Polygons with Global Point Indices", global_points=allp)
            print("SEG_PTS=", allp)
            print("CAC_REGIONS_FAKE=", new_region_info)
            print("WALL_PT_PATH=", wall_path)
            print("")

            # 2. 执行管道布线
            print("Running pipe routing...")
            # 准备输入数据
            seg_pts = [(x[0]/100, x[1]/100) for x in allp]  # 从原始数据转换并缩放
            regions = [(r[0], r[1]) for r in new_region_info]  # 从原始数据转换
            wall_path = wall_path
            
            # 使用新的管道布局求解器
            import cactus_data
            cactus_data.SEG_PTS = seg_pts
            cactus_data.CAC_REGIONS_FAKE = regions
            cactus_data.WALL_PT_PATH = wall_path
            import cactus4
            cactus4.test_gen_all_color_m1()
    
    print("Pipeline completed!")

def main():
    # 先只测试case 4
    case_id = 4
    print(f"\n{'='*50}")
    print(f"Running case {case_id}")
    print('='*50)
    
    run_pipeline(case_id, num_x=1, num_y=2)

if __name__ == "__main__":
    main() 