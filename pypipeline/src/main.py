from partition import polygon_grid_partition_and_merge, plot_polygons
from cactus import (CacRegion, dijk2, g2_get_start_nodes, g3_tarjan_for_a_color, 
                   gen_one_color_m1, test_gen_all_color_m1)
import numpy as np
from data.test_data import SEG_PTS
import matplotlib.pyplot as plt

def run_pipeline(case_id: int, num_x: int = 3, num_y: int = 4):
    """
    运行管道布线的完整流程
    
    Args:
        case_id: 测试用例编号 (0-5)
        num_x: 网格x方向划分数
        num_y: 网格y方向划分数
    """
    print(f"Processing case {case_id}...")
    
    # 1. 获取输入数据
    polygon_coords = SEG_PTS[case_id]
    if case_id != 5:
        # 缩放坐标
        polygon_coords = [(x[0]/100, x[1]/100) for x in polygon_coords]
    
    # 2. 执行分区
    print("Performing partition...")
    final_polygons, nat_lines, global_points, region_info = polygon_grid_partition_and_merge(
        polygon_coords, 
        num_x=num_x, 
        num_y=num_y
    )
    
    # 3. 可视化分区结果
    print("Plotting partition result...")
    plot_polygons(final_polygons, nat_lines=nat_lines, 
                 title="Partition Result", 
                 global_points=global_points)
    
    # 4. 准备数据给cactus.py使用
    print("Preparing data for pipe routing...")
    # 构建墙体路径
    wall_pt_path = [i for i in range(len(polygon_coords))]
    
    # 构建区域信息
    cac_regions = []
    for r, color in region_info:
        cac_regions.append(CacRegion(r[::-1], color % 5))
    
    # 5. 运行管道布线
    print("Running pipe routing...")
    
    # 5.1 构建点的邻接关系 PT_EDGE_TO
    PT_EDGE_TO = [[] for _ in range(len(global_points))]
    for r in cac_regions:
        for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
            PT_EDGE_TO[x].append(y)
            PT_EDGE_TO[y].append(x)
    # 去重并按极角排序
    for id in range(len(global_points)):
        PT_EDGE_TO[id] = list(set(PT_EDGE_TO[id]))
        PT_EDGE_TO[id].sort(key=lambda x: np.arctan2(
            global_points[x][1] - global_points[id][1],
            global_points[x][0] - global_points[id][0]
        ))
    
    # 5.2 计算区域到目标点的距离
    DESTINATION_PT = 0  # 分水器所在区域编号
    G0_PIPE_WIDTH = 5.0  # 管道宽度
    
    def pt_dis(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def dijk1(set_pts, dest_pt, pt_to):
        dis = np.ones(len(set_pts), dtype=float) * float('inf')
        from queue import PriorityQueue
        q = PriorityQueue()
        q.put((0, dest_pt))
        dis[dest_pt] = 0
        while not q.empty():
            _, pt = q.get()
            for y in pt_to[pt]:
                if dis[y] > dis[pt] + pt_dis(set_pts[pt], set_pts[y]):
                    dis[y] = dis[pt] + pt_dis(set_pts[pt], set_pts[y])
                    q.put((dis[y], y))
        return dis

    PTS_DIS = dijk1(global_points, DESTINATION_PT, PT_EDGE_TO)
    cac_regions_dis = [min([PTS_DIS[x] for x in r.ccw_pts_id]) for r in cac_regions]
    
    # 5.3 执行第一阶段Dijkstra算法
    edge_pipes, pt_pipe_sets, pipe_color = dijk2(
        global_points, 
        PT_EDGE_TO=PT_EDGE_TO,
        cac_regions=cac_regions,
        destination_pt=DESTINATION_PT,
        cac_regions_dis=cac_regions_dis,
        w_sug=G0_PIPE_WIDTH
    )
    
    # 5.4 获取起始节点并生成管道路径
    print("Generating pipe routes for each color...")
    test_gen_all_color_m1()  # 这会显示最终的管道布线结果
    
    print("Pipeline completed!")
    return final_polygons, nat_lines, global_points, region_info

def main():
    # 先只测试case 4
    case_id = 4
    print(f"\n{'='*50}")
    print(f"Running case {case_id}")
    print('='*50)
    
    run_pipeline(case_id, num_x=1, num_y=2)

if __name__ == "__main__":
    main() 