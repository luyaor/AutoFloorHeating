import json
from typing import List, Tuple, Dict, Any


def parse_polygon_group_intermediate(
    filename: str,
) -> Tuple[List[List[float]], List[Tuple[List[int], int]], List[int], int, float]:
    """
    解析多边形数据，提取关键字段

    参数:
    filename: str - 输入的JSON文件路径

    返回:
    Tuple[
        List[List[float]],  # seg_pts
        List[Tuple[List[int], int]],  # regions
        List[int],  # wall_path
        int,  # destination_pt
        float  # pipe_interval
    ]
    """
    # 读取JSON文件
    with open(filename, "r") as f:
        data = json.load(f)

    # 提取各个字段
    seg_pts = data.get("seg_pts", [])
    regions = [(region[0], region[1]) for region in data.get("regions", [])]
    wall_path = data.get("wall_path", [])
    destination_pt = data.get("destination_pt", 0)
    pipe_interval = data.get("pipe_interval", 0.0)

    return seg_pts, regions, wall_path, destination_pt, pipe_interval


if __name__ == "__main__":
    # 示例数据文件
    example_filename = "../../pypipeline/output/1_polygon_group_2_intermediate.json"

    # 解析数据
    seg_pts, regions, wall_path, dest_pt, interval = parse_polygon_group_intermediate(
        example_filename
    )

    # 打印结果
    print("分段点列表:")
    print(seg_pts)
    print("\n区域列表:")
    print(regions)
    print("\n墙路径:")
    print(wall_path)
    print("\n目标点:", dest_pt)
    print("管道间隔:", interval)
