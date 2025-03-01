import json
from typing import List, Tuple, Dict, Any
from pathlib import Path


def parse_polygon_data_file(
    file_path: str,
) -> Tuple[List[List[float]], List[Tuple[List[int], int]], List[int], int, float]:
    """
    从文件中读取并解析多边形数据，提取关键字段

    参数:
    file_path: str - JSON文件路径

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
    with open(file_path, "r") as f:
        data = json.load(f)

    # 提取各个字段
    seg_pts = data.get("seg_pts", [])
    regions = [(region[0], region[1]) for region in data.get("regions", [])]
    wall_path = data.get("wall_path", [])
    destination_pt = data.get("destination_pt", 0)
    pipe_interval = data.get("pipe_interval", 0.0)

    return seg_pts, regions, wall_path, destination_pt, pipe_interval


if __name__ == "__main__":
    # 创建示例数据文件
    example_data = {
        "floor_name": "1",
        "seg_pts": [[13800.0, 4999.98], [15400.0, 4999.98]],
        "regions": [[[34, 44, 43, 42, 35], 0], [[35, 38, 37, 36], 3]],
        "wall_path": [0, 1, 2, 3, 4],
        "destination_pt": 43,
        "pipe_interval": 0.25,
    }

    # 保存示例数据到文件
    example_file = "example_polygon_data.json"
    with open(example_file, "w") as f:
        json.dump(example_data, f, indent=2)

    # 解析数据文件
    seg_pts, regions, wall_path, dest_pt, interval = parse_polygon_data_file(
        example_file
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

    # 清理示例文件
    Path(example_file).unlink()
