from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, nearest_points
from shapely.geometry.base import BaseGeometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Union, NamedTuple, Optional, Dict, Set
from typeguard import typechecked
from dataclasses import dataclass
from loguru import logger

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def erode_polygon_list(polygon: Polygon, width: float) -> List[Polygon]:
    """
    对多边形进行向内腐蚀处理

    参数:
    polygon: 输入的多边形
    width: 腐蚀的宽度

    返回:
    腐蚀后的多边形
    """
    # 使用buffer实现腐蚀效果
    eroded_polygon = polygon.buffer(-width, quad_segs=4)

    if eroded_polygon.is_empty or not eroded_polygon.is_valid:
        return []

    if isinstance(eroded_polygon, MultiPolygon):
        return list(eroded_polygon.geoms)
    assert isinstance(eroded_polygon, Polygon), (
        f"eroded_polygon is not a Polygon: {type(eroded_polygon)}"
    )

    return [eroded_polygon]


def eroded_and_guarenteed_width(
    polygon: Polygon, width: float, quad_segs=4
) -> List[Polygon]:
    """
    对多边形进行向内腐蚀处理后，再进行一次开运算
    """
    eroded_polygon = (
        polygon.buffer(-width, quad_segs=quad_segs)
        .buffer(-width / 2.0, quad_segs=quad_segs)
        .buffer(width / 2.0, quad_segs=quad_segs)
    )

    if eroded_polygon.is_empty or not eroded_polygon.is_valid:
        return []

    if isinstance(eroded_polygon, MultiPolygon):
        return list(eroded_polygon.geoms)
    assert isinstance(eroded_polygon, Polygon), (
        f"eroded_polygon is not a Polygon: {type(eroded_polygon)}"
    )

    return [eroded_polygon]


@typechecked
def get_eroded_polygons(
    polygon: Polygon, width: float, max_num_layers: Union[int, None] = 21
) -> List[Union[Polygon, MultiPolygon]]:
    """
    生成多层腐蚀的多边形序列

    参数:
    polygon: 原始多边形
    width: 每层腐蚀的宽度
    max_num_layers: 最大腐蚀层数，None表示不限制

    返回:
    腐蚀多边形列表，从外到内排序（第一个是原始多边形）
    """
    eroded_polygons = [polygon]  # 从原始多边形开始

    i = 0
    while max_num_layers is None or i < max_num_layers:
        # 对最后一个多边形进行腐蚀
        eroded = polygon.buffer(-width)
        if eroded.is_empty or not eroded.is_valid:
            break
        eroded_polygons.append(eroded)

        i += 1

    return eroded_polygons


def quick_plot(
    geometry: List[Union[Polygon, MultiPolygon]],
    alpha=0.3,
    title="Polygon Visualization",
    ax=None,
    show=True,
    label=None,
    with_num=True,
):
    """
    快速绘制 Shapely 的 Polygon 或 MultiPolygon 对象，并标注顶点编号

    参数:
    geometry: Polygon 或 MultiPolygon 对象
    color: 填充颜色，默认为蓝色
    alpha: 透明度，默认为 0.3
    title: 图表标题，默认为 'Polygon Visualization'
    ax: matplotlib axes对象，若为None则创建新图
    show: 是否立即显示图形，默认为True
    label: 图例标签，默认为None

    返回:
    ax: matplotlib axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    def plot_polygon_with_numbers(poly, color):
        # 绘制外边界
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha, fc=color, label=label)
        ax.plot(x, y, color=color, linewidth=2)

        # 标注外边界顶点编号
        for i, (xi, yi) in enumerate(zip(x[:-1], y[:-1])):  # [:-1]去掉重复的最后一个点
            if with_num:
                ax.text(
                    xi,
                    yi,
                    str(i),
                    fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )

        # 绘制内部孔洞（如果有的话）
        for interior in poly.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color, linewidth=2)
            ax.fill(x, y, alpha=alpha, fc="white")
            # 标注孔洞顶点编号
            for i, (xi, yi) in enumerate(zip(x[:-1], y[:-1])):
                ax.text(
                    xi,
                    yi,
                    f"h{i}",
                    fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )

    for idx, poly in enumerate(geometry):
        if isinstance(poly, Polygon):
            # 根据 idx 传入不同颜色
            plot_polygon_with_numbers(poly, f"C{idx}")
        else:  # MultiPolygon
            for i, poly in enumerate(poly.geoms):
                plot_polygon_with_numbers(poly)
                # 在每个多边形的中心添加多边形编号
                centroid = poly.centroid
            ax.text(
                centroid.x,
                centroid.y,
                f"Poly{i}",
                fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

    ax.set_aspect("equal")
    ax.grid(True)
    if title and ax.get_title() == "":  # 只在没有标题时设置标题
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()
        plt.pause(0)

    return ax


def plot_multiple_erosion_layers(polygon, widths, colors=None, alphas=None):
    """
    在同一个图上展示多层腐蚀效果

    参数:
    polygon: 原始多边形
    widths: 腐蚀宽度列表，从小到大排序
    colors: 每层使用的颜色列表，若为None则自动生成
    alphas: 每层的透明度，若为None则自动生成
    """
    if colors is None:
        # 使用颜色映射生成渐变色
        cmap = plt.cm.viridis
        colors = [cmap(i) for i in np.linspace(0, 0.8, len(widths) + 1)]

    if alphas is None:
        # 透明度从不透明到透明
        alphas = np.linspace(0.8, 0.2, len(widths) + 1)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 存储所有腐蚀后的多边形
    eroded_polygons = [polygon]  # 从原始多边形开始
    for width in widths:
        eroded = polygon.buffer(-width)
        if not eroded.is_empty and eroded.is_valid:
            eroded_polygons.append(eroded)
        else:
            break

    # 从内到外绘制（先绘制最内层）
    for i, poly in enumerate(reversed(eroded_polygons)):
        label = (
            f"腐蚀 {sum(widths[: len(eroded_polygons) - i - 1]):.1f}"
            if i < len(eroded_polygons) - 1
            else "原始"
        )
        quick_plot(
            poly, color=colors[i], alpha=alphas[i], ax=ax, show=False, label=label
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("多层腐蚀效果叠加展示")

    plt.tight_layout()
    plt.show()


def find_points_at_distance_from_line(
    line: LineString, point: Point, width: float
) -> list[Point]:
    """
    找出 LineString 上所有距离给定 Point 为 width 的点

    参数:
    line: 输入的线段
    point: 参考点
    width: 目标距离

    返回:
    list[Point]: 所有满足条件的点的列表
    """
    # 创建一个以给定点为中心、半径为width的圆
    circle = point.buffer(width)
    circle_boundary = circle.boundary

    # 计算圆边界与线段的交点
    # [NOTE] intersection 应该是 absorbing 的吧？即距离小于 EPS 即认为相交
    if line.intersects(circle_boundary):
        intersection = line.intersection(circle_boundary)

        # 交点可能是Point或MultiPoint
        if intersection.geom_type == "Point":
            return [intersection]
        elif intersection.geom_type == "MultiPoint":
            return list(intersection.geoms)

    return []


@typechecked
def find_point_at_distance_or_remove_last_line_string(
    polygon: Polygon, width: float
) -> tuple[Point, int]:
    """
    在多边形边界上查找距离0号顶点指定距离的点. Polygon 类型 -1 和 0 原来是同一个点，这里保证删除了重复的0点
    如果没有任何点满足条件，width 压缩为最后一条边 width

    参数:
    polygon: 输入的多边形
    width: 指定的距离

    返回:
    tuple: ((x, y), edge_index)
        - (x, y): 找到的点的坐标
        - edge_index: 点所在的边的索引（从0开始）. edge_index - 1 -> edge_index 这条边是非法的。此值范围 [1, len(coords)]
    """
    # 排除重复的最后一个点
    coords = list(polygon.exterior.coords)[0:-1]
    point0 = Point(coords[0])  # 0号点

    # 从0..-1边开始,依次检查每条边
    # 0 .. -1; -n + 1.. -n
    for i in range(0, -len(coords), -1):
        # 获取当前边的起点和终点
        start = Point(coords[i])
        end = Point(coords[i - 1])

        # 创建当前边的LineString
        edge = LineString([start, end])

        # 找到所有距离0号点为width的点
        points_at_width = find_points_at_distance_from_line(edge, point0, width)

        if points_at_width:
            point = min(points_at_width, key=lambda x: x.distance(start))
            return point, len(coords) + i

    return find_point_at_distance_or_remove_last_line_string(
        polygon, LineString(coords[-2:]).length
    )


@typechecked
def find_nearest_point_on_exterior(
    point: Point, ext_cut: LineString
) -> tuple[Point, int, float]:
    """
    找到点在外部多边形上的最近点及其所在边的末端点编号

    参数:
    point: 待检查的点
    ext_cut: 外部多边形

    返回:
    tuple: (nearest_point, edge_end_idx, distance)
        - nearest_point: 最近点
        - edge_end_idx: 最近点所在边的末端点编号
        - distance: 最近点到边末端点的距离
    """
    min_dist = float("inf")
    nearest_point = None
    edge_end_idx = -1
    dist_to_end = 0

    for i in range(len(ext_cut.coords) - 1):
        start = Point(ext_cut.coords[i])
        end = Point(ext_cut.coords[(i + 1)])
        edge = LineString([start, end])

        curr_nearest = nearest_points(point, edge)[1]
        dist = point.distance(curr_nearest)

        if dist < min_dist:
            min_dist = dist
            nearest_point = curr_nearest
            edge_end_idx = i + 1
            dist_to_end = curr_nearest.distance(end)

    return nearest_point, edge_end_idx, dist_to_end


class FromEnd(NamedTuple):
    """从内部多边形端点创建的新多边形"""

    polygon: Polygon


class FromMid(NamedTuple):
    """从内部多边形中间点创建的新多边形"""

    polygon: Polygon
    pt_i_nearest_edge_end_idx: int  # 最近点所在外部多边形边的终点索引
    pt_i_nearest: Point  # 最近点
    dist_to_end: float  # 最近点到边末端点的距离


PolygonInfo = Union[FromEnd, FromMid]


@typechecked
def check_intersection_excluding_endpoints(
    line: LineString, polygon_boundary: LineString
) -> bool:
    """
    检查线段与多边形边界是否相交，不考虑端点处的相交

    参数:
    line: 待检查的线段
    polygon_boundary: 多边形边界

    返回:
    bool: 如果存在非端点的相交则返回True
    """
    if not line.intersects(polygon_boundary):
        return False

    intersection = line.intersection(polygon_boundary)
    if intersection.is_empty:
        return False

    # 获取线段的端点
    line_endpoints = [Point(p) for p in line.coords]

    # 如果交点是Point
    if intersection.geom_type == "Point":
        # 检查交点是否是线段的端点
        return not any(intersection.equals(ep) for ep in line_endpoints)

    # 如果交点是MultiPoint，检查每个交点
    if intersection.geom_type == "MultiPoint":
        for point in intersection.geoms:
            # 如果有任何一个交点不是端点，则认为相交
            if not any(point.equals(ep) for ep in line_endpoints):
                return True

    return False


@typechecked
def find_nearest_point_and_create_polygon(
    ext: Polygon, interior: List[Polygon], width: float, tolerance: float = 1e-10
) -> tuple[int, LineString, List[PolygonInfo]]:
    """
    在外部多边形上找到距离0号顶点width距离的点pt1，然后根据条件创建新的多边形

    参数:
    ext: 外部多边形
    interior: 内部多边形或多个多边形
    width: 指定的距离
    tolerance: 距离比较时的容差

    返回:
    tuple: (end_idx, ext_cut, polygon_infos)
        - end_idx: 外部裁剪后的end_idx
        - ext_cut: 外部裁剪结果. 最后一个必是 cut 的点
        - polygon_infos: 新创建的多边形信息列表，每个元素可能是:
            - FromEnd: 从端点创建的新多边形
            - FromMid: 从中间点创建的新多边形，包含额外的外部多边形信息
    """
    # 找到外部多边形上距离0号顶点width距离的点
    pt1, end_idx = find_point_at_distance_or_remove_last_line_string(ext, width)
    assert end_idx > 0
    # is a LineString
    ext_cut = LineString(list(ext.exterior.coords)[:end_idx] + [pt1])

    # 存储所有创建的多边形信息
    polygon_infos: List[PolygonInfo] = []

    for curr_interior in interior:
        int_coords = list(curr_interior.exterior.coords)[:-1]

        # 找到距离pt1最近的点
        min_dist = float("inf")
        nearest_edge_start = 0
        nearest_point = None

        for i in range(len(int_coords)):
            start = Point(int_coords[i])
            end = Point(int_coords[(i + 1) % len(int_coords)])
            edge = LineString([start, end])
            curr_nearest = nearest_points(pt1, edge)[1]
            dist = pt1.distance(curr_nearest)

            if dist < min_dist:
                min_dist = dist
                nearest_edge_start = i
                nearest_point = curr_nearest

        # 检查是否需要使用FromMid逻辑
        connection_line = LineString([pt1, nearest_point])
        if min_dist > width + tolerance or check_intersection_excluding_endpoints(
            connection_line, ext_cut
        ):
            # 遍历内部多边形的端点，找到最优的外部多边形最近点
            vertex_nearest_info = []

            for i, coord in enumerate(int_coords):
                pt_i = Point(coord)
                pt_i_nearest, pt_i_nearest_edge_end_idx, dist_i = (
                    find_nearest_point_on_exterior(pt_i, ext_cut)
                )
                assert pt_i_nearest_edge_end_idx < len(ext_cut.coords)
                vertex_nearest_info.append(
                    (pt_i, pt_i_nearest, pt_i_nearest_edge_end_idx, dist_i, i)
                )

            # 按照pt_i_nearest_edge_end_idx降序和dist_i升序排序
            vertex_nearest_info.sort(key=lambda x: (-x[2], x[3]))
            best_info = vertex_nearest_info[0]
            pt_i = best_info[0]
            pt_i_nearest = best_info[1]
            pt_i_nearest_edge_end_idx = best_info[2]
            dist_to_end = best_info[3]
            vertex_idx = best_info[4]

            # 创建新的顶点序列，从pt_i开始
            new_coords = [pt_i]
            for i in range(vertex_idx + 1, len(int_coords)):
                new_coords.append(Point(int_coords[i]))
            for i in range(0, vertex_idx):
                new_coords.append(Point(int_coords[i]))
            new_coords.append(pt_i)

            new_polygon = Polygon([(p.x, p.y) for p in new_coords])
            polygon_infos.append(
                FromMid(
                    polygon=new_polygon,
                    pt_i_nearest_edge_end_idx=pt_i_nearest_edge_end_idx,
                    pt_i_nearest=pt_i_nearest,
                    dist_to_end=dist_to_end,
                )
            )
        else:
            # 使用FromEnd逻辑
            new_coords = [nearest_point]
            for i in range(nearest_edge_start + 1, len(int_coords)):
                new_coords.append(Point(int_coords[i]))
            for i in range(0, nearest_edge_start + 1):
                new_coords.append(Point(int_coords[i]))
            new_coords.append(nearest_point)

            new_polygon = Polygon([(p.x, p.y) for p in new_coords])
            polygon_infos.append(FromEnd(polygon=new_polygon))

    return end_idx, ext_cut, polygon_infos


def visualize_polygon_creation(
    ext: Polygon,
    interior: Union[Polygon, MultiPolygon],
    end_idx: int,
    ext_cut: LineString,
    polygon_infos: List[PolygonInfo],
):
    """
    可视化多边形创建的结果

    参数:
    ext: 外部多边形
    interior: 内部多边形或多边形集合
    end_idx: 外部裁剪后的end_idx
    ext_cut: 外部裁剪结果
    polygon_infos: 新创建的多边形信息列表
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 绘制外部多边形
    quick_plot(ext, color="blue", alpha=0.2, ax=ax, show=False, label="外部多边形")

    # 绘制pt1点
    ax.plot(
        ext_cut.coords[-1][0], ext_cut.coords[-1][1], "go", markersize=10, label="pt1"
    )

    # 为每个新创建的多边形使用不同的颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(polygon_infos)))

    for i, info in enumerate(polygon_infos):
        # 绘制新创建的多边形
        quick_plot(
            info.polygon,
            color=colors[i],
            alpha=0.3,
            ax=ax,
            show=False,
            label=f"新多边形 {i + 1}",
        )

        if isinstance(info, FromMid):
            # 绘制连接线
            ax.plot(
                [info.pt_i_nearest.x, info.polygon.exterior.xy[0][0]],
                [info.pt_i_nearest.y, info.polygon.exterior.xy[1][0]],
                "--",
                color=colors[i],
                linewidth=2,
            )
            # 标记最近点
            ax.plot(info.pt_i_nearest.x, info.pt_i_nearest.y, "ro", markersize=8)
        else:
            # 绘制连接线 from pt1
            ax.plot(
                [ext_cut.coords[-1][0], info.polygon.exterior.xy[0][0]],
                [ext_cut.coords[-1][1], info.polygon.exterior.xy[1][0]],
                "--",
                color=colors[i],
                linewidth=2,
            )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("多边形创建结果可视化")
    plt.tight_layout()
    plt.show()


@dataclass
class SpiralNode:
    """盘旋图的树节点"""

    point: Point
    depth: int
    children: List["SpiralNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@typechecked
def generate_spiral_tree(
    polygon: Polygon, width: float, depth: int = 0
) -> tuple[Optional[SpiralNode], int, dict[int, int]]:
    """
    生成盘旋图的树形结构

    参数:
    polygon: 输入的多边形
    width: 腐蚀宽度

    返回:
    tuple[SpiralNode, int, dict[int, int]]: (根节点, 外层多边形的终点索引, 本次生成的盘旋图顶点 id 到外围多边形顶点 idx 的映射)
    """
    # 获取腐蚀后的多边形
    eroded = eroded_and_guarenteed_width(polygon, width)

    # 获取外部多边形的所有顶点
    id_idx_map = {}

    # 获取内部多边形信息
    end_idx, ext_cut, polygon_infos = find_nearest_point_and_create_polygon(
        polygon, eroded, width
    )
    assert end_idx > 0, f"end_idx: {end_idx} 应该大于0"
    # logger.info(f"size: {len(ext_cut.coords)}")

    # 创建外部轮廓的主链
    ext_nodes = []
    for i, xy in enumerate(ext_cut.coords):
        if len(ext_nodes) == 0:
            ext_nodes.append(SpiralNode(Point(xy), depth=depth))
        else:
            ext_nodes.append(SpiralNode(Point(xy), depth=depth))
            ext_nodes[-2].children.append(ext_nodes[-1])
        if i != len(ext_cut.coords) - 1:  # 最后一个点必是 cut 新点
            id_idx_map[id(ext_nodes[-1])] = len(ext_nodes) - 1

    # 添加pt1到主链

    # 创建一个字典来存储每条边上的最近点
    edge_points: Dict[int, List[tuple[Point, SpiralNode, float]]] = {}

    # 处理每个内部多边形
    for info in polygon_infos:
        if isinstance(info, FromEnd):
            # 直接从pt1连接到内部多边形的起点
            inner_start, _, _ = generate_spiral_tree(info.polygon, width, depth + 1)
            if inner_start is None:
                continue
            ext_nodes[-1].children.append(inner_start)
            # 递归处理内部多边形
        else:  # FromMid
            # 获取内部多边形的起点
            inner_start, _, _ = generate_spiral_tree(info.polygon, width, depth + 1)
            if inner_start is None:
                continue
            # 将最近点信息添加到对应的边
            edge_end_idx = info.pt_i_nearest_edge_end_idx
            if edge_end_idx not in edge_points:
                edge_points[edge_end_idx] = []
            edge_points[edge_end_idx].append(
                (info.pt_i_nearest, inner_start, info.dist_to_end)
            )

    # 处理每条边上的最近点
    for edge_end_idx, points in edge_points.items():
        if not points:
            continue

        # 按照到终点的距离从大到小排序
        points.sort(key=lambda x: x[2], reverse=True)

        # 找到起点节点（edge_idx - 1对应的节点）

        # 创建最近点的链
        current = ext_nodes[edge_end_idx - 1]
        for pt, inner_node, _ in points:
            pt_node = SpiralNode(pt, depth=depth)
            current.children = [pt_node]  # 替换原有的连接
            pt_node.children.append(inner_node)  # 连接到内部多边形
            current = pt_node

        # 连接到 endpoint
        current.children.append(ext_nodes[edge_end_idx])

    return ext_nodes[0], end_idx, id_idx_map


@typechecked
def visualize_spiral_tree(root: SpiralNode, ax=None, alpha=0.3, show=True):
    """
    可视化盘旋树，根据深度使用不同颜色

    参数:
    root: 树的根节点
    ax: matplotlib axes对象
    alpha: 透明度
    show: 是否显示图形
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # 获取最大深度以生成颜色映射
    def get_max_depth(node: SpiralNode) -> int:
        max_depth = node.depth
        for child in node.children:
            max_depth = max(max_depth, get_max_depth(child))
        return max_depth

    max_depth = get_max_depth(root)
    colors = plt.cm.viridis(np.linspace(0, 1, max_depth + 1))

    def plot_node(node: SpiralNode, parent_point=None):
        color = colors[node.depth]
        if parent_point is not None:
            ax.plot(
                [parent_point.x, node.point.x],
                [parent_point.y, node.point.y],
                color=color,
                alpha=alpha,
            )
        ax.plot(node.point.x, node.point.y, "o", color=color)

        # 添加深度和id标注
        ax.annotate(
            f"d{node.depth}\n",
            (node.point.x, node.point.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

        for child in node.children:
            plot_node(child, node.point)

    plot_node(root)
    ax.set_aspect("equal")
    ax.grid(True)

    # 添加深度图例
    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], marker="o", label=f"深度 {i}")
        for i in range(max_depth + 1)
    ]
    ax.legend(handles=legend_elements)

    if show:
        plt.show()


@typechecked
def union_line_strings(
    line_strings: List[LineString], width: float, quad_segs=4
) -> BaseGeometry:
    """
    将多个LineString按指定宽度膨胀后求并集
    """
    buffered_polygons = [
        line_string.buffer(width, quad_segs=quad_segs) for line_string in line_strings
    ]
    return unary_union(buffered_polygons)


def asymmetric_buffer_linestring_v1(
    line, lw, rw, show=True, ax=None, color="blue", alpha=0.5
):
    """
    对LineString进行不对称膨胀，左侧宽度为lw，右侧宽度为rw

    参数:
    line: LineString - 输入的线段
    lw: float - 左侧膨胀宽度
    rw: float - 右侧膨胀宽度
    show: bool - 是否显示图形
    ax: matplotlib axes对象 - 可选的绘图区域
    color: str - 填充颜色
    alpha: float - 透明度

    返回:
    polygon: Polygon - 膨胀后的多边形
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    # 构建左右两侧的偏移点
    left_points = []
    right_points = []

    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i + 1])

        # 计算方向向量和法向量
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue

        direction = direction / length
        normal = np.array([-direction[1], direction[0]])  # 逆时针旋转90度

        # 计算左右偏移点
        left_points.append(p1 + normal * lw)
        right_points.append(p1 - normal * rw)

        # 添加最后一个点的偏移
        if i == len(coords) - 2:
            left_points.append(p2 + normal * lw)
            right_points.append(p2 - normal * rw)

    # 构建多边形顶点序列（按逆时针顺序）
    polygon_points = left_points + right_points[::-1]

    # 创建多边形
    polygon = Polygon(polygon_points)
    return polygon


@typechecked
def asymmetric_buffer_linestring_v2(
    line, lw, rw, show=False, ax=None, color="blue", alpha=0.5, line_dir_only=False
) -> Polygon:
    """
    对LineString进行不对称膨胀，左侧宽度为lw，右侧宽度为rw

    参数:
    line: LineString - 输入的线段
    lw: float - 左侧膨胀宽度
    rw: float - 右侧膨胀宽度
    show: bool - 是否显示图形
    ax: matplotlib axes对象 - 可选的绘图区域
    color: str - 填充颜色
    alpha: float - 透明度
    line_dir_only: bool - 是否只保留线段方向一侧的多边形

    返回:
    polygon: Polygon - 膨胀后的多边形
    """
    coords = list(line.coords)
    if len(coords) < 2:
        raise ValueError("LineString must have at least 2 points")

    # 计算第一个线段的方向和法向量
    p1 = np.array(coords[0])
    p2 = np.array(coords[1])
    assert not Point(p1).equals(Point(p2)), (
        f"p1 and p2 are the same point, {p1} and {p2}"
    )
    direction = p2 - p1
    normal = np.array([-direction[1], direction[0]])  # 逆时针旋转90度
    normal = normal / np.linalg.norm(normal)

    # 左移和右移线段
    left_coords = []
    right_coords = []
    for coord in coords:
        coord = np.array(coord)
        left_coords.append(tuple(coord + normal * lw / 2.0))
        right_coords.append(tuple(coord - normal * rw / 2.0))

    # 创建左右两条线段并膨胀
    left_line = LineString(left_coords)
    right_line = LineString(right_coords)

    l_poly = left_line.buffer(lw / 2.0)
    r_poly = right_line.buffer(rw / 2.0)

    # 合并两个多边形
    unary = unary_union([l_poly, r_poly])
    if not line_dir_only:
        return unary

    rect_width = np.linalg.norm(p2 - p1) + (lw + rw)
    rect_height = 4 * (lw + rw)
    # 计算矩形的四个顶点
    rect_p1 = p1 + normal * rect_height / 2
    rect_p2 = p1 - normal * rect_height / 2
    rect_p3 = rect_p2 + direction * rect_width / np.linalg.norm(direction)
    rect_p4 = rect_p1 + direction * rect_width / np.linalg.norm(direction)
    rect = Polygon([rect_p1, rect_p2, rect_p3, rect_p4])
    result = unary.intersection(rect)

    # 删除相邻共线线段
    def remove_collinear_segments(polygon: Polygon) -> Polygon:
        """
        删除多边形中相邻的共线线段，返回简化后的多边形

        参数:
        polygon: Polygon - 输入的多边形

        返回:
        Polygon - 删除共线线段后的多边形
        """
        # 获取多边形外边界坐标
        coords = list(polygon.exterior.coords)

        # 至少需要4个点(首尾重复)才可能有共线
        if len(coords) < 4:
            return polygon

        # 存储非共线点
        new_coords = [coords[0]]

        # 判断三点是否共线
        def is_collinear(p1, p2, p3):
            # 计算叉积,判断三点是否共线
            x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
            x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
            cross_product = abs(x1 * y2 - x2 * y1)
            return cross_product < 1e-10

        # 遍历所有点(除了首尾重复点)
        for i in range(1, len(coords) - 1):
            # 如果当前点与前后点不共线,则保留该点
            if not is_collinear(new_coords[-1], coords[i], coords[i + 1]):
                new_coords.append(coords[i])

        # 添加闭合点
        new_coords.append(new_coords[0])

        # 构造新多边形
        return Polygon(new_coords)

    # 删除共线线段
    remove = remove_collinear_segments(result)

    return remove


def iter_ext_segments(polygon: Polygon):
    """
    遍历Polygon的外边界线段

    参数:
    polygon: Polygon - 输入的多边形

    返回:
    generator - 生成(start_point, end_point)元组的生成器
    """
    # 遍历外边界线段
    exterior_coords = list(polygon.exterior.coords)
    for i in range(len(exterior_coords) - 1):  # -1是因为最后一个点和第一个点相同
        yield (exterior_coords[i], exterior_coords[i + 1])


if __name__ == "__main__":
    # 创建8字形多边形
    width = 0.2
    points = [
        (0, 0),
        (2, 1),
        (0, 2),
        (-2, 3),
        (-1, 4),
        (0, 3),
        (2, 4),
        (3, 2),
        (2, 0),
        (0, -1),
        (-2, 0),
        (0, 0),
    ]
    # 六边形
    # points = [(0, 0), (1, 0), (1, 1), (0.5, 1.5), (0, 1), (0, 0)]
    custom_polygon = Polygon(points)

    # 创建一个内部多边形用于测试
    inner_points = [(0.5, 1), (1, 1.5), (0.5, 2), (0, 1.5), (0.5, 1)]
    inner_polygon = Polygon(inner_points)

    eroded_polygons = get_eroded_polygons(custom_polygon, width, 1)
    interior_polygon = eroded_polygons[1]

    # 创建一个MultiPolygon用于测试
    interior_polygon2 = Polygon([(1.5, 1.5), (2.5, 2), (2, 2.5), (1.5, 2), (1.5, 1.5)])
    interior_multi = MultiPolygon([interior_polygon, interior_polygon2])

    # quick_plot([custom_polygon], color="blue", alpha=0.2, show=True)
    # 测试盘旋树生成
    root, ed_idx, id_idx_map = generate_spiral_tree(custom_polygon, width, 0)
    print(id_idx_map)
    visualize_spiral_tree(root)
