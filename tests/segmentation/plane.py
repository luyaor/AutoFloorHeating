import random
import matplotlib.pyplot as plt
import copy
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Callable, Optional, Dict, Hashable
from dataclasses import dataclass
from loguru import logger


EPS = 1e-8


Line = np.ndarray  # 3
Vec = np.ndarray  # 2
Point = np.ndarray  # 2
Polygon = List[Point]
Seg = Tuple[Point, Point]


@np.vectorize
def eq(a: float, b: float) -> bool:
    return abs(a - b) < EPS


def same_point(p: Point, q: Point) -> bool:
    return eq(p[0], q[0]) and eq(p[1], q[1])


def directed_area(p: Point, q: Point, r: Point) -> float:
    return ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) / 2.0


def pt0_convex(pt_m1: Point, pt_0: Point, pt_1: Point, ccw: bool) -> bool:
    return (
        directed_area(pt_m1, pt_0, pt_1) > 0
        if ccw
        else directed_area(pt_m1, pt_0, pt_1) < 0
    )


def test_directed_area():
    p = arr(0, 0)
    q = arr(1, 0)
    r = arr(0, 1)
    assert eq(directed_area(p, q, r), 0.5)
    p = arr(0, 0)
    q = arr(1, 1)
    r = arr(1, 0)
    assert eq(directed_area(p, q, r), -0.5)


def arr(*args):
    return np.array(args)


def same_line(p: np.ndarray, q: np.ndarray) -> bool:
    # p = (A, B, C) means Ax + By + C = 0
    return abs(p[0] * q[1] - p[1] * q[0]) < EPS and abs(p[0] * q[2] - p[2] * q[0]) < EPS


def parallel(p: Line, q: Line) -> bool:
    return abs(p[0] * q[1] - p[1] * q[0]) < EPS


def vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:  # [0, pi]
    return np.arccos(v1 @ v2 / (norm(v1) * norm(v2)))


def vec_angle_signed(v1: np.ndarray, v2: np.ndarray) -> float:  # [-pi, pi]
    return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)


def dir_left(dir: np.ndarray):
    return np.array([-dir[1], dir[0]])


def normalized(v: np.ndarray):
    if norm(v) < EPS:
        return v
    return v / norm(v)


def norm_offset(v: np.ndarray, offset: float):
    return normalized(v) * (norm(v) + offset)


def norm_fn(v: np.ndarray, fn: Callable):
    return normalized(v) * fn(norm(v))


# A, B, C are the coefficients of the line Ax + By + C = 0
def line_cross(a: np.ndarray, b: np.ndarray) -> Optional[Point]:
    if parallel(a, b):
        return None
    x = (a[1] * b[2] - a[2] * b[1]) / (a[0] * b[1] - a[1] * b[0])
    y = (a[2] * b[0] - a[0] * b[2]) / (a[0] * b[1] - a[1] * b[0])
    return np.array([x, y])


def line_at_dir(at: np.ndarray, dir: np.ndarray):
    a = -dir[1]
    b = dir[0]
    c = -a * at[0] - b * at[1]
    return np.array([a, b, c])


def dir_of_line(line: np.ndarray):
    return np.array([line[1], -line[0]])


def line_from_1_to_2(pt1: np.ndarray, pt2: np.ndarray):
    return line_at_dir(pt1, normalized(pt2 - pt1))


def _plot_lines(
    lines: List[np.array],
    x_range: Tuple[float, float] = (0, 2),
    y_range: Tuple[float, float] = (0, 2),
):
    plt.figure(figsize=(8, 8))

    for line in lines:
        A, B, C = line
        x = np.linspace(x_range[0], x_range[1], 400)
        if B != 0:
            y = -(A * x + C) / B
        else:
            x = np.full_like(x, -C / A)
            y = np.linspace(y_range[0], y_range[1], 400)

        plt.plot(x, y, label=f"{A}x + {B}y + {C} = 0", color="red")

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Line Plot")
    # plt.legend()

    plt.show()


def _plot_points_linked(pts: List[np.ndarray]):
    pts = np.array(pts)
    x_max, x_min = np.max(pts[:, 0]), np.min(pts[:, 0])
    y_max, y_min = np.max(pts[:, 1]), np.min(pts[:, 1])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    width = max(x_max - x_min, y_max - y_min) * 1.2
    plt.figure(figsize=(8, 8))
    plt.plot(pts[:, 0], pts[:, 1], color="blue")
    plt.grid(True)
    plt.xlabel("x")
    plt.xlim(x_mid - width / 2, x_mid + width / 2)
    plt.ylim(y_mid - width / 2, y_mid + width / 2)
    plt.ylabel("y")
    plt.title("Points Plot")
    plt.show()


def inner_nxt_pt_line(outer: List[Line], width) -> Tuple[Point, Line]:
    pt_0_1 = line_cross(outer[0], outer[1])
    pt_0_m1 = line_cross(outer[0], outer[-1])
    pt_m1_m2 = line_cross(outer[-1], outer[-2])
    nxt_pt = pt_m1_m2 + norm_fn(pt_0_m1 - pt_m1_m2, lambda n: n - width)
    nxt_line = line_at_dir(nxt_pt, normalized(pt_0_1 - pt_0_m1))
    return nxt_pt, nxt_line


# [problem] 现在无脑删除 outer[-1] 和 res[-1]，这样可能会导致删除了一个正确的点。
def inner_nxt_pt_dir(
    outer: List[Tuple[Point, Line]], width, ccw: bool
) -> Tuple[Optional[Point], Optional[Vec]]:
    # print(outer[-1], outer[0])
    # logger.info("#", outer[-1], outer[0])
    cross = line_cross(line_at_dir(*outer[-1]), line_at_dir(*outer[0]))
    if cross is None:
        # print("parallel")
        return None, None
    if not eq(normalized(cross - outer[-1][0]) @ outer[-1][1], 1):
        # print("cross not on line")
        return None, None
    theta = vec_angle_signed(outer[-1][1], outer[0][1])
    go = cross - outer[-1][0]
    expected_norm = norm(go) + (-1 if ccw else 1) * width / np.sin(theta)
    if expected_norm <= 0:
        # print("expected norm <= 0")
        return None, None
    nxt_pt = outer[-1][0] + norm_fn(go, lambda _: expected_norm)
    return nxt_pt, outer[0][1]


# [deprecated]
def inner_recursive_with_depth(
    outer: List[Line], width, depth
) -> List[Tuple[Point, Line]]:
    if depth == 0:
        return []
    nxt_pt, nxt_line = inner_nxt_pt_line(outer, width)
    return [(nxt_pt, nxt_line)] + inner_recursive_with_depth(
        outer[1:] + [nxt_line], width, depth - 1
    )


def point_to_seg_distance(p: Point, s: Seg) -> float:
    """计算点到线段的距离"""
    a, b = s
    if np.all(a == b):
        return norm(p - a)
    ab = b - a
    ap = p - a
    bp = p - b
    if np.dot(ap, ab) <= 0:
        return norm(ap)
    if np.dot(bp, ab) >= 0:
        return norm(bp)
    return np.abs(np.cross(ab, ap)) / norm(ab)


def seg_dis(s1: Seg, s2: Seg) -> float:
    """计算两条线段之间的距离"""
    a, b = s1
    c, d = s2

    # 检查线段是否相交
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    if (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d)):
        return 0.0

    # 计算线段端点到另一条线段的距离，取最小值
    return min(
        point_to_seg_distance(a, s2),
        point_to_seg_distance(b, s2),
        point_to_seg_distance(c, s1),
        point_to_seg_distance(d, s1),
    )


def inner_recursive_v2(
    outer: List[Tuple[Point, Vec]],
    width,
    dont_delete_outer=False,
    start_must_be_convex=False,
) -> Optional[Tuple[List[Tuple[Point, Vec]], List[Hashable]]]:
    outer = copy.deepcopy(outer)
    res = copy.deepcopy(outer)

    # [ccw]
    ccw = not is_clockwise([pt for pt, _ in outer])
    indices = list(zip(["outer"] * len(outer), range(len(outer))))
    # ("outer", 0), ("outer", 1)...
    # ("inner", 6, 0), ("inner", 6, 1)...
    if start_must_be_convex and not pt0_convex(
        outer[-1][0], outer[0][0], outer[1][0], ccw
    ):
        logger.info("Can't satisfy start_must_be_convex")
        return None

    debug = 0
    while len(outer) >= 3 and debug < 250:
        debug += 1
        pt, dir = inner_nxt_pt_dir(outer, width, ccw)
        if pt is None or (
            len(res) > 0 and (res[-1][1]) @ (pt - res[-1][0]) < width / 2.0
        ):
            if dont_delete_outer and indices[-1][0] == "outer":
                logger.info("Can't satisfy dont_delete_outer")
                return None
            outer.pop()
            res.pop()
            indices.pop()
            continue
        outer = outer[1:] + [(pt, dir)]
        idx = (
            ("inner", indices[-1][1], 0)
            if indices[-1][0] == "outer"
            else ("inner", indices[-1][1], indices[-1][2] + 1)
        )
        indices.append(idx)
        res.append((pt, dir))
    return res, indices


def is_clockwise(points: List[Point]) -> bool:
    """
    判断一系列二维点是顺时针还是逆时针排列。
    :param points: 二维点列表，表示多边形的顶点。
    :return: 如果点是顺时针排列，返回 True；否则返回 False。
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    return area > 0


def poly_edge_pipe_width_v1(
    poly: Polygon, edge_pipe_num: List[float], sug_w: float, verbose=False
) -> List[float]:
    """
    返回输入顺序第 i 条边上所有管道的宽度
    poly 为逆时针
    """
    # edge[0] is poly[0] -> poly[1]
    # 每边限制：不超邻边，与非邻边不相交
    n = len(poly)
    assert n >= 3
    ans = [sug_w] * n

    def ith_edge(i) -> Seg:
        return poly[i % n], poly[(i + 1) % n]

    def seg_norm(seg: Seg):
        return norm(seg[1] - seg[0])

    def seg_dir(seg: Seg):
        return normalized(seg[1] - seg[0])

    print("------") if verbose else None
    for i in range(n):
        # edge i, poly[i] -> poly[i + 1]
        print(poly[i], poly[(i + 1) % n]) if verbose else None
        if edge_pipe_num[i] == 0:
            ans[i] = 0
            continue
        # i 为凸点
        # 临时方案配合 fallback. 见 [pin] 241227.1.
        print("---") if verbose else None
        print(ans[i]) if verbose else None
        if pt0_convex(poly[(i - 1 + n) % n], poly[i], poly[(i + 1) % n], ccw=True):
            ans[i] = min(
                ans[i],
                seg_norm(ith_edge(i - 1))
                / (edge_pipe_num[i] + edge_pipe_num[(i - 1 + n) % n]),
            )

        print(ans[i]) if verbose else None
        # i + 1 为凸点
        if pt0_convex(poly[i], poly[(i + 1) % n], poly[(i + 2) % n], ccw=True):
            ans[i] = min(
                ans[i],
                seg_norm(ith_edge(i + 1))
                / (edge_pipe_num[i] + edge_pipe_num[(i + 1) % n]),
            )

        print(ans[i]) if verbose else None
        # expand 后退需求
        ans[i] = min(ans[i], seg_norm(ith_edge(i)) / edge_pipe_num[i])

        print(ans[i]) if verbose else None

        for j in range(n):

            def nxt(x):
                return (x + 1) % n

            if i == j or i == nxt(j) or j == nxt(i):
                continue
            # 两边必须相对才约束
            if seg_dir(ith_edge(i)) @ seg_dir(ith_edge(j)) < 0:
                ans[i] = min(
                    ans[i],
                    seg_dis(ith_edge(i), ith_edge(j))
                    / (edge_pipe_num[i] + edge_pipe_num[j]),
                )

        print(ans[i]) if verbose else None
    return ans


@dataclass
class PipeOnAxis:
    id: int
    x: float
    rw: float  # 右边宽度
    lw: float


def pt_edge_pipes_expand_pts_v1(
    center: Point, edge_pipes: List[List[PipeOnAxis]], edge_dir: List[Vec]
) -> Dict[int, Point]:
    """
    !!! [仅适用于垂直情况]
    给出 dir 朝外，pipe 逆时针（从右往左）
    Axis 轴正方向为向左
    边也是逆时针
    返回 pipe id -> 坐标
    """
    n = len(edge_pipes)
    assert n == len(edge_dir)
    di = dict()
    for i in range(n):
        # 后退距离: 左右相邻的较大值
        # delta x is (w1 + w2) / 2
        m = len(edge_pipes[i])
        if m == 0:
            continue
        assert all(
            eq(
                edge_pipes[i][j].x - edge_pipes[i][j - 1].x,
                (edge_pipes[i][j - 1].lw + edge_pipes[i][j].rw),
            )
            for j in range(1, m)
        )
        pre = (i - 1 + n) % n
        nxt = (i + 1) % n

        # 有时同样管道，宽度会突变
        fallback = max(edge_pipes[i][-1].lw, edge_pipes[i][0].rw)

        pre_contrib = (
            +(edge_pipes[pre][-1].x + edge_pipes[pre][-1].lw)
            if 0 <= vec_angle_signed(edge_dir[pre], edge_dir[i]) <= np.pi * 0.75
            and len(edge_pipes[pre]) > 0
            else fallback
        )
        nxt_contrib = (
            -(edge_pipes[nxt][0].x - edge_pipes[nxt][0].rw)
            if 0 <= vec_angle_signed(edge_dir[i], edge_dir[nxt]) <= np.pi * 0.75
            and len(edge_pipes[nxt]) > 0
            else fallback
        )
        back = max(pre_contrib, nxt_contrib, fallback)
        expand_start_pt = center + edge_dir[i] * back
        # expand
        left = dir_left(edge_dir[i])
        for p in edge_pipes[i]:
            di[p.id] = expand_start_pt + left * p.x
        # res[i] = expanded
    return di


def test3():
    def get_p(id, x, w):
        return PipeOnAxis(id, x, w / 2.0, w / 2.0)

    res = pt_edge_pipes_expand_pts_v1(
        arr(10, 10),
        [
            [get_p(0, -3.5, 1), get_p(1, -2, 2), get_p(2, -0.5, 1)],
            [
                get_p(3, -2, 1),
                get_p(4, -1, 1),
                get_p(5, 0, 1),
                get_p(6, 1, 1),
                get_p(7, 2, 1),
            ],
            [get_p(8, 0.5, 1), get_p(9, 1.5, 1), get_p(10, 2.5, 1)],
            [
                get_p(11, -2.5, 2),
                get_p(12, -1, 1),
                get_p(13, 0, 1),
                get_p(14, 1, 1),
                get_p(15, 2.5, 2),
            ],
        ],
        [arr(1, 0), arr(0, 1), arr(-1, 0), arr(0, -1)],
    )
    sets: Dict[int, List[int]] = {0: [0, 14, 10], 2: [8, 2, 4]}
    # scatter all pt
    plt.scatter([p[0] for p in res.values()], [p[1] for p in res.values()])
    for li in sets.values():
        for i in range(len(li) - 1):
            plt.plot(
                [res[li[i]][0], res[li[(i + 1)]][0]],
                [res[li[i]][1], res[li[(i + 1)]][1]],
            )
    plt.show()


def test4():
    def get_p(id, x, w):
        return PipeOnAxis(id, x, w / 2.0, w / 2.0)

    res = pt_edge_pipes_expand_pts_v1(
        arr(72, 38),
        [
            [get_p(0, 2.5, 5)],
            [get_p(1, -1.0, 2.0)],
        ],
        [arr(-1, 0), arr(0, 1)],
    )
    sets: Dict[int, List[int]] = {0: [0, 1]}
    # scatter all pt
    plt.scatter([p[0] for p in res.values()], [p[1] for p in res.values()])
    plt.scatter([72], [38], color="red")
    for li in sets.values():
        for i in range(len(li) - 1):
            plt.plot(
                [res[li[i]][0], res[li[(i + 1)]][0]],
                [res[li[i]][1], res[li[(i + 1)]][1]],
            )
    plt.show()


# 外部需要知道哪些外围点被删除了（一定是最后连续若干条），需要投影到新的外围最后一边
def inner_recursive_v2_api(
    poly: Polygon, width: float, dont_delete_outer=False, start_must_be_convex=False
) -> Tuple[List[Point], List[Hashable]]:
    # 不删除重复点
    outer = [
        (
            poly[i],
            normalized(poly[(i + 1) % len(poly)] - poly[i]),
        )
        for i in range(len(poly))
    ]
    if (
        res := inner_recursive_v2(outer, width, dont_delete_outer, start_must_be_convex)
    ) is not None:
        res, indices = res
        return [pt for pt, _ in res], indices
    return None


def test101():
    pts = [
        np.array([102.0, 86.0]),
        np.array([106.0, 90.0]),
        np.array([152.0, 90.0]),
        np.array([156.0, 86.0]),
        np.array([156.0, 30.4]),
        np.array([152.0, 26.6]),
        np.array([99.8, 26.6]),
        np.array([96.0, 30.4]),
        np.array([96.0, 36.0]),
        np.array([97.0, 37.0]),
        np.array([102.0, 42.0]),
    ]
    res = inner_recursive_v2_api(
        pts, 8, dont_delete_outer=True, start_must_be_convex=True
    )
    if res is None:
        print("None")
        return
    res, indices = res
    print(indices)
    _plot_points_linked(res)


def test1():
    # poly: Polygon = list(reversed([arr(0, 0), arr(1, 2), arr(2, 1), arr(1, 0)]))
    poly: Polygon = [
        arr(0, 0),
        arr(0.5, 0),
        arr(0.98, 0),
        arr(1, 0.02),
        arr(1, 1),
        arr(0.43, 1),
        arr(0.4, 1.03),
        arr(0.4, 1.1),
        arr(0.4, 1.2),
        arr(0, 1.2),
    ]
    # poly.reverse()
    res = inner_recursive_v2_api(
        poly, 0.1, dont_delete_outer=True, start_must_be_convex=True
    )
    _plot_points_linked(res)


def test2():
    print(
        poly_edge_pipe_width_v1(
            [arr(0, 0), arr(3, 0), arr(1, 2), arr(3, 4), arr(0, 4)],
            [1, 3, 3, 1, 3],
            0.5,
        )
    )


if __name__ == "__main__":
    test101()
