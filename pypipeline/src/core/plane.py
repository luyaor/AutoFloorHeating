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


def arr(*args):
    return np.array(args)


@np.vectorize
def eq(a: float, b: float) -> bool:
    return abs(a - b) < EPS


@np.vectorize
def strictly_less(a: float, b: float) -> bool:
    return a < b - EPS


def same_point(p: Point, q: Point) -> bool:
    return eq(p[0], q[0]) and eq(p[1], q[1])


def signed_area(p: Point, q: Point, r: Point) -> float:
    return ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])) / 2.0


def test_signed_area():
    p = arr(0, 0)
    q = arr(1, 0)
    r = arr(0, 1)
    assert eq(signed_area(p, q, r), 0.5)
    p = arr(0, 0)
    q = arr(1, 1)
    r = arr(1, 0)
    assert eq(signed_area(p, q, r), -0.5)


def signed_poly_area(poly: Polygon, ccw: bool) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    return sum(
        signed_area(poly[0], poly[i], poly[(i + 1) % n]) for i in range(1, n - 1)
    ) * (1 if ccw else -1)


def test_signed_poly_area():
    assert eq(
        signed_poly_area([arr(0, 0), arr(1, 0), arr(1, 1), arr(0, 2)], ccw=False),
        -1.5,
    )


if __name__ == "__main__":
    test_signed_poly_area()
    test_signed_area()


def pt0_convex(pt_m1: Point, pt_0: Point, pt_1: Point, ccw: bool) -> bool:
    return (
        signed_area(pt_m1, pt_0, pt_1) > 0
        if ccw
        else signed_area(pt_m1, pt_0, pt_1) < 0
    )


def dir0_convex(dir_m1: Vec, dir0: Vec, ccw: bool) -> bool:
    return (
        vec_angle_signed(dir_m1, dir0) > 0
        if ccw
        else vec_angle_signed(dir_m1, dir0) < 0
    )


def same_line(p: np.ndarray, q: np.ndarray) -> bool:
    # p = (A, B, C) means Ax + By + C = 0
    return abs(p[0] * q[1] - p[1] * q[0]) < EPS and abs(p[0] * q[2] - p[2] * q[0]) < EPS


def parallel(p: Line, q: Line) -> bool:
    return abs(p[0] * q[1] - p[1] * q[0]) < EPS


def vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:  # [0, pi]
    return np.arccos(v1 @ v2 / (norm(v1) * norm(v2)))


def vec_angle_signed(v1: np.ndarray, v2: np.ndarray, ccw=True) -> float:  # [-pi, pi]
    res = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)
    return res if ccw else -res


def dir_left(dir: Vec):
    return np.array([-dir[1], dir[0]])


def dir_right(dir: Vec):
    return np.array([dir[1], -dir[0]])


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


def pt_dir_intersect(
    pt_dir_start: Tuple[Point, Vec], pt_dir_end: Tuple[Point, Vec]
) -> Optional[Point]:
    cross = line_cross(line_at_dir(*pt_dir_start), line_at_dir(*pt_dir_end))
    if cross is None:
        logger.warning("parallel")
        return None
    if not eq(normalized(cross - pt_dir_start[0]) @ pt_dir_start[1], 1):
        logger.warning("cross in the inverse direction")
        return None
    return cross


# [BUG] 现在无脑删除 outer[-1] 和 res[-1]，这样可能会导致删除了一个正确的点。
# [NOTE] 就算想切分成多个凸多边形，也是外部的事情
#   - [pin] 241229.1
def inner_nxt_pt_dir(
    outer: List[Tuple[Point, Vec]], width, ccw: bool
) -> Tuple[Optional[Point], Optional[Vec]]:
    # [v2]
    # cross = pt_dir_intersect(outer[-1], outer[0])
    # if cross is None:
    #     return None, None

    # theta = vec_angle_signed(outer[-1][1], outer[0][1])
    # go = cross - outer[-1][0]
    # expected_norm = norm(go) + (-1 if ccw else 1) * width / np.sin(theta)
    # if expected_norm <= 0:
    #     # print("expected norm <= 0")
    #     logger.warning("expected norm <= 0")
    #     return None, None
    # nxt_pt = outer[-1][0] + norm_fn(go, lambda _: expected_norm)
    # return nxt_pt, outer[0][1]

    # [v3]
    # last 向左平移 if ccw
    pt0, dir0 = outer[0]
    dir_inner = dir_left(dir0) if ccw else dir_right(dir0)
    pt_inner = pt0 + dir_inner * width
    cross = pt_dir_intersect(outer[-1], (pt_inner, dir0))
    if cross is None:
        return None, None
    return cross, outer[0][1]


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


def pt_from_pt_dir_signed_distance(pt: Point, pt_dir: Tuple[Point, Vec]) -> float:
    ...
    # 距离正方向为 dir 方向向左
    return np.dot(pt - pt_dir[0], dir_left(pt_dir[1]))


def pt_project_to_pt_dir_x(pt: Point, pt_dir: Tuple[Point, Vec]) -> float:
    return np.dot(pt - pt_dir[0], pt_dir[1])


def seg_line_cross(seg: Seg, line: Line) -> Optional[Point]:
    cross = line_cross(line, line_from_1_to_2(*seg))
    if cross is None:
        return None
    pt, dir = seg[0], normalized(seg[1] - seg[0])
    proj_x = pt_project_to_pt_dir_x(cross, (pt, dir))
    if 0 <= proj_x <= norm(seg[1] - seg[0]):
        return cross
    return None


def seg1_from_seg2_signed_distance(seg1: Seg, seg2: Seg) -> Optional[float]:
    # seg2 正方向为 seg[1] - seg[0]
    # 距离正方向为 seg[1] - seg[0] 方向向左

    pt2, dir2 = seg2[0], normalized(seg2[1] - seg2[0])
    candidate = []
    for endpoint in [seg1[0], seg1[1]]:
        if (
            0
            <= pt_project_to_pt_dir_x(endpoint, (pt2, dir2))
            <= norm(seg2[1] - seg2[0])
        ):
            candidate.append(endpoint)
    for line in [line_at_dir(seg2pt, dir_left(dir2)) for seg2pt in seg2]:
        if (cross := seg_line_cross(seg1, line)) is not None:
            candidate.append(cross)
    if len(candidate) == 0:
        return None
    return min([pt_from_pt_dir_signed_distance(pt, (pt2, dir2)) for pt in candidate])


def test_seg1_from_seg2_signed_distance():
    assert eq(
        seg1_from_seg2_signed_distance(
            (arr(9, 10), arr(10, 12)), (arr(9, 9), arr(10, 10))
        ),
        1 / 2**0.5,
    )


if __name__ == "__main__":
    test_seg1_from_seg2_signed_distance()


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

    debug = 240
    while len(outer) >= 3 and debug > 0:
        # print("------")
        debug -= 1
        # if len(res) >= 18:
        # print("ok")

        # 此为 .[-1] pt dir 和向内平移后的 .[0] line(pt dir) 交点

        pt_new, dir_new = inner_nxt_pt_dir(outer, width, ccw)
        # 新点为 pt_new，新线段为 outer[-1][0] -> pt_new
        seg_new = (outer[-1][0], pt_new)
        idx_new = (
            ("inner", indices[-1][1], 0)
            if indices[-1][0] == "outer"
            else ("inner", indices[-1][1], indices[-1][2] + 1)
        )
        # [无法求得交点（逆方向或平行）]：跳过下一边
        if pt_new is None:
            # [BUG] 这个还合法吗
            if dont_delete_outer and indices[-1][0] == "outer":
                logger.info("Can't satisfy dont_delete_outer")
                return None
            # outer.pop()
            # res.pop()
            # indices.pop()
            outer = outer[1:]
            continue

        # [线段太近, 充分不必要探测]：跳以后并忽略新点
        # 检查未来所有线段（1->2 ..= -2->-1）平移后是否相交. 这里不查直线，只查线段
        # [预测删除 v2]
        f"""
        未来某个线段 s -> s + 1 到新线段有向距离（定义见上方函数）{seg1_from_seg2_signed_distance}
        <= width 则选择跳过 s 及之前或 s + 1 及之后
        否则不用跳过，正常执行即可
        """

        def need_jump_and_jumped() -> bool:
            nonlocal outer, res, indices
            for s_idx in range(1, len(outer) - 2):
                # 传入函数的距离正方向是向左
                # 如果 clockwise，实际需要的距离正方向是向右，传入函数之前反一下
                seg_s = (
                    (outer[s_idx][0], outer[s_idx + 1][0])
                    if ccw
                    else (
                        outer[s_idx + 1][0],
                        outer[s_idx][0],
                    )
                )
                if (
                    dis := seg1_from_seg2_signed_distance(seg_new, seg_s)
                ) is not None and dis <= 1.1 * width:
                    # logger.info(f"{(s_idx, s_idx + 1)} is too close")
                    s_and_before_area_estimated = signed_poly_area(
                        [pt_new] + list(map(lambda x: x[0], outer[: s_idx + 1])), ccw
                    )
                    s1_and_after_area_estimated = signed_poly_area(
                        list(map(lambda x: x[0], outer[s_idx + 1 :])), ccw
                    )
                    # print(s_and_before_area_estimated, s1_and_after_area_estimated)
                    assert (
                        s_and_before_area_estimated >= 0
                        and s1_and_after_area_estimated >= 0
                    )
                    if s_and_before_area_estimated > s1_and_after_area_estimated:
                        # ok to append new one, delete some outer
                        # logger.warning(f"jumped {s_idx + 1} and after")
                        outer = (
                            # outer[1:s_idx]
                            # + [(outer[s_idx][0], normalized(pt_new - outer[s_idx][0]))]
                            # + [(pt_new, dir_new)]
                            outer[1 : s_idx + 1] + outer[-1:] + [(pt_new, dir_new)]
                        )
                        # [fix]
                        indices.append(idx_new)
                        res.append((pt_new, dir_new))
                    else:
                        # don't append new one, delete some outer
                        # logger.warning(
                        #     f"jumped {s_idx}'s before, which is {outer[s_idx - 1]}"
                        # )
                        outer = outer[s_idx:]
                    return True
            return False

        if need_jump_and_jumped():
            continue

        # [normal case]
        outer = outer[1:] + [(pt_new, dir_new)]
        indices.append(idx_new)
        res.append((pt_new, dir_new))
        # logger.info(f"normal case {idx_new}")
    if debug == 0:
        logger.error("Inner debug limit reached, YOU MUST CHECK HERE!")
    logger.warning(f"len(res) = {len(res)}")

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


def is_counter_clockwise(points: List[Point]) -> bool:
    return not is_clockwise(points)


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
        ), edge_pipes[i]
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
        np.array([146.0, 16.0]),
        np.array([143.5, 8.5]),
        np.array([101.5, 8.5]),
        np.array([99.0, 11.0]),
        np.array([99.0, 92.25]),
        np.array([99.0, 98.5]),
        np.array([99.0, 99.75]),
        np.array([101.5, 101.625]),
        np.array([115.5, 101.625]),
        np.array([118.0, 99.75]),
        np.array([118.0, 91.0]),
        np.array([123.0, 86.0]),
        np.array([143.5, 86.0]),
        np.array([146.0, 83.5]),
    ]
    res = inner_recursive_v2_api(
        pts, 5, dont_delete_outer=True, start_must_be_convex=True
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
    res, indices = inner_recursive_v2_api(
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
