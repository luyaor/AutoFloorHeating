# [---]

import copy
from copy import deepcopy
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
from typeguard import typechecked
import matplotlib.colors as mcolors
import numpy as np
from . import plane, shapely_spiral
from .parse_json import parse_polygon_group_intermediate
from .plane import (
    is_counter_clockwise,
    strictly_less,
    poly_edge_pipe_width_v1,
    PipeOnAxis,
    pt_edge_pipes_expand_pts_v1,
    normalized,
    same_point,
    inner_recursive_v2_api,
    dir_left,
    dir_right,
    NpVec,
    pt_dir_intersect,
)
from .shapely_spiral import (
    SHAPE_ABSORB_EPS,
    generate_spiral_tree,
    quick_plot,
    SpiralNode,
    asymmetric_buffer_linestring_v1,
    iter_ext_segments,
)
from shapely import Polygon, LineString, unary_union, Point, MultiPoint

from queue import PriorityQueue
from typing import List, Tuple, Dict, Set, Any, Union
from .utils import DisjointSet


# [util]
def arr(*l):
    return np.array(l)


INF = int(1e9)


def pt_dis(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


# [solver]
class CacRegion:
    def __init__(self, ccw, color):
        self.ccw_pts_id = ccw
        self.color = color


RegionEdgeId = Tuple[int, int]


def region_edge_id(tu: Tuple[int, int]) -> RegionEdgeId:
    return min(tu), max(tu)


class EdgePipes:
    def __init__(self, pipes=None):
        self.ccw_pipes = pipes if pipes is not None else list()

    def insert_pipe(self, x_lt_y, idx, pipe_seg_id):
        self.ccw_pipes.insert(idx if x_lt_y else len(self.ccw_pipes) - idx, pipe_seg_id)

    def get_pipes(self, x_lt_y):
        return self.ccw_pipes if x_lt_y else list(reversed(self.ccw_pipes))


# [solve.dijk2]
# (pt, y, i)，当前在 (pt, y) 管道的 pt 点上，第 i 个管道
StateT = Tuple[int, int, int]
TransferT = Dict[StateT, List[Tuple[StateT, float]]]


@dataclass
class RegionEdgeInfo:
    num_regions_that_contains_me: int


# [solve.xw]
def edge_is_wall(eid: RegionEdgeId, wall_pt_path):
    assert eid[0] < eid[1]
    for e in zip(wall_pt_path, wall_pt_path[1:] + [wall_pt_path[0]]):
        if region_edge_id(e) == eid:
            return True
    return False


# [solve.g2s1]
# pipe_id, pt_id
# G2Node = Tuple[int, int]
@typechecked
@dataclass(frozen=True)
class G2Node:
    pipe_id: int
    pt_id: int

    def __lt__(self, other):
        if not isinstance(other, G2Node):
            return NotImplemented
        return (self.pipe_id, self.pt_id) < (other.pipe_id, other.pt_id)


G2Edge = Tuple[G2Node, G2Node]


@dataclass
class G0EdgeInfo:
    # x -> y, 逆时针(右向左). x id < y id
    #      y node
    #      ^
    # l<---|----r
    #      |
    #      x node
    rw: float
    lw: float


# [solve.g2s2]
def the_other(x, li):
    assert len(li) == 2
    return li[0] if x == li[1] else li[1]


def get_g2_edge_id(e: G2Edge):
    assert e[0] != e[1]
    if e[0] < e[1]:
        return e
    return (e[1], e[0])


# [solve.g3]
# G3Node = Tuple[
#     str, Hashable
# ]  # ("outer", (pipe_id, pt_id)) | ("inner", ((pipe_id, pt_id), son_idx))
@typechecked
@dataclass(frozen=True)
class G3NewNode:
    id: int

    def __hash__(self):
        return hash(self.id)


G3Node = Union[G2Node, G3NewNode]
G3Edge = Tuple[G3Node, G3Node]


@typechecked
def g3_node_id_less(e1: G3Node, e2: G3Node):
    # outer 排前面
    if isinstance(e1, G2Node) != isinstance(e2, G2Node):
        return int(isinstance(e1, G2Node)) > int(isinstance(e2, G2Node))
    if isinstance(e1, G2Node):
        if e1.pipe_id != e2.pipe_id:
            return e1.pipe_id < e2.pipe_id
        return e1.pt_id < e2.pt_id
    return e1.id < e2.id


@typechecked
def get_g3_edge_id(e: G3Edge):
    assert e[0] != e[1]
    if g3_node_id_less(e[0], e[1]):
        return e
    return (e[1], e[0])


# [solver]
@dataclass
class CactusSolverDebug:
    show_wall: bool = False
    show_regions_with_colors: bool = False
    solve_dijk1: bool = False
    dijk2: bool = False
    xw: bool = False
    g2s1: bool = False
    g2s2: bool = False
    g2s3: bool = False
    g3: bool = True
    m1: bool = True


class CactusSolver:
    def __init__(
        self,
        cmap,
        seg_pts,
        wall_pt_path,
        cac_regions: List[CacRegion],
        destination_pt,
        suggested_m0_pipe_interval,
    ):
        # [param]
        self.cmap = cmap
        self.seg_pts = seg_pts
        self.wall_pt_path = wall_pt_path
        self.cac_regions = cac_regions
        self.destination_pt = destination_pt
        self.suggested_m0_pipe_interval = suggested_m0_pipe_interval

        # [consequent]
        self.g0_pipe_width = self.suggested_m0_pipe_interval * 2.0

        # [fix region]
        self.cac_regions = self.fixed_regions(cac_regions, seg_pts)

        # [assert]
        for cac in self.cac_regions:
            assert is_counter_clockwise([self.seg_pts[x] for x in cac.ccw_pts_id]), (
                cac.ccw_pts_id
            )
        assert all(isinstance(x, np.ndarray) for x in self.seg_pts)

        assert all(region.color in self.cmap for region in self.cac_regions), (
            f"some region color not in cmap, max color is {max(region.color for region in self.cac_regions)}",
        )

    @staticmethod
    def fixed_regions(cac_regions_fake: List[CacRegion], seg_pts: List[plane.NpPoint]):
        fixed_cac_regions = []
        for region in cac_regions_fake:
            fixed_skipped_ccw_pts_id = []
            for i, j in zip(
                region.ccw_pts_id, region.ccw_pts_id[1:] + [region.ccw_pts_id[0]]
            ):
                line_string = LineString([seg_pts[i], seg_pts[j]])
                skipped = [
                    (idx, pt)
                    for idx, pt in enumerate(seg_pts)
                    if line_string.intersects(Point(pt).buffer(SHAPE_ABSORB_EPS))
                    and idx not in region.ccw_pts_id
                ]
                fixed_skipped_ccw_pts_id.append(i)
                # 根据 i -> j 方向的投影从小到大排序
                if len(skipped) > 0:
                    logger.warning(f"edge {i, j} skipped {skipped}")
                    fixed_skipped_ccw_pts_id.extend(
                        [
                            idx
                            for idx, _ in sorted(
                                skipped,
                                key=lambda x: np.dot(
                                    x[1] - seg_pts[i],
                                    normalized(seg_pts[j] - seg_pts[i]),
                                ),
                            )
                        ]
                    )
            # [step 2 fix overlap]
            # 共线的调整顺序
            fixed_overlap_ccw_pts_id = deepcopy(fixed_skipped_ccw_pts_id)
            last_dir = arr(0, 0)
            cur_colinear_start_idx = None
            cur_colinear_first_dir = None
            cur_colinear_end_pts = []
            n = len(fixed_skipped_ccw_pts_id)
            for i in range(2 * n):
                cur = i % n
                nxt = (i + 1) % n
                dir = normalized(
                    seg_pts[fixed_skipped_ccw_pts_id[nxt]]
                    - seg_pts[fixed_skipped_ccw_pts_id[cur]]
                )
                if plane.eq(abs(last_dir @ dir), 1):  # 共线
                    if cur_colinear_start_idx is None:
                        assert len(cur_colinear_end_pts) == 0
                        cur_colinear_start_idx = cur - 1
                        cur_colinear_first_dir = last_dir
                        cur_colinear_end_pts.append(fixed_skipped_ccw_pts_id[cur])
                    cur_colinear_end_pts.append(fixed_skipped_ccw_pts_id[nxt])
                else:
                    if cur_colinear_start_idx is not None:
                        # 根据 start -> idx 方向的投影从小到大排序
                        cur_colinear_end_pts.sort(
                            key=lambda pt: np.dot(
                                seg_pts[pt]
                                - seg_pts[
                                    fixed_skipped_ccw_pts_id[cur_colinear_start_idx]
                                ],
                                cur_colinear_first_dir,
                            )
                        )
                        for i in range(len(cur_colinear_end_pts)):
                            fixed_overlap_ccw_pts_id[
                                (cur_colinear_start_idx + 1 + i) % n
                            ] = cur_colinear_end_pts[i]
                        cur_colinear_start_idx = None
                        cur_colinear_first_dir = None
                        cur_colinear_end_pts = []
                last_dir = dir

            fixed_repetition_ccw_pts_id = [
                x
                for i, x in enumerate(fixed_overlap_ccw_pts_id)
                if i == 0
                or (
                    x != fixed_overlap_ccw_pts_id[i - 1]
                    and (
                        i != len(fixed_overlap_ccw_pts_id) - 1
                        or x != fixed_overlap_ccw_pts_id[0]
                    )
                )
            ]
            fixed_cac_regions.append(
                CacRegion(fixed_repetition_ccw_pts_id, region.color)
            )
        return fixed_cac_regions

    # don't show
    def plot_wall(self, plt):
        for i in range(len(self.wall_pt_path)):
            plt.plot(
                [
                    self.seg_pts[self.wall_pt_path[i]][0],
                    self.seg_pts[self.wall_pt_path[(i + 1) % len(self.wall_pt_path)]][
                        0
                    ],
                ],
                [
                    self.seg_pts[self.wall_pt_path[i]][1],
                    self.seg_pts[self.wall_pt_path[(i + 1) % len(self.wall_pt_path)]][
                        1
                    ],
                ],
                color="black",
            )

    def plot_num(self, nums):
        for idx, (x, y) in enumerate(nums):
            plt.text(
                x, y, str(idx), color="black", fontsize=12, ha="center", va="center"
            )

    def show_wall(self):
        plt.figure(figsize=(20, 10))  # 设置图像大小为 12x7
        plt.axis("equal")
        self.plot_wall(plt)
        self.plot_num(self.seg_pts)
        plt.show()

    def show_regions_with_colors(self):
        regions = self.cac_regions
        seg = self.seg_pts
        dest_pt = self.destination_pt

        plt.figure(figsize=(20, 10))  # 设置图像大小为 20x10
        plt.axis("equal")
        self.plot_wall(plt)
        # plt.imshow(matrix, cmap=custom_cmap, norm=norm)
        for idx, region in enumerate(regions):
            x, y = zip(*[seg[i] for i in region.ccw_pts_id])
            x = list(x) + [x[0]]  # 首尾相连
            y = list(y) + [y[0]]  # 首尾相连
            plt.plot(
                x, y, color=self.cmap[region.color] if region.color > 0 else "gray"
            )  # 绘制多边形边缘，颜色为白色
            # 在多边形重心绘制编号
            center_x = sum(x) / len(x)
            center_y = sum(y) / len(y)
            plt.text(
                center_x,
                center_y,
                str(idx),
                color="red",
                fontsize=12,
                ha="center",
                va="center",
            )
        # 显示 destination
        plt.scatter([seg[dest_pt][0]], [seg[dest_pt][1]], s=[100], color="red")
        self.plot_num(self.seg_pts)
        plt.show()

    def get_region_edge_info(self) -> Dict[RegionEdgeId, RegionEdgeInfo]:
        region_edge_info: Dict[RegionEdgeId, RegionEdgeInfo] = dict()
        for r in self.cac_regions:
            for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
                region_edge_info[region_edge_id((x, y))] = RegionEdgeInfo(0)
        for r in self.cac_regions:
            for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
                region_edge_info[
                    region_edge_id((x, y))
                ].num_regions_that_contains_me += 1
        return region_edge_info

    def solve_dijk1(self, debug=False):
        # dijkstra 求各个 Region 反向 dijkstra 顺序
        pt_edge_to: List = [[] for _ in range(len(self.seg_pts))]
        for r in self.cac_regions:
            for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
                pt_edge_to[x].append(y)
                pt_edge_to[y].append(x)
        # 每个点的出点按极角排序 -pi ~ pi
        for id in range(len(self.seg_pts)):
            # 去重
            pt_edge_to[id] = list(set(pt_edge_to[id]))
            pt_edge_to[id] = sorted(
                pt_edge_to[id],
                key=lambda x: np.arctan2(
                    self.seg_pts[x][1] - self.seg_pts[id][1],
                    self.seg_pts[x][0] - self.seg_pts[id][0],
                ),
            )

        def dijk1(set_pts, dest_pt, pt_to):
            dis = np.ones(len(set_pts), dtype=float) * INF
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

        pts_dis: Any = dijk1(self.seg_pts, self.destination_pt, pt_edge_to)
        cac_regions_dis: List = [
            min([pts_dis[x] for x in r.ccw_pts_id]) for r in self.cac_regions
        ]
        # printself.pt_edge_to
        if debug:
            print("self.pt_edge_to:")
            for idx, pt in enumerate(self.seg_pts):
                print(idx, pt_edge_to[idx])

        return pt_edge_to, pt_dis, cac_regions_dis

    @staticmethod
    def get_djk_states_for_color_at_pt(
        pt_to, color, pt, edge_pipes: Dict[Tuple[int, int], EdgePipes], pipe_color
    ) -> List[StateT]:
        """
        保证逆时针给出
        """
        # from -pi to pi
        # i 的意义：
        # 若 i 为同色: i 直接对应管道在 EdgePipes 上的编号（存这个编号是为了方便插入）
        # 异色或边缘: i 对应 i 和 i - 1 之间的管道，将新成为 i
        states = []
        for y in pt_to[pt]:
            # pt -> y 方向的逆时针
            eid = region_edge_id((pt, y))
            ccw_pipes = edge_pipes[eid].ccw_pipes
            # 允许情况: i 颜色相同, 或位于 0 号位置, 或 i - 1 为异色, 若最右边则 i - 1 必须为异色
            i2 = []
            for i in list(range(len(edge_pipes[eid].ccw_pipes) + 1)):
                if i == 0:
                    i2.append(i)
                    continue
                if i == len(ccw_pipes):
                    if pipe_color[ccw_pipes[i - 1]] != color:
                        i2.append(i)
                    continue
                if (
                    pipe_color[ccw_pipes[i]] == color
                    or pipe_color[ccw_pipes[i - 1]] != color
                ):
                    i2.append(i)
            if pt > y:
                i2 = list(reversed(i2))
            for i in i2:
                states.append((pt, y, i))
        return states

    @staticmethod
    def test_get_djk_states_for_color_at_pt():
        # 随便构建一张图
        pt_to = [[1], [3, 0, 2, 4], [1], [1], [1]]
        edge_pipes = {
            (0, 1): EdgePipes([8, 7]),
            (1, 2): EdgePipes([1, 2, 3, 0]),
            (1, 3): EdgePipes([6, 4, 5]),
            (1, 4): EdgePipes(),
        }
        pipe_color = [6, 7, 6, 1, 7, 6, 6, 7, 1]
        states = CactusSolver.get_djk_states_for_color_at_pt(
            pt_to, 1, 1, edge_pipes, pipe_color
        )
        states = CactusSolver.get_djk_states_for_color_at_pt(
            pt_to, 1, 2, edge_pipes, pipe_color
        )
        assert tuple(states) == tuple([(2, 1, 4), (2, 1, 2), (2, 1, 1), (2, 1, 0)])
        # 输出 [(2, 1, 4)]? 表示一个在 2 点上，指向 1 的边上，在 (1, 2) 边的第 4 个管道L位置上

    @staticmethod
    def test_get_djk_states_for_color_at_pt2():
        # 随便构建一张图
        pt_to = [[1], [3, 0, 2, 4], [1], [1], [1]]
        edge_pipes = {
            (0, 1): EdgePipes([8, 7]),
            (1, 2): EdgePipes([1, 2, 3, 0]),
            (1, 3): EdgePipes([6, 4, 5]),
            (1, 4): EdgePipes(),
        }
        pipe_color = [6, 7, 6, 1, 7, 6, 6, 7, 1]
        state_set = set()
        for pt in range(5):
            states = CactusSolver.get_djk_states_for_color_at_pt(
                pt_to, 1, pt, edge_pipes, pipe_color
            )
            for state in states:
                state_set.add(state)
        for st in state_set:
            pt, y, i = st
            assert (y, pt, i) in state_set

    @staticmethod
    def get_djk_transfer_for_color(
        pt_to, region_color, edge_pipes, pipe_color, seg_pts, pt_pipe_sets, w_sug
    ):
        transfer: TransferT = dict()
        # List as Dict
        ccw_djk_states_of_pt = [
            CactusSolver.get_djk_states_for_color_at_pt(
                pt_to, region_color, pt, edge_pipes, pipe_color
            )
            for pt in range(len(seg_pts))
        ]
        # print(f'{edge_pipes[(2, 3)].ccw_pipes}, {edge_pipes[(4, 5)].ccw_pipes}')
        for pt in range(len(seg_pts)):
            djk_states: List[StateT] = ccw_djk_states_of_pt[pt]
            for s in djk_states:
                transfer[s] = []
                inv = (s[1], s[0], s[2])
                # check color
                ccw_pipe_li = edge_pipes[region_edge_id((s[0], s[1]))].ccw_pipes
                # 同色管道存在，则转移到对面的代价为 0.
                # 注意，同色管道存在，但可能尚未连接到 destination
                if (
                    s[2] < len(ccw_pipe_li)
                    and region_color == pipe_color[ccw_pipe_li[s[2]]]
                ):
                    transfer[s].append((inv, 0))
                else:
                    transfer[s].append((inv, pt_dis(seg_pts[s[0]], seg_pts[s[1]])))

            """
            下面判断一个点内部的状态转移，需要逆时针遍历管道并根据各并查集经过次数的元组来判断是否可连过去
            例：
                - 某一个 state: 并查集 1 经过次数 0，并查集 2 经过次数 2
                - 另一个 state: 并查集 1 经过次数 0，并查集 2 经过次数 2
                - 则他们可以转移
            """

            # 遍历任何一个状态为 [[set1_id(祖先), set1_cnt], [set2_id, set2_cnt], ...]
            # 更新 cnt 方式: 根据 edge pipe idx 找到对应的 pipe id

            pipe_sets_di = pt_pipe_sets[pt].get_sets_di()
            ancestor_di = pt_pipe_sets[pt].get_ancestor_di()
            # 记录祖先对应集合的元素的出现次数
            cnt_di = {
                x: 0 for x in pipe_sets_di.keys() if pipe_color[x] != region_color
            }

            def key(di):
                """
                传入 cnt_di，转换为键（忽略同色键值对, 即若 pipe_color[k] == color 不计入 key tuple）
                """
                nonlocal region_color, pipe_color
                # return tuple(sorted(di.items()))
                return tuple(
                    sorted([x for x in di.items() if pipe_color[x[0]] != region_color])
                )

            def go_through(x, y, st, ed):
                """
                &mut: cnt_di
                &ancestor_di
                &cnt_state_sets
                &edge_pipes
                &region_color
                &pipe_color

                方案 10L -> 方案 8L ，路过了 9, 8
                方案 0L -> 方案 2L: 路了 0, 1
                接同色 / 生长本颜色
                """
                eid = region_edge_id((x, y))
                pipes = edge_pipes[eid].ccw_pipes
                st, ed = np.clip(st, 0, len(pipes)), np.clip(ed, 0, len(pipes))

                def ccw_cross_i(i):
                    pipe_id = pipes[i]
                    if pipe_color[pipe_id] == region_color:
                        return
                    sz = len(pipe_sets_di[ancestor_di[pipe_id]])
                    cnt_di[ancestor_di[pipe_id]] += 1
                    cnt_di[ancestor_di[pipe_id]] %= sz

                # [NOTE] 这里其实不用判断方向
                if st < ed:
                    for i in range(st, ed):
                        # 越过 i 号现存管道，统计 cnt 变化
                        ccw_cross_i(i)
                else:
                    for i in range(st - 1, ed - 1, -1):
                        ccw_cross_i(i)

            if len(djk_states) == 0:
                continue

            ls = djk_states[0]
            cnt_state_sets: Dict[Tuple, Set[StateT]] = {key(cnt_di): {ls}}
            for s in djk_states[1:]:
                # 更新 cnt_di
                if s[1] == ls[1]:
                    # 内部判断走向
                    go_through(ls[0], ls[1], ls[2], s[2])
                else:
                    # 不是同一个边内的，这里得判断走向
                    if ls[0] < ls[1]:
                        go_through(ls[0], ls[1], ls[2], +INF)
                    else:
                        go_through(ls[0], ls[1], ls[2], -INF)
                    if s[0] < s[1]:
                        go_through(s[0], s[1], -INF, s[2])
                    else:
                        go_through(s[0], s[1], +INF, s[2])
                k = key(cnt_di)
                if k not in cnt_state_sets:
                    cnt_state_sets[k] = set()
                cnt_state_sets[k].add(s)
                ls = s  # 不用 copy

            # 走了一圈，各并查集路过次数必须都归 0
            assert np.all([v == 0 for v in cnt_di.values()])
            # 一个 key 的所有状态可以互相到达
            for _, v in cnt_state_sets.items():
                for s in v:
                    for ss in v:
                        if s != ss:
                            # 此处默认同点上转移代价为 w_sug，
                            # 如果设置为 0 可能多次转移
                            transfer[s].append((ss, w_sug))
        return ccw_djk_states_of_pt, transfer

    def test_transfer(self):
        pt_to = [[1], [3, 0, 2, 4], [1], [1], [1]]
        edge_pipes = {
            (0, 1): EdgePipes([8, 7]),
            (1, 2): EdgePipes([0, 1, 2, 3]),
            (1, 3): EdgePipes([6, 4, 5]),
            (1, 4): EdgePipes(),
        }
        pipe_color = [1, 7, 6, 6, 6, 7, 6, 7, 1]
        seg_pts = [(3, 0), (1, 1), (2, 4), (-1, -1), (0, 4)]
        pt_pipe_sets = {x: DisjointSet() for x in range(len(seg_pts))}
        pt_pipe_sets[0].add(7, 8)

        pt_pipe_sets[1].add(*range(9))
        pt_pipe_sets[1].mix(3, 6)
        pt_pipe_sets[1].mix(2, 4)
        pt_pipe_sets[1].mix(1, 5)
        pt_pipe_sets[1].mix(1, 7)
        pt_pipe_sets[1].mix(0, 8)

        pt_pipe_sets[2].add(0, 1, 2, 3)
        pt_pipe_sets[3].add(4, 5, 6)

        _, transfer = CactusSolver.get_djk_transfer_for_color(
            pt_to, 7, edge_pipes, pipe_color, seg_pts, pt_pipe_sets, self.g0_pipe_width
        )

    def dijk2(
        self,
        seg_pts,
        pt_to,
        cac_regions,
        destination_pt,
        cac_regions_dis,
        w_sug,
        debug,
    ):
        """
        [约定]
        - 每条边的正方向：小点 -> 大点
        - 新状态位置: 0 (0) 1 (1) 2

        - 终点为根区域边
        - 同色边代价为 0. (dijkstra 为新增边最短)
        - 如何表示 djk 状态？
            - 点，边，管道
            - 同色：直接对应 i 管道
            - 连续两个异色 / 边缘异色：挤到原 i （右边） 和 i - 1 之间
        - 初始化 edge_state. edge_state 用 CacEdge 完全表示
            - 区域边缘加管道（逆时针遍历自己，插左边）
        - 每个区域寻路：建新图
            - state
                - pt, y, pipenum
                - (edge_state, color) -> state?
                    - pipe: get_pipes
                    - x, y, i (and y, x, i)
            - transfer
        """
        djk_order = np.argsort(cac_regions_dis)
        edge_pipes: Dict[Tuple[int, int], EdgePipes] = dict()
        for r in cac_regions:
            for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
                if region_edge_id((x, y)) not in edge_pipes:
                    edge_pipes[region_edge_id((x, y))] = EdgePipes()

        # 每个 Disjoint Set 存储管道线段的 id 的集合
        pt_pipe_sets: Dict[int, DisjointSet] = {
            x: DisjointSet() for x in range(len(seg_pts))
        }
        pipe_color = (
            list()
        )  # start from 0，.[i] 存储第 i 个管道的颜色. pipe 实际上是 pipe 线段.
        region_start_pipes = [[] for _ in range(len(cac_regions))]
        # 每个有色区域逆时针一周添加管道。相邻的添加的管道在一个 pt_pipe_seg 中
        for i, r in enumerate(cac_regions):
            if r.color == 0:
                continue
            pipes_added = []
            for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
                """
                x -> y 加一条边
                """
                pipe_color.append(r.color)
                pipe_id = len(pipe_color) - 1
                pipes_added.append(pipe_id)
                # ex: 2 -> 3, [2 <- 1 <- 0] 从右往左存，插入到左边最后一个位置
                edge_pipes[region_edge_id((x, y))].insert_pipe(x > y, 0, pipe_id)
                region_start_pipes[i].append(pipe_id)

            for x, pre_pipe, nxt_pipe in zip(
                r.ccw_pts_id, pipes_added[-1:] + pipes_added[:-1], pipes_added
            ):
                pt_pipe_sets[x].add(pre_pipe)
                pt_pipe_sets[x].add(nxt_pipe)
                pt_pipe_sets[x].mix(pre_pipe, nxt_pipe)

        def state_attach_region(s, region_id):
            pipes = edge_pipes[region_edge_id((s[0], s[1]))].ccw_pipes
            return s[2] < len(pipes) and pipes[s[2]] in region_start_pipes[region_id]

        # 存储每个颜色已经连到 destination 的状态
        colors_finished_states: Dict[int, Set[StateT]] = dict()
        for region_ord in djk_order:
            # print(f"--- {region_ord} ---")
            region_color = cac_regions[region_ord].color
            if region_color == 0:
                continue
            djk_states_pt, transfer = CactusSolver.get_djk_transfer_for_color(
                pt_to,
                region_color,
                edge_pipes,
                pipe_color,
                seg_pts,
                pt_pipe_sets,
                w_sug,
            )
            dis = {}
            # Father state
            fa_s: Dict[StateT, StateT] = dict()
            for pt in range(len(seg_pts)):
                for s in djk_states_pt[pt]:
                    dis[s] = INF
                    fa_s[s] = (-1, -1, -1)

            def state_is_finished(s):
                if region_color not in colors_finished_states:
                    return s[0] == destination_pt
                return s in colors_finished_states[region_color]

            def state_is_illegal(s):
                # 本颜色已经有区域连到终点了，则不许再搜到终点
                return region_color in colors_finished_states and s[0] == destination_pt

            q = PriorityQueue()
            this_region_is_finished = False

            region_start_states = []
            for pt in range(len(seg_pts)):
                for s in djk_states_pt[pt]:
                    if state_attach_region(s, region_ord):
                        if state_is_finished(s):
                            this_region_is_finished = True
                        if not state_is_illegal(s):
                            region_start_states.append(s)
                            q.put((0, s))
                            dis[s] = 0

            if this_region_is_finished:
                for s in region_start_states:
                    colors_finished_states.setdefault(region_color, set()).add(s)
                # logger.info(f"region {region_ord} already connected to destination")
                continue
            # logger.info(f"region {region_ord} will find a path to destination")

            while not q.empty():
                _, s = q.get()
                # 终点不用继续搜
                if state_is_finished(s):
                    continue
                for t, d in transfer[s]:
                    if not state_is_illegal(t) and strictly_less(dis[s] + d, dis[t]):
                        dis[t] = dis[s] + d
                        fa_s[t] = s
                        q.put((dis[t], t))
            found_finished_states_and_dis = []
            for k, v in dis.items():
                if state_is_finished(k):
                    found_finished_states_and_dis.append((k, v))
            back_s = found_finished_states_and_dis[
                np.argmin([v for k, v in found_finished_states_and_dis])
            ][0]
            assert dis[back_s] < INF  # 有解
            s_chain: List[StateT] = [back_s]
            while not state_attach_region(back_s, region_ord):
                back_s = fa_s[back_s]
                s_chain.append(back_s)
            s_chain.reverse()

            for s in s_chain:
                colors_finished_states.setdefault(region_color, set()).add(s)
            for s in region_start_states:
                colors_finished_states.setdefault(region_color, set()).add(s)

            # [v2] 可能一条边上绕了远路加多个管道
            # - Tuple[int, int] 为要加的 idx 和 pipe_id
            # - 由于 dijkstra 的广搜性质，不会走 pt 上非必要的点
            pipe_indices_to_add_on_edge: Dict[RegionEdgeId, List[Tuple[int, int]]] = (
                dict()
            )
            pipe_id_of_chain_states: Dict[StateT, int] = dict()
            for s, ns in zip(s_chain[:-1], s_chain[1:]):
                """
                如何优雅获得每个状态对应 pipe?
                - 反转边
                    - 可能是一个新增的 pipe
                - 同点边
                """
                if (s[0], s[1]) == (ns[1], ns[0]):
                    eid = region_edge_id((s[0], s[1]))
                    pipes = edge_pipes[eid].ccw_pipes

                    idx_is_same_color = (
                        s[2] < len(pipes) and pipe_color[pipes[s[2]]] == region_color
                    )  # 意味着其实是经过存在的一条边而不是加边
                    if idx_is_same_color:
                        pipe_id_of_chain_states[s] = pipes[s[2]]
                        pipe_id_of_chain_states[ns] = pipes[s[2]]
                        continue
                    pipe_color.append(region_color)
                    pipe_id = len(pipe_color) - 1

                    pipe_id_of_chain_states[s] = pipe_id
                    pipe_id_of_chain_states[ns] = pipe_id

                    pt_pipe_sets[s[0]].add(pipe_id)
                    pt_pipe_sets[s[1]].add(pipe_id)
                    pipe_indices_to_add_on_edge.setdefault(eid, []).append(
                        (s[2], pipe_id)
                    )
            for s in s_chain:
                if s not in pipe_id_of_chain_states:
                    # assert edge_id((s[0], s[1])) in pipe_indices_to_add_on_edge
                    pipe_id_of_chain_states[s] = edge_pipes[
                        region_edge_id((s[0], s[1]))
                    ].ccw_pipes[s[2]]

            for eid, idx_pipes in pipe_indices_to_add_on_edge.items():
                cnt = 0
                for idx, pipe_id in sorted(idx_pipes, key=lambda x: x[0]):
                    edge_pipes[eid].insert_pipe(True, idx + cnt, pipe_id)
                    cnt += 1  # 后续插入的管道会往后挤

            for s, ns in zip(s_chain[:-1], s_chain[1:]):
                # mix them
                if s[0] == ns[0]:
                    pt_pipe_sets[s[0]].mix(
                        pipe_id_of_chain_states[s], pipe_id_of_chain_states[ns]
                    )

            """
            # [v1]
            pipe_path_at_pt = {x: set() for x in range(len(seg_pts))}
            # 起始边加入 path set
            pipe_path_at_pt[s_chain[0][0]].add(
                edge_pipes[edge_id((s_chain[0][0], s_chain[0][1]))].ccw_pipes[s_chain[0][2]]
            )
            for s, ns in zip(s_chain[:-1], s_chain[1:]):
                if (s[0], s[1]) == (ns[1], ns[0]):
                    # 以下部分假设走了一条边
                    eid = edge_id((s[0], s[1]))
                    pipes = edge_pipes[eid].ccw_pipes
                    in_fact_same_color = (
                        s[2] < len(pipes) and pipe_color[pipes[s[2]]] == region_color
                    )
                    # 可能以不同色多次经过一条边
                    if not in_fact_same_color:
                        pipe_color.append(region_color)
                        pipe_id = len(pipe_color) - 1
                        edge_pipes[eid].insert_pipe(True, s[2], pipe_id)
                        pt_pipe_sets[s[0]].add(pipe_id)
                        pt_pipe_sets[s[1]].add(pipe_id)

                        pipe_path_at_pt[s[0]].add(pipe_id)
                        pipe_path_at_pt[s[1]].add(pipe_id)
                        # print(
                        #     f"insert {s[0]} -> {s[1]} at {s[2]}, color {region_color} pipe_id {pipe_id}"
                        # )
                    else:
                        pipe_id = pipes[s[2]]
                        pipe_path_at_pt[s[0]].add(pipe_id)
                        pipe_path_at_pt[s[1]].add(pipe_id)

            for pt in range(len(seg_pts)):
                pipes = pipe_path_at_pt[pt]
                if len(pipes) >= 2:
                    assert len(pipes) == 2
                    pt_pipe_sets[pt].mix(*pipes)
            """

        # [test]
        def test_plot_transfer():
            def plot_transfer(transfer, seg_pts):
                for s, li in transfer.items():
                    for t in li:
                        st = s[0]
                        ed = t[0][0]
                        d = t[1]
                        if st == ed:
                            continue
                        start_pt = seg_pts[st]
                        end_pt = seg_pts[ed]
                        plt.arrow(
                            start_pt[1],
                            start_pt[0],
                            end_pt[1] - start_pt[1],
                            end_pt[0] - start_pt[0],
                            head_width=1,
                            head_length=2,
                            fc="red",
                            ec="red",
                        )
                        mid_x = (start_pt[0] + end_pt[0]) / 2
                        mid_y = (start_pt[1] + end_pt[1]) / 2
                        plt.text(
                            mid_y,
                            mid_x,
                            f"{d:.1f}",
                            color="blue",
                            fontsize=8,
                            ha="center",
                            va="center",
                        )

            plt.figure(figsize=(20, 10))
            plt.axis("equal")
            self.plt_wall(plt)
            self.plot_num(self.seg_pts)
            plot_transfer(transfer, self.seg_pts)
            plt.show()

        def test_plot_pipes():
            def plot_pipes(edge_pipes, seg_pts, pipe_color, cmap):
                for (x, y), edge in edge_pipes.items():
                    st, ed = seg_pts[x], seg_pts[y]

                    def normalized(v):
                        return v / np.linalg.norm(v)

                    dir = normalized(ed - st)
                    dir_left = np.array([-dir[1], dir[0]])
                    oy, ox = dir_left * 1.6

                    sti = -len(edge.ccw_pipes) / 2 + 0.5
                    for idx, pipe_id in enumerate(edge.ccw_pipes):
                        color = cmap[pipe_color[pipe_id]]
                        i = idx + sti
                        plt.plot(
                            [st[0] + i * oy, ed[0] + i * oy],
                            [st[1] + i * ox, ed[1] + i * ox],
                            color=color,
                            linewidth=2,
                        )

            plt.figure(figsize=(20, 10))
            plt.axis("equal")
            self.plot_num(self.seg_pts)
            self.plot_wall(plt)
            plot_pipes(edge_pipes, self.seg_pts, pipe_color, self.cmap)
            plt.show()

        if debug:
            test_plot_pipes()

        # [def] pipe_pt[pipe_pt] is a tuple of 2 pt_id
        # A pipe consists of 2 pt_id
        pipe_pt: Dict[int, Tuple[int, int]] = dict()
        for eid, edge in edge_pipes.items():
            for i, pipe_id in enumerate(edge.ccw_pipes):
                pipe_pt[pipe_id] = (eid[0], eid[1])

        return edge_pipes, pt_pipe_sets, pipe_color, pipe_pt

    @staticmethod
    def get_xw_for_each_pipe(
        regions, seg_pts, wall_pt_path, edge_pipes, sug_w, region_edge_info
    ):
        pipe_xw: Dict[int, PipeOnAxis] = dict()
        for cac in regions:
            for pti, nxt in zip(
                cac.ccw_pts_id, cac.ccw_pts_id[1:] + [cac.ccw_pts_id[0]]
            ):
                for pipe_id in edge_pipes[region_edge_id((pti, nxt))].ccw_pipes:
                    if pipe_id not in pipe_xw:
                        pipe_xw[pipe_id] = PipeOnAxis(pipe_id, None, None, None)
        # fill this

        for cac in regions:
            pts: plane.NpPolygon = [seg_pts[i] for i in cac.ccw_pts_id]
            edge_pipe_num = []
            for uid, vid in zip(
                cac.ccw_pts_id, cac.ccw_pts_id[1:] + [cac.ccw_pts_id[0]]
            ):
                eid = region_edge_id((uid, vid))
                if region_edge_info[eid].num_regions_that_contains_me == 1:
                    edge_pipe_num.append(len(edge_pipes[eid].ccw_pipes))
                else:
                    edge_pipe_num.append(len(edge_pipes[eid].ccw_pipes) / 2.0)
            edge_pipe_width = poly_edge_pipe_width_v1(
                pts, edge_pipe_num, sug_w, debug=False
            )

            # 更新所谓 xw
            def dir_range(x, y, dir):
                if x > y:
                    return range(0)
                if dir > 0:
                    return range(x, y + dir, dir)
                return range(y, x + dir, dir)

            def dir_range2(x, y, mid, dir):
                if x > y:
                    return range(0)
                if dir > 0:
                    return range(mid, y + dir, dir)
                return range(mid, x + dir, dir)

            for e_idx_in_region, (uid, vid) in enumerate(
                zip(cac.ccw_pts_id, cac.ccw_pts_id[1:] + [cac.ccw_pts_id[0]])
            ):
                eid = region_edge_id((uid, vid))
                m = len(edge_pipes[eid].ccw_pipes)
                half_w = edge_pipe_width[e_idx_in_region] / 2.0
                x, y, mid, dir = 0, m - 1, (m - 1) / 2, (1 if uid < vid else -1)

                def range_update(r: range, last: float):
                    for idx in r:
                        pipe_id = edge_pipes[eid].ccw_pipes[idx]
                        pipe_xw[pipe_id].rw = half_w
                        pipe_xw[pipe_id].lw = half_w
                        pipe_xw[pipe_id].x = last + half_w * dir
                        last = pipe_xw[pipe_id].x + half_w * dir

                if region_edge_info[eid].num_regions_that_contains_me == 1:
                    range_update(dir_range(x, y, dir), 0)
                else:
                    # 填一半。朝向当前 region 内部的方向
                    if m % 2 == 0:
                        range_update(
                            dir_range2(x, y, int(np.round(mid + 0.5 * dir)), dir), 0
                        )
                    else:
                        int_mid = int(np.round(mid))
                        if dir == 1:
                            # 朝着 l 方向走的
                            pipe_xw[edge_pipes[eid].ccw_pipes[int_mid]].lw = half_w
                        else:
                            pipe_xw[edge_pipes[eid].ccw_pipes[int_mid]].rw = half_w
                        pipe_xw[edge_pipes[eid].ccw_pipes[int_mid]].x = 0
                        range_update(
                            dir_range2(x, y, int(np.round(mid + dir)), dir),
                            half_w * dir,
                        )

        return pipe_xw

    def test_plot_pipes(self, edge_pipes, pipe_color, pipe_xw, pipes_to_plot=None):
        if pipes_to_plot is not None:
            logger.error(f"debug mode: only plot {pipes_to_plot}")

        def plot_pipes(edge_pipes, seg_pts, pipe_color, cmap, pipe_xw):
            for (x, y), edge in edge_pipes.items():
                st, ed = seg_pts[x], seg_pts[y]

                def normalized(v):
                    return v / np.linalg.norm(v)

                dir = normalized(ed - st)
                dir_left = np.array([-dir[1], dir[0]])  # x 正方向

                for idx, pipe_id in enumerate(edge.ccw_pipes):
                    if pipes_to_plot is not None and pipe_id not in pipes_to_plot:
                        continue
                    color = cmap[pipe_color[pipe_id]]
                    x = pipe_xw[pipe_id].x
                    so_st = st + dir_left * x
                    so_ed = ed + dir_left * x
                    plt.plot(
                        [so_st[0], so_ed[0]],
                        [so_st[1], so_ed[1]],
                        color=color,
                        linewidth=0.5,
                    )
                    # 在中点位置添加 pipe_id 标注
                    mid = (so_st + so_ed) / 2
                    plt.text(
                        mid[0],
                        mid[1],
                        str(pipe_id),
                        color="black",
                        fontsize=8,
                        ha="center",
                        va="center",
                    )

        plt.figure(figsize=(20, 10))
        plt.axis("equal")
        self.plot_wall(plt)
        self.plot_num(self.seg_pts)
        plot_pipes(edge_pipes, self.seg_pts, pipe_color, self.cmap, pipe_xw)
        plt.show()

    @typechecked
    @staticmethod
    def get_endpoint_for_each_pipe(
        seg_pts, pt_edge_to, edge_pipes, pipe_wx
    ) -> Tuple[
        Set[G2Node],
        Dict[G2Node, List[G2Node]],
        Dict[G2Node, plane.NpPoint],
        Dict[G2Edge, G0EdgeInfo],
    ]:
        # 初步生成
        node_set: Set[G2Node] = set()
        edge_dict: Dict[G2Node, List[G2Node]] = dict()
        node_pos: Dict[G2Node, plane.NpPoint] = dict()
        edge_info_s1: Dict[G2Edge, G0EdgeInfo] = dict()

        for uid in range(len(seg_pts)):
            center = seg_pts[uid]
            edge_dir = []
            u_edge_pipes = []
            for vid in pt_edge_to[uid]:
                eid = region_edge_id((uid, vid))
                edge_dir.append(normalized(seg_pts[vid] - center))
                # 此 list 的 pipe_id 顺序为 uid -> vid 方向
                pipe_id_li = edge_pipes[eid].get_pipes(uid < vid)
                wx_uv = [copy.deepcopy(pipe_wx[pipe_id]) for pipe_id in pipe_id_li]

                # [g2.step1]
                for pipe_id in pipe_id_li:
                    node_set.add(G2Node(pipe_id, uid))
                    edge_dict[G2Node(pipe_id, uid)] = [G2Node(pipe_id, vid)]

                if uid > vid:
                    # 反转 x，因为这种情况轴转了
                    for wx in wx_uv:
                        wx.x = -wx.x
                        wx.rw, wx.lw = wx.lw, wx.rw
                u_edge_pipes.append(wx_uv)

                # [edge info]
                for pipe_id in edge_pipes[eid].ccw_pipes:
                    g2_node_ex = G2Node(pipe_id, eid[0])
                    g2_node_ey = G2Node(pipe_id, eid[1])
                    assert g2_node_ex < g2_node_ey
                    # pipe_wx 内 rw 为 min(u, v) -> max(u, v) 主方向
                    # 此步骤中恰好 ex -> ey 为主方向
                    g2_edge_id = (g2_node_ex, g2_node_ey)
                    if g2_edge_id not in edge_info_s1:
                        edge_info_s1[g2_edge_id] = G0EdgeInfo(
                            pipe_wx[pipe_id].rw, pipe_wx[pipe_id].lw
                        )

            # 传入的所有 x 正方向为所属 edge_dir 方向的左方向
            expanded = pt_edge_pipes_expand_pts_v1(center, u_edge_pipes, edge_dir)
            for pipe_id, pos in expanded.items():
                # if pipe_id not in pipe_pt:
                #     pipe_pt[pipe_id] = []
                # pipe_pt[pipe_id].append(uid)
                node_pos[G2Node(pipe_id, uid)] = pos
        return node_set, edge_dict, node_pos, edge_info_s1

    def test_plot_pipes2(self, edge_pipes, pipe_color, pipe_pt, node_pos_s1):
        def plot_pipes(edge_pipes, seg_pts, pipe_color, cmap, pipe_pt, node_pos):
            for (x, y), edge in edge_pipes.items():
                st, ed = seg_pts[x], seg_pts[y]

                def normalized(v):
                    return v / np.linalg.norm(v)

                for idx, pipe_id in enumerate(edge.ccw_pipes):
                    color = cmap[pipe_color[pipe_id]]
                    assert len(pipe_pt[pipe_id]) == 2
                    pt0, pt1 = pipe_pt[pipe_id]
                    st = node_pos[G2Node(pipe_id, pt0)]
                    ed = node_pos[G2Node(pipe_id, pt1)]
                    plt.plot([st[0], ed[0]], [st[1], ed[1]], color=color, linewidth=1.1)
                    # 在中点绘制文字 pipe_id
                    mid = (st + ed) / 2
                    plt.text(
                        mid[0],
                        mid[1],
                        f"{pipe_id}",
                        color="blue",
                        fontsize=8,
                        ha="center",
                        va="center",
                    )

        # [pt sets]
        # print('--- pt sets ---')
        # for k, v in pt_pipe_sets.items():
        #     print(f'{k}:')
        #     print(v.get_sets_di())

        plt.figure(figsize=(20, 10))
        plt.axis("equal")
        self.plot_num(self.seg_pts)
        self.plot_wall(plt)
        plot_pipes(
            edge_pipes,
            self.seg_pts,
            pipe_color,
            self.cmap,
            pipe_pt,
            node_pos_s1,
        )
        plt.show()

    @typechecked
    @staticmethod
    def build_linked_g2(
        node_set: Set[G2Node],
        edge_dict: Dict[G2Node, List[G2Node]],
        pt_pipe_sets: Dict[int, DisjointSet],
        edge_info_s1: Dict[G2Edge, G0EdgeInfo],
        pipe_pt: Dict[int, Tuple[int, int]],
    ):
        node_set = copy.deepcopy(node_set)
        edge_dict = copy.deepcopy(edge_dict)
        edge_info_s2 = copy.deepcopy(edge_info_s1)

        for pt_id in pt_pipe_sets.keys():
            for _, disjoint in pt_pipe_sets[pt_id].get_sets_di().items():
                for i in range(1, len(disjoint)):
                    assert G2Node(disjoint[i], pt_id) in node_set
                    u = G2Node(disjoint[i], pt_id)
                    v = G2Node(disjoint[i - 1], pt_id)
                    edge_dict[u].append(v)
                    edge_dict[v].append(u)

                    uv_edge_id = get_g2_edge_id((u, v))
                    uo = G2Node(disjoint[i], the_other(pt_id, pipe_pt[disjoint[i]]))
                    vo = G2Node(
                        disjoint[i - 1], the_other(pt_id, pipe_pt[disjoint[i - 1]])
                    )
                    u_pipe_edge_id = get_g2_edge_id((uo, u))
                    v_pipe_edge_id = get_g2_edge_id((vo, v))

                    lw = min(
                        edge_info_s1[u_pipe_edge_id].lw,
                        edge_info_s1[v_pipe_edge_id].lw,
                        edge_info_s1[v_pipe_edge_id].rw,
                        edge_info_s1[u_pipe_edge_id].rw,
                    )
                    rw = lw
                    edge_info_s2[uv_edge_id] = G0EdgeInfo(rw, lw)

        return node_set, edge_dict, edge_info_s2

    def test_g2_s2(self, edge_dict_s2, node_pos_s1, pipe_color):
        plt.figure(figsize=(20, 10))
        plt.axis("equal")
        self.plot_wall(plt)
        # plot_num(SEG_PTS)
        # plot all edge in EDGE_DICT_S2

        for k, v in edge_dict_s2.items():
            for vv in v:
                st = node_pos_s1[k]
                ed = node_pos_s1[vv]
                plt.plot(
                    [st[0], ed[0]],
                    [st[1], ed[1]],
                    color=self.cmap[pipe_color[k.pipe_id]],
                    linewidth=1.1,
                )
        plt.show()

    @staticmethod
    def g2_unique_xy(
        node_set_s2: Set[G2Node],
        edge_dict_s2: Dict[G2Node, Set[G2Node]],
        node_pos_s2: Dict[G2Node, plane.NpPoint],
        edge_info_s2: Dict[G2Edge, G0EdgeInfo],
        dest_pt: int,
    ):
        node_set_s3: Set[G2Node] = set()
        edge_dict_s3: Dict[G2Node, Set[G2Node]] = dict()
        node_pos_s3: Dict[G2Node, plane.NpPoint] = dict()
        map_s2_s3: Dict[G2Node, G2Node] = dict()
        edge_info_s3: Dict[G2Edge, G0EdgeInfo] = dict()

        not_taken = copy.deepcopy(node_set_s2)
        for u_s2 in node_set_s2:
            if u_s2 not in not_taken:
                continue
            same_li = []
            for v_s2 in not_taken:
                if same_point(node_pos_s2[u_s2], node_pos_s2[v_s2]):
                    same_li.append(v_s2)  # can be u_s2
            with_dest = [x for x in same_li if x.pt_id == dest_pt]
            assert len(with_dest) <= 1
            s3 = with_dest[0] if len(with_dest) > 0 else same_li[0]
            for same in same_li:
                not_taken.remove(same)
                map_s2_s3[same] = s3
            node_set_s3.add(s3)
            node_pos_s3[s3] = node_pos_s2[s3]

        # copy edges
        for u_s2 in node_set_s2:
            u_s3 = map_s2_s3[u_s2]
            if u_s3 not in edge_dict_s3:
                edge_dict_s3[u_s3] = set()
            for v_s2 in edge_dict_s2[u_s2]:
                v_s3 = map_s2_s3[v_s2]
                if u_s3 != v_s3 and v_s3 not in edge_dict_s3[u_s3]:
                    edge_dict_s3[u_s3].add(v_s3)

        for (u_s2, v_s2), info in edge_info_s2.items():
            assert u_s2 < v_s2
            u_s3 = map_s2_s3[u_s2]
            v_s3 = map_s2_s3[v_s2]
            # [NOTE] 这是 first cover，可能有问题
            if (u_s3, v_s3) not in edge_info_s3:
                edge_info_s3[(u_s3, v_s3)] = info

        return node_set_s3, edge_dict_s3, node_pos_s3, edge_info_s3

    def test_g2_s3(self, g2_edge_dict_s3, g2_node_pos_s3, pipe_color):
        plt.figure(figsize=(20, 10))
        plt.axis("equal")
        self.plot_num(self.seg_pts)
        for k, v in g2_edge_dict_s3.items():
            for vv in v:
                st = g2_node_pos_s3[k]
                ed = g2_node_pos_s3[vv]
                plt.plot(
                    [st[0], ed[0]],
                    [st[1], ed[1]],
                    color=self.cmap[pipe_color[k.pipe_id]],
                    linewidth=1.1,
                )
        plt.show()

    @staticmethod
    @typechecked
    def g3_tarjan_for_a_color(
        start_node: G2Node,
        node_set: Set[G2Node],
        edge_dict: Dict[G2Node, Set[G2Node]],
        node_pos: Dict[G2Node, plane.NpPoint],
        edge_info: Dict[G2Edge, G0EdgeInfo],
        w_sug: float,
    ) -> Tuple[
        Set[G3Node],
        Dict[G3Node, List[G3Node]],
        Dict[G3Node, plane.NpPoint],
        Dict[G3Edge, G0EdgeInfo],
    ]:
        """
        参数均为 G2
        G3 为有向树
        """
        g3_node: Set[G3Node] = set()
        # 有向边
        g3_edge: Dict[G3Node, List[G3Node]] = dict()  # copy.deepcopy(edge_dict)
        g3_edge_info: Dict[G3Edge, G0EdgeInfo] = dict()  # copy.deepcopy(edge_dict)
        g3_node_pos: Dict[G3Node, plane.NpPoint] = dict()
        stack = []

        g3_new_id_cnt = 0

        dfn = dict()
        low = dict()
        cnt = 0

        def tarjan(u, fa):
            nonlocal cnt  # nonlocal stack value
            cnt += 1
            dfn[u] = low[u] = cnt
            stack.append(u)

            # [g3]
            g3_node.add(u)
            g3_node_pos[u] = node_pos[u]

            def solve_id(id1: G3Node, id2: G3Node):
                # 存储时 rw 方向如何确定？永远都是小 id 指向大 id
                nonlocal w_sug
                if isinstance(id1, G2Node) and isinstance(id2, G2Node):
                    # 顺序一致
                    g3_edge_info[get_g3_edge_id((id1, id2))] = edge_info[
                        get_g2_edge_id((id1, id2))
                    ]
                    return
                g3_edge_info[get_g3_edge_id((id1, id2))] = G0EdgeInfo(
                    w_sug / 2.0, w_sug / 2.0
                )

            def solve_g2_cycle_by_shapely_spiral(cycle: List[G2Node]):
                # rt 不可删除
                # 已有子节点不可删除
                # [readonly]
                nonlocal w_sug
                # cannot_delete_idx = [0]
                # for i in range(1, len(cycle)):
                #     if ("outer", cycle[i]) in g3_edge:
                #         cannot_delete_idx.append(i)
                make_a_polygon = Polygon([node_pos[g2] for g2 in cycle])
                root, ed_idx, id_idx_map = generate_spiral_tree(
                    make_a_polygon, w_sug, 0
                )

                # 该 map 仅在当前环内有效
                node_id_g3_map: Dict[int, G3Node] = {
                    id: cycle[idx] for id, idx in id_idx_map.items()
                }

                def get_g3_id(spiral_node: SpiralNode) -> G3Node:
                    id_node = id(spiral_node)
                    nonlocal node_id_g3_map
                    if id_node not in node_id_g3_map:
                        nonlocal g3_new_id_cnt
                        g3_new_id_cnt += 1
                        new_node = G3NewNode(g3_new_id_cnt)
                        node_id_g3_map[id_node] = new_node
                        g3_node.add(new_node)
                        g3_node_pos[new_node] = arr(
                            spiral_node.point.xy[0][0],
                            spiral_node.point.xy[1][0],
                        )
                    return node_id_g3_map[id_node]

                # 遍历这棵树
                def dfs(u: SpiralNode):
                    nonlocal id_idx_map
                    nonlocal node_id_g3_map, g3_edge, g3_edge_info
                    uid = get_g3_id(u)
                    if u.children:
                        for v in u.children:
                            vid = get_g3_id(v)
                            width = max(u.actual_width, v.actual_width)
                            g3_edge[uid] = g3_edge.get(uid, []) + [vid]
                            g3_edge_info[get_g3_edge_id((uid, vid))] = G0EdgeInfo(
                                width / 2.0, width / 2.0
                            )
                            dfs(v)

                dfs(root)

                real_end_point = cycle[ed_idx - 1]
                for idx in range(ed_idx, len(cycle)):
                    nonlocal g3_edge, g3_edge_info
                    deleted_g2 = cycle[idx]
                    # g3 edge. All link to ed_idx - 1
                    for v in g3_edge.get(deleted_g2, []):
                        g3_edge[real_end_point] = g3_edge.get(real_end_point, []) + [v]
                        g3_edge_info[get_g3_edge_id((real_end_point, v))] = (
                            g3_edge_info[get_g3_edge_id((deleted_g2, v))]
                        )

            def solve_g2_cycle(cycle: List[G2Node]):
                rt = cycle[0]

                # 由于保持内盘旋 w_sug 需要，尾部若干点可能被删除
                # [方法1]
                # 原点保持存在，并连接到当前外围最后一点
                # - 为何不投影
                #   - 会交叉
                # - 为何不放宽缺口处宽度要求
                #   - 首尾共线时宽度可能为 0
                # [方法2] [当前方案]
                # - 强制找到某一个起点使得不用删除 outer。可能失败
                def fn():
                    no_del_cycle = copy.deepcopy(cycle)
                    for _ in range(len(cycle)):
                        pts_xy = [node_pos[x] for x in no_del_cycle]

                        # print(f"pts = {pts_xy}")
                        # print(f"w_sug = {w_sug}")

                        ret = inner_recursive_v2_api(
                            pts_xy,
                            w_sug,
                            dont_delete_outer=True,
                            start_must_be_convex=True,
                        )
                        if ret is not None:
                            inner_pts, indices = ret
                            return no_del_cycle, inner_pts, indices
                        no_del_cycle = no_del_cycle[1:] + [no_del_cycle[0]]
                    raise ValueError(
                        "Can't find any circle without outer deletion. Try to reduce the `w_sug`."
                    )

                no_del_cycle, inner_pts, indices = fn()
                print(f"found cycle: {cycle}")
                print(f"pts (test it): {[node_pos[x] for x in no_del_cycle]}")

                rt_idx = no_del_cycle.index(rt)
                # [反向加 outer 边]
                for i in range(rt_idx, 0, -1):
                    g3_id = ("outer", no_del_cycle[i])
                    g3_nxt_id = ("outer", no_del_cycle[i - 1])
                    g3_edge[g3_id] = g3_edge.get(g3_id, []) + [g3_nxt_id]
                    solve_id(g3_id, g3_nxt_id)
                # [正向加 outer 边]
                for i in range(rt_idx, len(no_del_cycle) - 1):
                    g3_id = ("outer", no_del_cycle[i])
                    g3_nxt_id = ("outer", no_del_cycle[i + 1])
                    g3_edge[g3_id] = g3_edge.get(g3_id, []) + [g3_nxt_id]
                    solve_id(g3_id, g3_nxt_id)
                # [制作 inner 点和边]
                last_id = ("outer", no_del_cycle[-1])
                for idx, pt in enumerate(inner_pts[len(no_del_cycle) :]):
                    g3_id = ("inner", (no_del_cycle[-1], idx))
                    g3_node.add(g3_id)
                    g3_edge[last_id] = g3_edge.get(last_id, []) + [g3_id]
                    g3_node_pos[g3_id] = pt
                    solve_id(last_id, g3_id)
                    last_id = g3_id

            for v in edge_dict[u]:
                if v == fa:
                    continue
                if v not in dfn:
                    tarjan(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > dfn[u]:
                        # 非环边
                        assert stack[-1] == v
                        g3_uid = u
                        g3_vid = v
                        g3_edge[g3_uid] = g3_edge.get(g3_uid, []) + [g3_vid]
                        solve_id(g3_uid, g3_vid)
                        stack.pop()
                    elif low[v] == dfn[u]:
                        # 环首边，退栈求环
                        cycle = []
                        while stack[-1] != u:
                            cycle.append(stack.pop())
                        cycle.append(u)  # 不 pop
                        cycle.reverse()
                        # solve_g2_cycle(cycle)
                        solve_g2_cycle_by_shapely_spiral(cycle)
                else:
                    low[u] = min(low[u], dfn[v])

            """
            无向图二元环就是重边
            """

        tarjan(start_node, None)
        return g3_node, g3_edge, g3_node_pos, g3_edge_info

    @staticmethod
    def g2_get_start_nodes(
        g2_node_set: Set[G2Node], pipe_color: Dict[int, int], dest_pt: int
    ) -> List[G2Node]:
        # 同色保留第一个
        color_registered = set()
        res = []
        for node in g2_node_set:
            if node.pt_id == dest_pt:
                if pipe_color[node.pipe_id] not in color_registered:
                    color_registered.add(pipe_color[node.pipe_id])
                    res.append(node)
                else:
                    logger.error(f"{node}")
                    raise ValueError(
                        f"There are more than one start node in the same color {pipe_color[node.pipe_id]}. note: `dest_pt` must be in a white region."
                    )
        return res

    def test_g3(
        self,
        start_nodes,
        pipe_color,
        g2_node_set_s3,
        g2_edge_dict_s3,
        g2_node_pos_s3,
        g2_edge_info_s3,
        g0_pipe_width,
    ):
        plt.figure(figsize=(20, 10))
        plt.axis("equal")
        self.plot_wall(plt)

        for s in start_nodes:
            n, e, p, i = CactusSolver.g3_tarjan_for_a_color(
                s,
                g2_node_set_s3,
                g2_edge_dict_s3,
                g2_node_pos_s3,
                g2_edge_info_s3,
                g0_pipe_width,
            )

            # plot_num(SEG_PTS)
            def dfs_plot(u):
                for v in e.get(u, []):
                    plt.plot(
                        [p[u][0], p[v][0]],
                        [p[u][1], p[v][1]],
                        color=self.cmap[pipe_color[s.pipe_id]],
                        linewidth=1,
                    )
                    dfs_plot(v)

            dfs_plot(s)
        plt.show()

    @typechecked
    @staticmethod
    def gen_one_color_m1_by_shapely(
        start_g3_node: G3Node,
        dest_pt: int,
        g3_edge_dict: Dict[G3Node, List[G3Node]],
        g3_node_pos: Dict[G3Node, plane.NpPoint],
        g3_edge_info: Dict[G3Edge, G0EdgeInfo],
    ) -> List[plane.NpPoint]:
        # get all line strings by dfs
        polygons: List[Polygon] = []

        def dfs(
            u: G3Node,
            dir_to_u: NpVec = None,
            rw_to_u: float = None,
            lw_to_u: float = None,
        ):
            sons = g3_edge_dict.get(u, [])
            for v in sons:
                lw_uv = g3_edge_info[get_g3_edge_id((u, v))].lw
                rw_uv = g3_edge_info[get_g3_edge_id((u, v))].rw
                if not g3_node_id_less(u, v):
                    lw_uv, rw_uv = rw_uv, lw_uv
                # if not same point
                u_pos = g3_node_pos[u]
                v_pos = g3_node_pos[v]
                if not Point(u_pos).equals(Point(v_pos)):
                    polygons.append(
                        asymmetric_buffer_linestring_v1(
                            LineString([u_pos, v_pos]),
                            lw_uv / 2.0,
                            rw_uv / 2.0,
                        )
                    )
                dir_uv = (
                    normalized(v_pos - u_pos)
                    if not Point(u_pos).equals(Point(v_pos))
                    else dir_to_u
                )
                dir_uv_left = dir_left(dir_uv)
                if dir_to_u is not None:
                    dir_to_u_left = dir_left(dir_to_u)
                    if dir_to_u_left @ dir_uv > 0:  # 左转
                        polygons.append(
                            Polygon(
                                [
                                    u_pos,
                                    u_pos - dir_to_u_left * rw_to_u / 2.0,
                                    u_pos - dir_uv_left * rw_uv / 2.0,
                                ]
                            ).buffer(SHAPE_ABSORB_EPS)
                        )
                    else:
                        polygons.append(
                            Polygon(
                                [
                                    u_pos,
                                    u_pos + dir_to_u_left * lw_to_u / 2.0,
                                    u_pos + dir_uv_left * lw_uv / 2.0,
                                ]
                            ).buffer(SHAPE_ABSORB_EPS)
                        )
                dfs(v, dir_uv, rw_uv, lw_uv)

        dfs(start_g3_node)
        polygon = unary_union(polygons)
        if not isinstance(polygon, Polygon):
            # get a list of polygons from a MultiPolygon
            quick_plot(list(polygon.geoms), with_num=False)
        assert isinstance(polygon, Polygon), (
            f"Union of polygons is not a Polygon: {polygon}"
        )
        # 从 Polygon 中删除与 dest_pt 相交的点，并返回 pts
        dest_pt_pos = g3_node_pos[start_g3_node]

        # 找到首个与 dest_pt_pos 相交的线段 x, y，返回从 y 开始绕一圈到 x
        before_pts = []
        after_pts = []
        after = False
        for x, y in iter_ext_segments(polygon):
            if LineString([x, y]).contains(Point(dest_pt_pos)):
                assert not after
                before_pts.append(arr(*x))
                after = True
                continue
            if after:
                after_pts.append(arr(*x))
            else:
                before_pts.append(arr(*x))
        if not after:
            logger.error("Can't find any line containing dest_pt_pos")
            return []

        return after_pts + before_pts

    def process(self, debug: CactusSolverDebug) -> List[List[plane.NpPoint]]:
        if debug.show_wall:
            self.show_wall()
        if debug.show_regions_with_colors:
            self.show_regions_with_colors()

        region_edge_info = self.get_region_edge_info()

        # [dijk1]
        pt_edge_to, pt_dis, cac_regions_dis = self.solve_dijk1(debug=debug.solve_dijk1)

        # [dijk2]
        CactusSolver.test_get_djk_states_for_color_at_pt()
        CactusSolver.test_get_djk_states_for_color_at_pt2()
        self.test_transfer()
        edge_pipes, pt_pipe_sets, pipe_color, pipe_pt = self.dijk2(
            self.seg_pts,
            pt_edge_to,
            self.cac_regions,
            self.destination_pt,
            cac_regions_dis,
            self.g0_pipe_width,
            debug=debug.dijk2,
        )

        # [print pipe pt]

        # [xw]
        pipe_xw = CactusSolver.get_xw_for_each_pipe(
            self.cac_regions,
            self.seg_pts,
            self.wall_pt_path,
            edge_pipes,
            self.g0_pipe_width,
            region_edge_info,
        )
        if debug.xw:
            self.test_plot_pipes(edge_pipes, pipe_color, pipe_xw)
        # for k, v in pipe_xw.items():
        #     MES = f"pipe {k} has nan member. {v}"
        #     assert not np.isnan(v.x), MES
        #     assert not np.isnan(v.rw), MES
        #     assert not np.isnan(v.lw), MES

        # [g2s1]
        node_set_s1, edge_dict_s1, node_pos_s1, edge_info_s1 = (
            CactusSolver.get_endpoint_for_each_pipe(
                self.seg_pts, pt_edge_to, edge_pipes, pipe_xw
            )
        )
        if debug.g2s1:
            self.test_plot_pipes2(edge_pipes, pipe_color, pipe_pt, node_pos_s1)

        # [g2s2]
        node_set_s2, edge_dict_s2, edge_info_s2 = CactusSolver.build_linked_g2(
            node_set_s1, edge_dict_s1, pt_pipe_sets, edge_info_s1, pipe_pt
        )
        for k, v in edge_dict_s2.items():
            assert k in node_set_s2
            assert all([x in node_set_s2 for x in v])
        if debug.g2s2:
            self.test_g2_s2(edge_dict_s2, node_pos_s1, pipe_color)

        # [g2s3]
        g2_node_set_s3, g2_edge_dict_s3, g2_node_pos_s3, g2_edge_info_s3 = (
            CactusSolver.g2_unique_xy(
                node_set_s2,
                edge_dict_s2,
                node_pos_s1,
                edge_info_s2,
                self.destination_pt,
            )
        )

        # print(f"g2_node_set_s3: {edge_info_s2[G2Node(pipe_id=764, pt_id=92), G2Node(pipe_id=973, pt_id=92)]}")

        if debug.g2s3:
            self.test_g2_s3(g2_edge_dict_s3, g2_node_pos_s3, pipe_color)

        # [g3]
        g2_start_nodes = CactusSolver.g2_get_start_nodes(
            g2_node_set_s3, pipe_color, self.destination_pt
        )
        if debug.g3:
            self.test_g3(
                g2_start_nodes,  # you can adjust this
                pipe_color,
                g2_node_set_s3,
                g2_edge_dict_s3,
                g2_node_pos_s3,
                g2_edge_info_s3,
                self.g0_pipe_width,
            )

        # [m1]
        pipe_pt_seq: List[List[plane.NpPoint]] = []
        for s in g2_start_nodes:
            n, e, p, i = CactusSolver.g3_tarjan_for_a_color(
                s,
                g2_node_set_s3,
                g2_edge_dict_s3,
                g2_node_pos_s3,
                g2_edge_info_s3,
                self.g0_pipe_width,
            )
            seq = CactusSolver.gen_one_color_m1_by_shapely(
                s,
                self.destination_pt,
                e,
                p,
                i,
            )
            pipe_pt_seq.append(seq)

        if debug.m1:
            plt.figure(figsize=(20, 10))
            plt.axis("equal")
            self.plot_wall(plt)
            for s, pts in zip(g2_start_nodes, pipe_pt_seq):
                for i in range(len(pts) - 1):
                    plt.plot(
                        [pts[i][0], pts[i + 1][0]],
                        [pts[i][1], pts[i + 1][1]],
                        color=self.cmap[pipe_color[s.pipe_id]],
                        linewidth=1,
                    )
            plt.show()

        return pipe_pt_seq


def calc_pipe_length(li: List[plane.NpPoint]):
    ret = 0.0
    for i in range(1, len(li)):
        ret += np.linalg.norm(li[i] - li[i - 1])
    return ret


if __name__ == "__main__":
    # [---]

    example_filename = "../pypipeline/output/tmp.json"

    SEG_PTS, CAC_REGIONS_FAKE, WALL_PT_PATH, DESTINATION_PT, interval = (
        parse_polygon_group_intermediate(example_filename)
    )

    SEG_PTS = [arr(x[0], x[1]) for x in SEG_PTS]

    CAC_REGIONS_FAKE = [CacRegion(x[0][::1], x[1]) for x in CAC_REGIONS_FAKE]

    cmap_0 = [
        "blue",
        "yellow",
        "red",
        "cyan",
        "green",
        "purple",
        "orange",
        "pink",
    ]
    cmap = {
        -1: "black",
        0: "grey",
        **{(i + 1): cmap_0[i % len(cmap_0)] for i in range(100)},
    }

    solver = CactusSolver(
        cmap=cmap,
        seg_pts=SEG_PTS,
        wall_pt_path=WALL_PT_PATH,
        cac_regions=CAC_REGIONS_FAKE,
        destination_pt=DESTINATION_PT,
        suggested_m0_pipe_interval=interval,
    )

    pipe_pt_seq = solver.process(
        CactusSolverDebug(
            show_regions_with_colors=True,
            g2s1=True,
            g2s2=True,
            g2s3=True,
            g3=True,
            m1=True,
        )
    )
    for idx, pts in enumerate(pipe_pt_seq):
        print(f"pipe {idx} length: {calc_pipe_length(pts)} (m)")
        # print(f"pipe {idx}: {pts}")
