from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from queue import PriorityQueue
from copy import deepcopy

# -----------------------------
# 原 cactus.py 中的一些数据结构
# -----------------------------

@dataclass
class CacRegion:
    """仙人掌区域定义"""
    ccw_pts_id: List[int]  # 逆时针顶点ID列表
    color: int             # 区域颜色

@dataclass
class PipeSegment:
    """管道线段"""
    start: Tuple[float, float]
    end: Tuple[float, float]
    color: int

# 一个便捷输入类，用于收集所需的参数
@dataclass
class PipeLayoutInput:
    seg_pts: List[Tuple[float, float]]
    regions: List[Tuple[List[int], int]]
    wall_path: List[int]
    dest_pt: int
    pipe_width: float

# -----------------------------
# cactus2.py 的主体：Solver 类
# -----------------------------
class PipeLayoutSolver:
    def __init__(self, input_data: PipeLayoutInput):
        self.input = input_data
        
        # 将所有顶点从 tuple 转为 np.array
        self.seg_pts = [np.array(pt) for pt in input_data.seg_pts]
        
        # 将 region 转为 CacRegion
        self.regions = [
            CacRegion(r[0], r[1]) 
            for r in input_data.regions
        ]
        
        # 其他输入
        self.wall_path = input_data.wall_path
        self.dest_pt = input_data.dest_pt
        self.suggested_width = input_data.pipe_width
        self.global_pipe_width = self.suggested_width * 2.0  # 模仿 cactus.py 里的 G0_PIPE_WIDTH
        
        # 以下变量在 cactus.py 中是全局的，这里收拢成 Solver 实例的属性
        self.edge_pipes = {}      # (pt1, pt2) -> EdgeInfo
        self.pt_pipe_sets = {}    # pt -> 结构 (或并查集)
        self.pipe_color = []      # pipe_id -> color
        # 这里简化为 list[int], 可以与原始 "EdgePipes" 的 ccw_pipes 概念对应
        
        # 初始化点的邻接关系
        self.pt_edge_to = [[] for _ in range(len(self.seg_pts))]
        self._init_data_structures()
    
    def _init_data_structures(self):
        """初始化点-邻接信息 (PT_EDGE_TO)"""
        for r in self.regions:
            for x, y in zip(r.ccw_pts_id, r.ccw_pts_id[1:] + [r.ccw_pts_id[0]]):
                self.pt_edge_to[x].append(y)
                self.pt_edge_to[y].append(x)
        
        for idx in range(len(self.seg_pts)):
            # 去重 + 按极角排序
            nbrs = list(set(self.pt_edge_to[idx]))
            nbrs.sort(key=lambda x: np.arctan2(
                self.seg_pts[x][1] - self.seg_pts[idx][1],
                self.seg_pts[x][0] - self.seg_pts[idx][0]
            ))
            self.pt_edge_to[idx] = nbrs
    
    # -----------------------------
    # 1) 反向 Dijkstra (类似 dijk2)
    # -----------------------------
    def _run_dijkstra(self):
        """
        模仿 cactus.py 中 dijk2 的主要逻辑：
          - 为每个带颜色的 region，尝试一个"反向"搜索，让其能与 dest_pt 连接
          - 会在 edge_pipes/pt_pipe_sets/pipe_color 中记录结果
        """
        # 为了简化，这里假设我们先把"所有区域的环"都预先建好管道
        # （在 cactus.py 里，这部分一般是"region_start_pipes = [[] for _ in ...]"等逻辑）
        
        pipe_id = 0
        for region in self.regions:
            # color == 0 表示不需要管道
            if region.color == 0:
                continue
            # 把这个区域的一圈边构建"管道"
            edges = list(zip(region.ccw_pts_id, 
                             region.ccw_pts_id[1:] + [region.ccw_pts_id[0]]))
            for (p1, p2) in edges:
                eid = tuple(sorted([p1, p2]))
                
                # 在 edge_pipes 里累加新的管道
                if eid not in self.edge_pipes:
                    self.edge_pipes[eid] = []
                self.edge_pipes[eid].append(pipe_id)
                
                # pt_pipe_sets
                if p1 not in self.pt_pipe_sets:
                    self.pt_pipe_sets[p1] = set()
                if p2 not in self.pt_pipe_sets:
                    self.pt_pipe_sets[p2] = set()
                self.pt_pipe_sets[p1].add(pipe_id)
                self.pt_pipe_sets[p2].add(pipe_id)
                
                # pipe_color
                # 每创建一条边，就给一个新的 pipe_id
                self.pipe_color.append(region.color)
                pipe_id += 1
        
        # 之后可在这里实现真正的"反向 Dijkstra"，
        # 不过要完整复刻 cactus.py，需要非常多的原始函数/数据结构，比如：
        #   • get_djk_transfer_for_color
        #   • state_attach_region
        #   • PriorityQueue + dis数组
        #   • colors_finished_states
        #   • edge_id(...) / insert_pipe(...) / mix(...) ...
        #
        # 为了演示，这里只保留"在每个区域的环上建管道"这部分核心思路，
        # 其它更复杂的状态搜索、合并管道等，需要你把 cactus.py 中的函数们也粘贴过来，
        # 再做对应的移植。
        #
        # 最终返回 edge_pipes, pt_pipe_sets, pipe_color
        return self.edge_pipes, self.pt_pipe_sets, self.pipe_color
    
    # -----------------------------
    # 2) 构建 G2 图
    # -----------------------------
    def _build_g2_graph(self, edge_pipes, pt_pipe_sets, pipe_color):
        """对应 cactus.py 中类似 g2_xxx 的过程，把 (pipe_id, pt_id) 当成节点"""
        node_set = set()
        edge_dict = {}
        node_pos = {}
        edge_info = {}
        
        # 1. 所有 (pipe_id, pt_id) 组成节点
        for pt_id, pids in pt_pipe_sets.items():
            for pid in pids:
                node = (pid, pt_id)
                node_set.add(node)
                node_pos[node] = self.seg_pts[pt_id]
                edge_dict[node] = set()
        
        # 2. 在同一个 pipe 上相邻的点之间连边
        for (p1, p2), pipe_list in edge_pipes.items():
            # p1, p2 -> 在 edge_pipes 中存了某些 pipe_id
            for pid in pipe_list:
                n1 = (pid, p1)
                n2 = (pid, p2)
                # 双向连接
                edge_dict[n1].add(n2)
                edge_dict[n2].add(n1)
                edge_info[(n1, n2)] = [pid]
                edge_info[(n2, n1)] = [pid]
        
        return node_set, edge_dict, node_pos, edge_info
    
    def _get_start_nodes(self, node_set, pipe_color):
        """
        依照 cactus.py 里  "给每个 color 找到一个离分水器最近的起始节点" 的思路
        """
        color_nodes_map = {}
        for (pid, ptid) in node_set:
            c = pipe_color[pid]
            color_nodes_map.setdefault(c, []).append((pid, ptid))
        
        start_nodes = []
        for c, nodelist in color_nodes_map.items():
            # 选出与 dest_pt 最近的 node 作为起点
            best = min(nodelist, key=lambda node: np.linalg.norm(
                self.seg_pts[node[1]] - self.seg_pts[self.dest_pt]
            ))
            start_nodes.append(best)
        
        return start_nodes
    
    # -----------------------------
    # 3) Tarjan 处理 -> 构建 G3 图
    # -----------------------------
    def _build_g3_graph(self, start_node, node_set, edge_dict, node_pos, edge_info):
        """
        模拟 cactus.py 中的 g3_tarjan_for_a_color。
        在原代码里，它是针对单个颜色做 Tarjan 寻环，然后给出一套新的"外圈+内圈"结构。
        
        这里只能给出一个基础版本，直接把 G2 所有节点拷贝过来。
        如果需要严格实现 Tarjan，需要将 cactus.py 里的 tarjan/g3_xxx 函数全部移植。
        """
        g3_node = set(node_set)
        g3_edge = {n: set(edges) for n, edges in edge_dict.items()}
        g3_pos = dict(node_pos)
        g3_info = dict(edge_info)
        return g3_node, g3_edge, g3_pos, g3_info
    
    # -----------------------------
    # 4) 路径生成 (gen_one_color_m1)
    # -----------------------------
    def _generate_pipe_path(self, start_node, g3_edge, g3_pos, g3_info, color):
        """
        在原 cactus.py 里，gen_one_color_m1 会把 ("outer", start_node) 当作起点，
        反复 DFS/BFS 生成路径。
        
        这里简化为一个 DFS，用于演示大体思路即可。
        """
        result_segments = []
        visited = set()
        stack = [start_node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for nxt in g3_edge[current]:
                if nxt not in visited:
                    seg = PipeSegment(
                        start=tuple(g3_pos[current]),
                        end=tuple(g3_pos[nxt]),
                        color=color
                    )
                    result_segments.append(seg)
                    stack.append(nxt)
                    
        return result_segments
    
    # -----------------------------
    # 组合调用
    # -----------------------------
    def solve(self) -> List[PipeSegment]:
        """
        对应 test_gen_all_color_m1 里的完整流程：
          1) 先做"反向Dijkstra"（或类似处理），得到 edge_pipes/pt_pipe_sets/pipe_color
          2) 构建 G2 图
          3) 对每个颜色创建 G3 图 + 生成管道路径
          4) 合并所有颜色管道的线段
        """
        # 1. 反向Dijkstra
        edge_pipes, pt_pipe_sets, pipe_color = self._run_dijkstra()
        
        # 2. 构建 G2
        node_set, edge_dict, node_pos, edge_info = self._build_g2_graph(
            edge_pipes, pt_pipe_sets, pipe_color
        )
        
        # 3. 获取起始节点
        start_nodes = self._get_start_nodes(node_set, pipe_color)
        
        # 4. 对每种颜色 run G3 + 路径生成
        all_segments = []
        for st in start_nodes:
            # st -> (pipe_id, pt_id)
            c = pipe_color[st[0]]
            
            # g3
            g3_node, g3_edge, g3_pos, g3_info = self._build_g3_graph(
                st, node_set, edge_dict, node_pos, edge_info
            )
            
            # 路径生成
            segs = self._generate_pipe_path(st, g3_edge, g3_pos, g3_info, c)
            all_segments.extend(segs)
        
        return all_segments


# -----------------------------
# 可视化辅助
# -----------------------------
def visualize_pipe_layout(
    pipe_segments: List[PipeSegment],
    seg_pts: List[Tuple[float, float]],
    wall_path: List[int],
    regions: List[Tuple[List[int], int]],
    dest_pt: int,
    figsize: Tuple[int, int] = (20, 10)
):
    """可视化管道布局结果"""
    # 定义颜色映射
    CMAP = {
        -1: "black",
        0: "grey",
        1: "blue",
        2: "yellow",
        3: "red",
        4: "cyan"
    }
    for i in range(5, 50):
        CMAP[i] = CMAP[i % 5]

    plt.figure(figsize=figsize)
    
    # (1) 绘制墙体
    if wall_path:
        wall_points = [seg_pts[i] for i in wall_path]
        wall_points.append(wall_points[0])  # 闭合
        xs, ys = zip(*wall_points)
        plt.plot(ys, xs, color='black', linewidth=2, label='Wall')
    
    # (2) 绘制区域边界
    for reg_pts, c in regions:
        ring = [seg_pts[i] for i in reg_pts]
        ring.append(ring[0])
        xs, ys = zip(*ring)
        plt.plot(ys, xs, '--', color=CMAP.get(c, 'grey'), alpha=0.5)
    
    # (3) 绘制管道
    for seg in pipe_segments:
        plt.plot([seg.start[1], seg.end[1]],
                 [seg.start[0], seg.end[0]],
                 color=CMAP.get(seg.color, 'grey'),
                 linewidth=2)
    
    # (4) 标记分水器
    if 0 <= dest_pt < len(seg_pts):
        dest_loc = seg_pts[dest_pt]
        plt.scatter([dest_loc[1]], [dest_loc[0]],
                    color='red', s=100, marker='*', label='Destination')
    
    plt.title("Pipe Layout Visualization")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal')
    plt.show()

# -----------------------------
# 对外暴露的便捷函数
# -----------------------------
def solve_pipe_layout(
    seg_pts: List[Tuple[float, float]],
    regions: List[Tuple[List[int], int]],
    wall_path: List[int],
    dest_pt: int,
    pipe_width: float,
    visualize: bool = True
) -> List[PipeSegment]:
    """与原 cactus.py 的 test_gen_all_color_m1 功能类似"""
    input_data = PipeLayoutInput(
        seg_pts=seg_pts,
        regions=regions,
        wall_path=wall_path,
        dest_pt=dest_pt,
        pipe_width=pipe_width
    )
    solver = PipeLayoutSolver(input_data)
    pipe_segments = solver.solve()
    
    if visualize:
        visualize_pipe_layout(
            pipe_segments=pipe_segments,
            seg_pts=seg_pts,
            wall_path=wall_path,
            regions=regions,
            dest_pt=dest_pt
        )
    return pipe_segments