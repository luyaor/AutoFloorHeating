import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint, Point
from shapely.ops import split, unary_union

def get_natural_segmentation_lines(polygon):
    nat_lines = []
    coords = list(polygon.exterior.coords)
    n = len(coords) - 1  # 最后一个点与第一个点相同，故减1
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    max_len = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 10
    for i in range(n):
        pt1 = coords[i]
        pt2 = coords[i+1]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        line_start = (pt1[0] - dx * max_len, pt1[1] - dy * max_len)
        line_end   = (pt2[0] + dx * max_len, pt2[1] + dy * max_len)
        infinite_line = LineString([line_start, line_end])
        splitted = split(polygon, infinite_line)
        if len(splitted.geoms) > 1:
            seg = polygon.intersection(infinite_line)
            if seg.is_empty:
                continue
            if seg.geom_type == 'LineString':
                nat_lines.append(seg)
            elif seg.geom_type == 'MultiLineString':
                for line in seg.geoms:
                    nat_lines.append(line)
    return nat_lines

def split_polygon_by_multiline(polygon, lines):
    pieces = [polygon]
    for line in lines:
        new_pieces = []
        for piece in pieces:
            if piece.intersects(line):
                try:
                    splitted = split(piece, line)
                    new_pieces.extend(list(splitted.geoms))
                except Exception as e:
                    new_pieces.append(piece)
            else:
                new_pieces.append(piece)
        pieces = new_pieces
    return pieces

def plot_polygons(polygons, nat_lines=None, title="Polygons", global_points=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        label = f"Region {i+1}" if i < 12 else None
        ax.fill(x, y, alpha=0.5, label=label)
        ax.plot(x, y, color='black', linewidth=1)
    # if nat_lines:
    #     for line in nat_lines:
    #         if line.is_empty:
    #             continue
    #         x, y = line.xy
    #         ax.plot(x, y, color='red', linewidth=2, linestyle="--", label="Natural Segmentation")
    if global_points is not None:
        for idx, pt in enumerate(global_points):
            ax.plot(pt[0], pt[1], 'bo', markersize=4)
            ax.text(pt[0], pt[1], str(idx), fontsize=10, color='blue',
                    verticalalignment='bottom', horizontalalignment='right')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')
    plt.show()

def points_equal(p1, p2, tol=1e-7):
    return abs(p1[0]-p2[0]) < tol and abs(p1[1]-p2[1]) < tol

def compute_signed_area(points):
    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i+1)%n]
        area += (x1 * y2 - x2 * y1)
    return area/2

def remove_collinear(points, tol=1e-7):
    """
    去除连续共线的点。points 为顺序排列的不重复点序列。
    """
    if len(points) < 3:
        return points[:]
    filtered = []
    n = len(points)
    for i in range(n):
        prev = points[i - 1]
        curr = points[i]
        next = points[(i + 1) % n]
        # 计算三角形面积的两倍（不除以 2），判断是否共线
        area2 = abs((curr[0] - prev[0]) * (next[1] - prev[1]) - (next[0] - prev[0]) * (curr[1] - prev[1]))
        if area2 > tol:
            filtered.append(curr)
    return filtered

def polygon_grid_partition_and_merge(polygon_coords, num_x=3, num_y=4):
    # -------------------- Step 1: 网格切分 --------------------
    polygon = Polygon(polygon_coords)
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.linspace(minx, maxx, num_x + 1)
    y_coords = np.linspace(miny, maxy, num_y + 1)
    
    grid_lines = []
    for x in x_coords:
        grid_lines.append(LineString([(x, miny), (x, maxy)]))
    for y in y_coords:
        grid_lines.append(LineString([(minx, y), (maxx, y)]))
    
    nat_lines = get_natural_segmentation_lines(polygon)
    all_lines = grid_lines + nat_lines
    sub_polygons = split_polygon_by_multiline(polygon, all_lines)
    
    grid_area = (x_coords[1]-x_coords[0])*(y_coords[1]-y_coords[0])
    merge_threshold = grid_area / 2
    max_area = grid_area * 4 / 3

    # -------------------- Step 2: 构建邻接图 --------------------
    G = nx.Graph()
    # 为每个多边形分配一个唯一的 ID
    for i, poly in enumerate(sub_polygons):
        G.add_node(i, geometry=poly, area=poly.area, bounds=poly.bounds)
    
    # 添加边时，使用 ID 而不是几何对象
    for i in range(len(sub_polygons)):
        for j in range(i + 1, len(sub_polygons)):
            if sub_polygons[i].touches(sub_polygons[j]):
                inter = sub_polygons[i].intersection(sub_polygons[j])
                if inter.length > 1e-7:
                    G.add_edge(i, j)

    def polygon_bounding_rect_area(poly):
        bxmin, bymin, bxmax, bymax = poly.bounds
        return (bxmax - bxmin) * (bymax - bymin)

    def aspect_ratio(poly):
        bxmin, bymin, bxmax, bymax = poly.bounds
        w = bxmax - bxmin
        h = bymax - bymin
        if h < 1e-9 or w < 1e-9:
            return 999999
        return max(w / h, h / w)

    def shape_score(poly):
        area = poly.area
        if area > max_area:
            return -999999
        bound_area = polygon_bounding_rect_area(poly)
        area_ratio = area / bound_area * 10
        ar = aspect_ratio(poly)
        penalty = (ar - 3) * 5 if ar > 3 else 0
        return area_ratio - penalty

    # -------------------- Step 3: 循环合并面积较小的分区 --------------------
    merge_count = 0
    while True:
        nodes_sorted = sorted(G.nodes, key=lambda n: G.nodes[n]['area'])
        merged_flag = False
        for node_id in nodes_sorted:
            if node_id not in G:
                continue
            area = G.nodes[node_id]['area']
            if area < merge_threshold:
                neighbors = list(G.neighbors(node_id))
                if not neighbors:
                    continue
                best_score = -999999
                best_neighbor_id = None
                best_merged_poly = None
                current_poly = G.nodes[node_id]['geometry']
                for nb in neighbors:
                    nb_poly = G.nodes[nb]['geometry']
                    merged_poly = current_poly.union(nb_poly)
                    if merged_poly.area > max_area:
                        continue
                    sc = shape_score(merged_poly)
                    if sc > best_score:
                        best_score = sc
                        best_neighbor_id = nb
                        best_merged_poly = merged_poly
                if best_neighbor_id is not None and best_merged_poly is not None:
                    new_node_id = max(G.nodes) + 1
                    G.add_node(new_node_id, geometry=best_merged_poly,
                               area=best_merged_poly.area,
                               bounds=best_merged_poly.bounds)
                    all_adj = set(G.neighbors(node_id)) | set(G.neighbors(best_neighbor_id))
                    all_adj.discard(node_id)
                    all_adj.discard(best_neighbor_id)
                    for adj_id in all_adj:
                        if best_merged_poly.touches(G.nodes[adj_id]['geometry']):
                            inter = best_merged_poly.intersection(G.nodes[adj_id]['geometry'])
                            if inter.length > 1e-7:
                                G.add_edge(new_node_id, adj_id)
                    G.remove_node(node_id)
                    G.remove_node(best_neighbor_id)
                    merge_count += 1
                    merged_flag = True
                    break
        if not merged_flag:
            break

    final_polygons = [data['geometry'] for _, data in G.nodes(data=True)]
    print("最终分区数：", len(final_polygons))

    # -------------------- Step 4: 构造全局点列表和区域信息 --------------------
    # 1. 收集所有多边形的外部边界点，去除重复和共线点
    all_points = []
    for poly in final_polygons:
        pts = list(poly.exterior.coords)[:-1]
        # 判断顶点顺序，若是逆时针则反转为顺时针
        if compute_signed_area(pts) > 0:
            pts = list(reversed(pts))
        # 过滤掉共线的冗余点
        pts = remove_collinear(pts)
        all_points.extend(pts)
    
    # 2. 创建一个字典来存储唯一的点，并为每个点分配一个唯一的索引
    unique_points = {}
    global_points = []
    index = 0
    for pt in all_points:
        # 四舍五入坐标以处理浮点精度问题
        rounded_pt = (round(pt[0], 7), round(pt[1], 7))
        if rounded_pt not in unique_points:
            unique_points[rounded_pt] = index
            global_points.append(rounded_pt)
            index += 1

    # 3. 为每个区域创建其边界点的索引列表
    region_info = []
    for poly in final_polygons:
        pts = list(poly.exterior.coords)[:-1]
        # 判断顶点顺序，若是逆时针则反转为顺时针
        if compute_signed_area(pts) > 0:
            pts = list(reversed(pts))
        # 过滤掉共线的冗余点
        pts = remove_collinear(pts)
        region_idx = []
        for pt in pts:
            rounded_pt = (round(pt[0], 7), round(pt[1], 7))
            idx = unique_points.get(rounded_pt)
            if idx is not None:
                region_idx.append(idx)
            else:
                # 这应该不会发生，因为所有点都已经在 unique_points 中
                print(f"警告：点 {pt} 未在 unique_points 中找到。")
        region_info.append(region_idx)

    def angle_sort_key(pt, centroid):
        """
        根据相对于重心的角度进行排序，确保顺时针排列。
        """
        x, y = pt
        cx, cy = centroid
        return math.atan2(y - cy, x - cx)
    
    def ensure_clockwise_order(polygon_points):
        """
        确保多边形顶点按顺时针方向排列。
        如果是逆时针排列，则反转顶点顺序。
        """
        # 计算多边形的面积，面积为负则表示逆时针排列
        polygon = Polygon(polygon_points)
        if polygon.area < 0:
            # 逆时针，反转顺序
            return polygon_points[::-1]
        return polygon_points

    def update_shared_points(region_info, global_points, final_polygons):
        """
        检查共享点并将其添加到相关区域的边界上，按顺时针方向正确插入。
        """
        for i, region in enumerate(region_info):
            for pt_idx in region:
                pt = global_points[pt_idx]
                point_geom = Point(pt)  # 使用 Shapely 创建一个点对象
                
                for j, other_region in enumerate(region_info):
                    if i != j:
                        # 获取另一个区域的边界
                        other_polygon = final_polygons[j]
                        if other_polygon.boundary.intersects(point_geom):
                            if pt_idx not in other_region:
                                # 确保点按正确位置插入
                                inserted = False
                                for k in range(len(other_region)):
                                    pt1_idx = other_region[k]
                                    pt2_idx = other_region[(k + 1) % len(other_region)]
                                    pt1 = global_points[pt1_idx]
                                    pt2 = global_points[pt2_idx]
                                    line = LineString([pt1, pt2])

                                    # 如果点在某条边上，则插入到这两个点之间
                                    if line.intersects(point_geom):
                                        other_region.insert(k + 1, pt_idx)
                                        inserted = True
                                        break
                                if not inserted:
                                    # 如果点不在任何边上，直接附加到末尾（作为兜底处理）
                                    other_region.append(pt_idx)
                                
                                # 确保区域最终顺时针排序
                                region_points = [global_points[idx] for idx in other_region]
                                region_points = ensure_clockwise_order(region_points)
                                other_region[:] = [global_points.index(pt) for pt in region_points]
        return region_info  

    # 更新区域信息后，确保共享的点按顺序添加
    region_info = update_shared_points(region_info, global_points, final_polygons)

    return final_polygons, nat_lines, global_points, region_info

if __name__ == "__main__":
    # case-5
    polygon_coords = [
        (6, 54), (6, 48), (18, 48), (18, 38), (18, 0), (96, 0), 
        (168, 0), (238, 0), (238, 94), (238, 158), (168, 158), 
        (98, 158), (98, 94), (98, 38), (96, 38), (94, 38), 
        (94, 41), (94, 158), (58, 158), (58, 42), (72, 42), 
        (72, 41), (72, 38), (44, 38), (44, 60), (44, 144), 
        (18, 144), (18, 60), (6, 60)
    ]

    # case-0
    # SEG_PTS = [
    #     [12250,12550],
    #     [12250,13700],
    #     [14850,13700],
    #     [14850,12550],
    #     [17250,12550],
    #     [17250,10350],
    #     [14850,10350],
    #     [14850,8850],
    #     [12250,8850],
    #     [12250,10350],
    #     [9850,10350],
    #     [9850,12550]
    # ]

    # case-1
    # SEG_PTS = [
    #     [100, 100],
    #     [9650, 100],
    #     [14850, 100],
    #     [17750, 100],
    #     [25000, 100],
    #     [27500, 100],
    #     [28000, 100],
    #     [29000, 100],
    #     [29000, 3600],
    #     [28000, 3600],
    #     [28000, 8850],
    #     [27500, 8850],
    #     [27500, 9600],
    #     [25000, 9600],
    #     [17750, 9600],
    #     [17750, 10350],
    #     [14850, 10350],
    #     [14850, 8850],
    #     [12050, 8850],
    #     [12050, 10350],
    #     [9650, 10350],
    #     [9650, 9600],
    #     [4000, 9600],
    #     [4000, 8850],
    #     [100, 8850],
    # ]

    # case-2
    # SEG_PTS = [
    #     [13800, 150],
    #     [17550, 150],
    #     [21000, 150],
    #     [25700, 150],
    #     [27450, 150],
    #     [27450, 5750],
    #     [25700, 5750],
    #     [25700, 9350],
    #     [21000, 9350],
    #     [17550, 9350],
    #     [17550, 10150],
    #     [14850, 10150],
    #     [14850, 12550],
    #     [13500, 12550],
    #     [13500, 10150],
    #     [13500, 6900],
    #     [13800, 6900],
    #     [13800, 5750]
    # ]

    # case-3
    # SEG_PTS = [
    #     [100, 900],
    #     [3400, 900],
    #     [7250, -950],
    #     [10800, 900],
    #     [13600, 900],
    #     [16600, 900],
    #     [24000, -1020],
    #     [27604,180],
    #     [29000,871],
    #     [29000,3600],
    #     [27500,3600],
    #     [27500,5800],
    #     [25750,5800],
    #     [25750,6700],
    #     [25750,9400],
    #     [24000,9400],
    #     [22250,9400],
    #     [20350,9400],
    #     [17750,9400],
    #     [17750,10150],
    #     [15050,10150],
    #     [15050,6700],
    #     [13600,6700],
    #     [9650,6700],
    #     [9650,9400],
    #     [7250,9400],
    #     [5350,9400],
    #     [3400,9400],
    #     [1850,9400],
    #     [1850,6700],
    #     [100,6700],
    #     [100,2300]
    # ]

    # polygon_coords = [(x[0]/100, x[1]/100) for x in SEG_PTS]

    num_x, num_y = 3, 4
    final_polygons, nat_lines, global_points, region_info = polygon_grid_partition_and_merge(polygon_coords, num_x=num_x, num_y=num_y)
    plot_polygons(final_polygons, nat_lines=nat_lines, title="Final Merged Polygons with Global Point Indices", global_points=global_points)
    
    print("全局点列表（按索引排列）：")
    for i, pt in enumerate(global_points):
        print(f"{i}: {pt}")
    print("\n区域信息（区域边界点索引，均按顺时针排列）：")
    for i, region in enumerate(region_info):
        print(f"Region {i+1}: {region}")
