import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, MultiPoint, Point
from shapely.ops import split, unary_union

# from data.test_data import *

random.seed(1234)

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
            try:
                if piece.intersects(line):
                    splitted = split(piece, line)
                    new_pieces.extend(list(splitted.geoms))
                else:
                    new_pieces.append(piece)
            except Exception as e:
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
            ax.plot(pt[0], pt[1], 'bo', markersize=3)
            ax.text(pt[0], pt[1], str(idx), fontsize=5, color='blue',
                    verticalalignment='bottom', horizontalalignment='right')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')
    plt.show()

def compute_signed_area(points):
    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i+1)%n]
        area += (x1 * y2 - x2 * y1)
    return area/2

# def largest_inscribed_rect(poly):
#     minx, miny, maxx, maxy = poly.bounds
#     best_rect = None
#     max_area = 0

#     # 按行扫描，分割为小块
#     for y1 in np.linspace(miny, maxy, num=20):
#         for y2 in np.linspace(y1, maxy, num=20):
#             for x1 in np.linspace(minx, maxx, num=20):
#                 for x2 in np.linspace(x1, maxx, num=20):
#                     rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
#                     # 检查矩形是否完全包含在多边形内
#                     if poly.contains(rect):
#                         area = rect.area
#                         if area > max_area:
#                             max_area = area
#                             best_rect = rect

#     return best_rect, max_area

def largest_inscribed_rect(poly):
    minx, miny, maxx, maxy = poly.bounds
    best_rect = None
    max_area = 0

    # 获取多边形的外边界（线段），并按x坐标排序
    exterior = poly.exterior.coords
    x_coords = [p[0] for p in exterior]
    y_coords = [p[1] for p in exterior]
    
    # 扫描线法：遍历所有可能的上下边界 y1, y2
    for y1 in np.unique(y_coords):  # 只遍历多边形的y坐标（减少不必要的搜索）
        for y2 in np.unique(y_coords):
            if y1 >= y2:  # 确保 y1 < y2
                continue

            # 对于每个y值，找到外边界线上所有交点的x坐标
            valid_xs = []
            for i in range(len(exterior) - 1):
                x1, y1_exterior = exterior[i]
                x2, y2_exterior = exterior[i + 1]
                
                # 判断线段是否与当前y1, y2范围相交
                if (y1_exterior <= y1 <= y2_exterior or y2_exterior <= y1 <= y1_exterior):
                    if y1_exterior == y2_exterior:  # 水平线段，跳过
                        continue
                    # 计算线段与水平线的交点
                    try:
                        intersect_x = x1 + (y1 - y1_exterior) * (x2 - x1) / (y2_exterior - y1_exterior)
                        valid_xs.append(intersect_x)
                    except ZeroDivisionError:
                        # 如果出现除零错误，忽略该线段
                        continue

            valid_xs = sorted(valid_xs)  # 排序x坐标

            if len(valid_xs) < 2:
                continue  # 如果 x 坐标数量小于 2，跳过
            
            # 遍历有效的 x 坐标对，生成矩形并检查其是否在多边形内
            for i in range(len(valid_xs)):
                for j in range(i+1, len(valid_xs)):
                    x1, x2 = valid_xs[i], valid_xs[j]
                    rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                    # 检查矩形是否有效
                    if not rect.is_valid:
                        continue  # 跳过无效的矩形
                    
                    try:
                        # 检查矩形是否在多边形内
                        if poly.contains(rect):
                            area = rect.area
                            if area > max_area:
                                max_area = area
                                best_rect = rect
                    except Exception as e:
                        continue

    return best_rect, max_area

# 对多边形进行切割，分离出“多余部分”
def cut_extra_parts(poly):
    # 获取一个内接的最大内接矩形作为核心区域
    core, core_area = largest_inscribed_rect(poly)
    
    # 如果没有找到合适的矩形或者矩形占比太小，直接返回原始多边形
    if core_area == 0 or core_area < poly.area * 0.1:  # 设定一个阈值，避免小矩形无效切割
        return [poly]
    
    # 计算差集 poly - core，得到外部区域
    remainder = poly.difference(core)
    
    # 确保切割后的结果不为空
    if remainder.is_empty:
        return [core]  # 如果差集为空，返回核心区域
    
    # 返回核心区域和多余部分
    pieces = [core]
    
    if remainder.geom_type == 'Polygon':
        pieces.append(remainder)
    elif remainder.geom_type == 'MultiPolygon':
        for p in remainder.geoms:
            pieces.append(p)
    
    return pieces

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
        return (bxmax - bxmin) * (bymin - bymax)

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

        edge_num = len(poly.exterior.coords)

        score = area_ratio - penalty
        score = score - edge_num / 3

        return score


    # -------------------- Step 3: 循环合并面积较小的分区 --------------------
    merge_count = 0
    while True:
        # nodes_sorted = sorted(G.nodes, key=lambda n: G.nodes[n]['area'])
        nodes_list = list(G.nodes)
        random.shuffle(nodes_list)
        merged_flag = False
        for node_id in nodes_list:
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
                    G.add_node( new_node_id, geometry=best_merged_poly,
                                area=best_merged_poly.area,
                                bounds=best_merged_poly.bounds)
                    all_adj = set(G.neighbors(node_id)) | set(G.neighbors(best_neighbor_id))
                    all_adj.discard(node_id)
                    all_adj.discard(best_neighbor_id)
                    for adj_id in all_adj:
                        try:
                            if best_merged_poly.touches(G.nodes[adj_id]['geometry']):
                                inter = best_merged_poly.intersection(G.nodes[adj_id]['geometry'])
                                if inter.length > 1e-7:
                                    G.add_edge(new_node_id, adj_id)
                        except Exception as e:
                            continue
                    G.remove_node(node_id)
                    G.remove_node(best_neighbor_id)
                    merge_count += 1
                    merged_flag = True
                    break
        if not merged_flag:
            break

    final_polygons = [data['geometry'] for _, data in G.nodes(data=True)]

    # -------------------- 对每个区域进行“内接矩形切割” --------------------
    refined_polygons = []
    for poly in final_polygons:
        # 对每个区域计算最大内接矩形，并将多余部分（核心之外）切割出来
        pieces = cut_extra_parts(poly)
        refined_polygons.extend(pieces)
    final_polygons = refined_polygons
    # -------------------------------------------------------------------------

    # -------------------- Step 4: 构造全局点列表和区域信息 --------------------
    # 1. 收集所有多边形的外部边界点
    all_points = []
    for poly in final_polygons:
        pts = list(poly.exterior.coords)[:-1]
        # 判断顶点顺序，若是逆时针则反转为顺时针
        if compute_signed_area(pts) > 0:
            pts = list(reversed(pts))

        all_points.extend(pts)
    
    # 2. 创建一个字典来存储唯一的点，并为每个点分配一个唯一的索引
    unique_points = {}
    global_points = []
    index = 0
    for pt in all_points:
        # 四舍五入坐标以处理浮点精度问题
        # rounded_pt = (round(pt[0], 3), round(pt[1], 3))
        rounded_pt = pt
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
        
        region_idx = []
        for pt in pts:
            # rounded_pt = (round(pt[0], 3), round(pt[1], 3))
            rounded_pt = pt
            idx = unique_points.get(rounded_pt)
            if idx is not None:
                region_idx.append(idx)
            else:
                # 这应该不会发生，因为所有点都已经在 unique_points 中
                print(f"警告：点 {pt} 未在 unique_points 中找到。")
        region_info.append(region_idx)
    
    total_score = 0
    for poly in final_polygons:
        total_score += shape_score(poly)  # 累加得分

    return final_polygons, nat_lines, global_points, region_info, total_score

def bounding_box_aspect_ratio(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    return max(width / height, height / width)

def get_closest_ratios(target_aspect_ratio, possible_ratios):
    # 计算每种比例的长宽比，并与目标长宽比进行比较
    distances = []
    for (num_x, num_y) in possible_ratios:
        aspect_ratio = num_x / num_y
        distance = abs(aspect_ratio - target_aspect_ratio)
        distances.append((distance, num_x, num_y))
    # 按照距离排序，选取最接近的 5 个
    distances.sort()
    return [(num_x, num_y) for _, num_x, num_y in distances[:5]]

def partition_work(polygon_coords, num_x = 1, num_y = 2, collector = [0, 0]):
    polygon_coords = [(round(pt[0], 2), round(pt[1], 2)) for pt in polygon_coords]

    polygon = Polygon(polygon_coords)
    target_aspect_ratio = bounding_box_aspect_ratio(polygon)

    # 所有可能的比例
    possible_ratios = [(x, y) for x in [2, 3, 4, 5, 6] for y in [2, 3, 4, 5, 6]]

    possible_ratios = [(3, 3)]
    
    # 获取最接近的5个比例
    closest_ratios = get_closest_ratios(target_aspect_ratio, possible_ratios)

    shuffle_times = 1

    best_polygon = None  # 用来保存得分最高的多边形
    best_wall_path = None  # 用来保存得分最高的墙体路径
    best_region_info = None  # 用来保存最佳区域信息
    best_global_points = None  # 用来保存最佳全局点列表
    best_score = -float('inf')  # 初始化得分为负无穷
    best_destination_point = None

    closest_ratios = [(3, 3)]
    # 对于每个比例，运行算法
    for num_x, num_y in closest_ratios:
        print(f"Running for {num_x}x{num_y}")
        for _ in range(shuffle_times):
            final_polygons, nat_lines, global_points, region_info, score = polygon_grid_partition_and_merge(polygon_coords, num_x=num_x, num_y=num_y)

            def dis(x,y):
                return math.sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]))
            # --- 新增：清理 region_info 中的冗余共线点 ---
            def is_collinear(p_prev, p_cur, p_next, epsilon=1e-3):
                return (abs((p_cur[0]-p_prev[0])*(p_next[1]-p_prev[1]) - (p_next[0]-p_prev[0])*(p_cur[1]-p_prev[1])) < epsilon) or (dis(p_cur, p_prev) < 1)

            new_region_info = []
            for reg in region_info:
                i = 0
                while i < len(reg):
                    pi = global_points[reg[i]]
                    pi1 = global_points[reg[(i + 1) % len(reg)]]
                    for p in global_points:
                        if (dis(pi, p) > 1e-3) and (dis(p, pi1) > 1e-3) and \
                        (abs(dis(pi, p) + dis(p, pi1) - dis(pi, pi1)) < 1e-3):
                            reg = reg[:i + 1] + [global_points.index(p)] + reg[i + 1:]
                            break
                    i += 1
                new_region_info.append(reg)
            region_info = new_region_info
            
            freq = {}
            for reg in region_info:
                for idx in reg:
                    freq[idx] = freq.get(idx, 0) + 1

            cleaned_region_info = []
            for reg in region_info:
                if len(reg) < 3:
                    cleaned_region_info.append(reg)
                    continue
                cleaned = []
                n = len(reg)
                for i in range(n):
                    prev_idx = reg[i-1]
                    cur_idx = reg[i]
                    next_idx = reg[(i+1) % n]
                    p_prev = global_points[prev_idx]
                    p_cur = global_points[cur_idx]
                    p_next = global_points[next_idx]
                    # 若当前点与前后点共线且只属于本区域，则去除
                    if is_collinear(p_prev, p_cur, p_next) and freq[cur_idx] == 1:
                        continue
                    cleaned.append(cur_idx)
                if len(cleaned) < 3:
                    cleaned = reg  # 保证至少有3个点
                cleaned_region_info.append(cleaned)
            region_info = cleaned_region_info
            # --- 清理结束 ---

            # --- 新增：同步更新 global_points ---
            used_indices = set()
            for reg in region_info:
                used_indices.update(reg)
            new_mapping = {}
            new_global_points = []
            for old_idx, pt in enumerate(global_points):
                if old_idx in used_indices:
                    new_mapping[old_idx] = len(new_global_points)
                    new_global_points.append(pt)
            # 更新 region_info 中的索引
            region_info = [[ new_mapping[idx] for idx in reg ] for reg in region_info]
            global_points = new_global_points
            # --- 同步更新结束 ---

            allp = [x for x in polygon_coords]
            cleaned_allp = []
            n = len(allp)
            for i in range(n):
                p_prev = allp[i-1]
                p_cur = allp[i]
                p_next = allp[(i+1) % n]
                # 若当前点与前后点共线，则去除
                if is_collinear(p_prev, p_cur, p_next):
                    continue
                cleaned_allp.append(p_cur)
            allp = cleaned_allp
            global_points.append((round(collector[0], 2), round(collector[1], 2)))
            for p in global_points:
                l = len(allp)
                for i in range(l):
                    if (dis(allp[i], p) > 1e-3) and (dis(p, allp[(i + 1) % l]) > 1e-3) and \
                    (abs(dis(allp[i], p) + dis(p, allp[(i + 1) % l]) - dis(allp[i], allp[(i + 1) % l])) < 1e-3):
                        allp = allp[:i + 1] + [p] + allp[i + 1:]
                        break
            allp = allp[::-1]
            num_of_nodes = len(allp)

            ind = []
            for p in global_points:
                if p not in allp:
                    allp.append(p)
                ind.append(allp.index(p))

            new_region_info = []
            cnt = -1
            # threshold_area = 200
            threshold_area = 0
            
            is_collector = False
            p = (round(collector[0], 2), round(collector[1], 2))
            pid = global_points.index(p)
            for r in region_info:
                is_collector = False
                l = len(r)
                for i in range(l):
                    pi = global_points[r[i]]
                    pi1 = global_points[r[(i + 1) % l]]
                    if (dis(pi, p) > 1e-3) and (dis(p, pi1) > 1e-3) and \
                    (abs(dis(pi, p) + dis(p, pi1) - dis(pi, pi1)) < 1e-3):
                        r = r[:i + 1] + [pid] + r[i + 1:]
                        is_collector = True
                        break

                r = [ind[x] for x in r]
                cnt = cnt + 1
                
                # 获取区域的面积
                poly = final_polygons[cnt]  # 根据cnt索引找到对应的区域
                area = poly.area
                if area < 1e-5:
                    continue
                
                # 如果区域面积小于阈值，将颜色值设为-1，否则使用cnt
                color_value = -1 if area < threshold_area else cnt+1
                if is_collector:
                    color_value = 0
                    is_collector = False
                new_region_info.append((r[::-1], color_value))  # 用color_value代替cnt


            # 更新得分最高的多边形
            if score > best_score:
                best_score = score
                best_global_points = allp
                best_polygon = final_polygons
                best_wall_path = [i for i in range(num_of_nodes)]
                best_region_info = new_region_info
                best_destination_point = allp.index((round(collector[0], 2), round(collector[1], 2)))
            
            # plot_polygons(final_polygons, nat_lines=nat_lines, title="Final Merged Polygons with Global Point Indices", global_points=allp)

    print("WALL_PT_PATH=", best_wall_path)
    print("SEG_PTS=", best_global_points)
    print("CAC_REGIONS_FAKE=", best_region_info)
    print("DESTINATION_POINT=", best_destination_point)
    print("")
    plot_polygons(best_polygon, nat_lines=nat_lines, title="Final Merged Polygons with Global Point Indices", global_points=best_global_points)

    return best_polygon, best_global_points, best_region_info, best_wall_path, best_destination_point


if __name__ == "__main__":
    # work(2)
    # for i in [0,1,2,3,5]:
    #     work(i)
    floor_data = {}
    floor_data["Name"] = 'test'

    floor = [
    (6, 54), (6, 48), (18, 48), (18, 38), (18, 0), (96, 0), 
    (168, 0), (238, 0), (238, 94), (238, 158), (168, 158), 
    (98, 158), (98, 94), (98, 38), (96, 38), (94, 38), 
    (94, 41), (94, 158), (58, 158), (58, 42), (72, 42), 
    (72, 41), (72, 38), (44, 38), (44, 60), (44, 144), 
    (18, 144), (18, 60), (6, 60)
    ]
    final_polygons, allp, new_region_info, wall_path, destination_pt = partition_work(floor,3,3)
    seg_pts = [(x[0], x[1]) for x in allp]
    regions = [(r[0], r[1]) for r in new_region_info]  # 从原始数据转换

    start_point = 2
    
    def process_pipeline(key, floor_data, seg_pts, regions, wall_path, start_point):
        # 保存中间数据
        intermediate_data = {
            'floor_name': floor_data['Name'],
            'seg_pts': seg_pts,
            'regions': regions,  
            'wall_path': wall_path,
            'destination_pt': start_point,
            'pipe_interval': .1
        }

        import json
        output_file = "tmp.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

        # print(f"\n💾 中间数据已保存至: {output_file}")

        # output_file = output_dir / 'cases/case8_intermediate.json'
        # output_file = output_dir / '1_polygon_group_1_intermediate.json'
        import cactus_solver_tmp
        pipe_pt_seq = cactus_solver_tmp.solve_pipeline(output_file)
        return pipe_pt_seq

    pipe_pt_seq = process_pipeline("1", floor_data, seg_pts, regions, wall_path, start_point)
