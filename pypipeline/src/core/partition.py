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

def plot_polygons(polygons, title="Polygons", global_points=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, poly in enumerate(polygons):
        x, y = poly.exterior.xy
        label = f"Region {i+1}" if i < 12 else None
        ax.fill(x, y, alpha=0.5, label=label)
        ax.plot(x, y, color='black', linewidth=1)
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

# 对多边形进行切割，分离出"多余部分"
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

def polygon_grid_partition_and_merge(polygon_coords, threshold, min_area_ratio=0.2):
    """
    先使用自然分割线划分多边形，再使用网格划分，并合并小区域
    
    Args:
        polygon_coords: 多边形坐标列表
        threshold: 区域面积阈值
        min_area_ratio: 小区域占阈值的比例，用于判断是否为碎片区域
        
    Returns:
        tuple: (最终多边形列表, 全局点列表, 区域信息)
    """
    # 将坐标四舍五入到两位小数，以避免浮点精度问题
    polygon_coords = [(round(pt[0], 2), round(pt[1], 2)) for pt in polygon_coords]
    
    # 创建多边形对象
    polygon = Polygon(polygon_coords)
    
    # 步骤1: 使用自然分割线进行初步划分
    nat_lines = get_natural_segmentation_lines(polygon)
    if nat_lines:
        sub_polygons = split_polygon_by_multiline(polygon, nat_lines)
    else:
        sub_polygons = [polygon]
    
    # 步骤2: 对面积仍然超过阈值的区域进行网格划分
    grid_polygons = []
    for poly in sub_polygons:
        if poly.area > threshold:
            # 计算网格划分参数
            min_x, min_y, max_x, max_y = poly.bounds
            num_divisions = max(1, int(poly.area / threshold) + 1)
            
            # 计算合适的网格划分数量
            aspect_ratio = (max_x - min_x) / (max_y - min_y)
            
            if aspect_ratio > 1.5:
                # 横向长，增加横向划分
                num_x = math.ceil(math.sqrt(num_divisions * aspect_ratio))
                num_y = max(1, math.ceil(num_divisions / num_x))
            elif aspect_ratio < 0.67:
                # 纵向长，增加纵向划分
                num_y = math.ceil(math.sqrt(num_divisions / aspect_ratio))
                num_x = max(1, math.ceil(num_divisions / num_y))
            else:
                # 近似正方形
                num_x = math.ceil(math.sqrt(num_divisions))
                num_y = math.ceil(math.sqrt(num_divisions))
            
            # 确保至少有一个划分
            num_x = max(1, num_x)
            num_y = max(1, num_y)
            
            # 生成网格线
            x_coords = np.linspace(min_x, max_x, num_x + 1)
            y_coords = np.linspace(min_y, max_y, num_y + 1)
            
            grid_lines = []
            for x in x_coords[1:-1]:  # 排除边界线
                grid_lines.append(LineString([(x, min_y-1), (x, max_y+1)]))
            for y in y_coords[1:-1]:  # 排除边界线
                grid_lines.append(LineString([(min_x-1, y), (max_x+1, y)]))
            
            # 使用网格线划分区域
            grid_sub_polygons = split_polygon_by_multiline(poly, grid_lines)
            grid_polygons.extend(grid_sub_polygons)
        else:
            grid_polygons.append(poly)
    
    # 步骤3: 合并小区域
    # 构建邻接图
    G = nx.Graph()
    
    # 添加所有区域作为节点
    for i, poly in enumerate(grid_polygons):
        G.add_node(i, geometry=poly, area=poly.area)
    
    # 添加邻接关系
    for i in range(len(grid_polygons)):
        for j in range(i+1, len(grid_polygons)):
            if grid_polygons[i].touches(grid_polygons[j]):
                # 检查是否真的相邻（共享边，而不仅仅是点）
                intersection = grid_polygons[i].intersection(grid_polygons[j])
                if intersection.geom_type == 'LineString' or \
                   (intersection.geom_type == 'MultiLineString' and len(intersection.geoms) > 0):
                    G.add_edge(i, j)

    # 定义形状评分函数
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
        # 矩形度得分 = 面积 / 边界矩形面积（越接近1越好）
        rect_ratio = poly.area / polygon_bounding_rect_area(poly)
        # 长宽比得分，越接近1越好
        ar = aspect_ratio(poly)
        ar_score = 1 / (1 + abs(ar - 1))
        # 边数得分，边越少越好
        edge_num = len(poly.exterior.coords) - 1
        edge_score = 10 / (edge_num + 5)
        
        # 综合得分，权重可以调整
        return rect_ratio * 5 + ar_score * 3 + edge_score * 2
    
    # 合并小区域
    merge_count = 0
    min_area = threshold * min_area_ratio
    
    while True:
        # 按面积排序，优先处理小区域
        nodes_by_area = sorted(G.nodes, key=lambda n: G.nodes[n]['area'])
        merged_flag = False
        
        for node_id in nodes_by_area:
            if node_id not in G:
                continue
                
            node_area = G.nodes[node_id]['area']
            if node_area < min_area:  # 只合并被认为是碎片的小区域
                neighbors = list(G.neighbors(node_id))
                if not neighbors:
                    continue
                    
                best_score = -float('inf')
                best_neighbor = None
                best_merged_poly = None
                current_poly = G.nodes[node_id]['geometry']
                
                for nb in neighbors:
                    nb_poly = G.nodes[nb]['geometry']
                    merged_poly = current_poly.union(nb_poly)
                    
                    # 检查合并后的面积是否超过阈值
                    if merged_poly.area > threshold:
                        continue
                        
                    sc = shape_score(merged_poly)
                    if sc > best_score:
                        best_score = sc
                        best_neighbor = nb
                        best_merged_poly = merged_poly
                
                if best_neighbor is not None and best_merged_poly is not None:
                    # 创建新节点
                    new_node_id = max(G.nodes) + 1
                    G.add_node(new_node_id, geometry=best_merged_poly, area=best_merged_poly.area)
                    
                    # 更新邻接关系
                    all_neighbors = set(G.neighbors(node_id)) | set(G.neighbors(best_neighbor))
                    all_neighbors.discard(node_id)
                    all_neighbors.discard(best_neighbor)
                    
                    for neighbor in all_neighbors:
                        nb_poly = G.nodes[neighbor]['geometry']
                        if best_merged_poly.touches(nb_poly):
                            intersection = best_merged_poly.intersection(nb_poly)
                            if (intersection.geom_type == 'LineString' or 
                               (intersection.geom_type == 'MultiLineString' and len(intersection.geoms) > 0)):
                                G.add_edge(new_node_id, neighbor)
                    
                    # 移除旧节点
                    G.remove_node(node_id)
                    G.remove_node(best_neighbor)
                    
                    merged_flag = True
                    merge_count += 1
                    break
        
        if not merged_flag:
            break

    # 收集最终多边形
    final_polygons = [data['geometry'] for _, data in G.nodes(data=True)]

    # # 收集所有点并建立索引
    # all_points = []
    # for poly in final_polygons:
    #     pts = list(poly.exterior.coords)[:-1]  # 排除重复的最后一个点
    #     # 判断顶点顺序，若是逆时针则反转为顺时针
    #     if compute_signed_area(pts) > 0:
    #         pts = list(reversed(pts))
    #     all_points.extend(pts)
    
    # # 创建唯一点列表
    # unique_points = []
    # point_to_idx = {}
    # for pt in all_points:
    #     # rounded_pt = (round(pt[0], 4), round(pt[1], 4))  # 四舍五入减少浮点误差
    #     rounded_pt = pt
    #     if rounded_pt not in point_to_idx:
    #         point_to_idx[rounded_pt] = len(unique_points)
    #         unique_points.append(pt)
    
    # # 为每个区域创建点索引列表
    # region_info = []
    # for poly in final_polygons:
    #     pts = list(poly.exterior.coords)[:-1]
    #     # 判断顺序，逆时针转为顺时针
    #     if compute_signed_area(pts) > 0:
    #         pts = list(reversed(pts))
            
    #     region_indices = []
    #     for pt in pts:
    #         # rounded_pt = (round(pt[0], 4), round(pt[1], 4))
    #         rounded_pt = pt
    #         idx = point_to_idx.get(rounded_pt)
    #         if idx is not None:
    #             region_indices.append(idx)
        
    #     if len(region_indices) >= 3:  # 确保至少有3个点
    #         region_info.append(region_indices)
    
    # return final_polygons, unique_points, region_info
    return final_polygons

def plot_collector_regions(polygons, unique_points, collector_regions, collector_points_indices, title="Collector Regions"):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义颜色列表，不包括黑色（留给集水器标记）
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#B366FF']
    
    # 获取区域颜色映射（如果存在）
    region_colors = {}
    for collector_idx, data in collector_regions.items():
        if isinstance(data, dict) and 'regions' in data and 'colors' in data:
            for region, color in zip(data['regions'], data['colors']):
                region_colors[region] = color
    
    # 为每个集水器的区域分配颜色
    for collector_idx, regions in collector_regions.items():
        # 根据数据类型获取区域列表
        region_list = regions
        if isinstance(regions, dict) and 'regions' in regions:
            region_list = regions['regions']
        
        # 为该集水器的每个区域绘制
        for i, region_idx in enumerate(region_list):
            if region_idx < len(polygons):
                # 如果已有颜色映射，使用映射的颜色
                if region_idx in region_colors:
                    color_idx = region_colors[region_idx]
                    # 如果是0号颜色（包含集水器的区域或门区域），使用特殊颜色
                    if color_idx == 0:
                        color = '#FFD700'  # 金色
                    else:
                        color = colors[(color_idx - 1) % len(colors)]
                else:
                    # 否则为每个区域使用不同颜色
                    color = colors[i % len(colors)]
                
                poly = polygons[region_idx]
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, color=color)
                ax.plot(x, y, color='black', linewidth=1)
    
    # 绘制全局点
    if unique_points is not None:
        for idx, pt in enumerate(unique_points):
            ax.plot(pt[0], pt[1], 'bo', markersize=3)
            ax.text(pt[0], pt[1], str(idx), fontsize=5, color='blue',
                    verticalalignment='bottom', horizontalalignment='right')

    # 绘制集水器位置
    for i, collector_idx in enumerate(collector_points_indices):
        collector_point = unique_points[collector_idx]
        ax.plot(collector_point[0], collector_point[1], 'ko', markersize=10)
        ax.text(collector_point[0], collector_point[1], f'C{i+1}', 
                fontsize=10, color='white', 
                horizontalalignment='center', 
                verticalalignment='center')
      
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    
    # 准备图例元素
    legend_elements = []
    for i, color in enumerate(colors):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color,
                               markersize=10, label=f'Region {i+1}'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

def color_collector_regions(collector_regions, G, collector_points_indices, unique_points, door_regions):
    """
    为集水器的区域进行染色，为每个集水器的不同区域分配不同颜色，
    包含集水器的区域使用0号颜色
    
    Args:
        collector_regions: 集水器区域映射字典
        G: 区域邻接图
        collector_points_indices: 集水器点索引列表
        unique_points: 全局点列表
        
    Returns:
        dict: 区域颜色映射字典
    """
    # 初始化颜色映射
    region_colors = {}
    
    # 创建集水器点对象
    collector_points = [Point(unique_points[idx]) for idx in collector_points_indices]
    
    # 为每个集水器的区域分配颜色
    for collector_id, regions in collector_regions.items():
        if not regions:
            continue
            
        # 将颜色分配给该集水器的区域，颜色从1开始
        # for i, region in enumerate(regions):
        #     region_colors[region] = i + 1

        #     # 检查该区域是否包含集水器
        #     region_poly = G.nodes[region]['geometry']
        #     collector_point = collector_points[collector_id]
            
        #     if region_poly.contains(collector_point):
        #         # 如果包含集水器，使用0号颜色
        #         region_colors[region] = 0
        color = 0
        for i, region in enumerate(regions):
            # 将门区域的颜色设置为0
            if region in door_regions:
                region_colors[region] = 0
                continue

            # 检查该区域是否包含集水器
            region_poly = G.nodes[region]['geometry']
            collector_point = collector_points[collector_id]
            
            if region_poly.contains(collector_point):
                # 如果包含集水器，使用0号颜色
                region_colors[region] = 0
            else:
                color += 1
                region_colors[region] = color

    return region_colors

def partition_work(polygon_coords, room_infos, threshold=25000000, collectors=None, is_debug=False, door_info=None):
    """
    按房间划分区域并分配给集水器，使用得分函数优化分配结果
    
    Args:
        polygon_coords: 多边形坐标列表
        room_infos: 房间信息字典，包含每个房间的点坐标和其他信息
        threshold: 房间面积阈值，超过此值的房间将被划分
        collectors: 集水器位置列表，每个元素为(x,y)坐标
        is_debug: 是否开启调试模式
        door_info: 门信息字典，包含每个门的连接信息
    Returns:
        tuple: (最终多边形列表, 全局点列表, 区域信息, 墙路径, 集水器点索引, 集水器区域映射)
    """
    # 将坐标四舍五入到两位小数，以避免浮点精度问题
    polygon_coords = [(round(pt[0], 2), round(pt[1], 2)) for pt in polygon_coords]
    print(f"多边形点数: {len(polygon_coords)}")

    # 如果没有提供集水器位置，则使用多边形中心点
    if collectors is None or len(collectors) == 0:
        polygon = Polygon(polygon_coords)
        centroid = polygon.centroid
        collectors = [(centroid.x, centroid.y)]
    
    # 创建Shapely点对象用于距离计算
    collector_points = [Point(x, y) for x, y in collectors]
    
    # 初始化结果变量
    all_polygons = []  # 所有区域的多边形
    all_region_points = []  # 所有区域的边界点
    
    # 处理每个房间
    for room_name, room_info in room_infos.items():
        room_points = room_info['points']
        room_poly = Polygon(room_points)
        room_area = room_poly.area
        
        if room_area <= threshold:
            # 小房间不划分，直接添加
            all_polygons.append(room_poly)
            all_region_points.append(room_points)
        else:
            # 大房间需要划分
            sub_polygons = polygon_grid_partition_and_merge(room_points, threshold, min_area_ratio=0.2)
            
            # 添加划分后的子区域
            for sub_poly in sub_polygons:
                all_polygons.append(sub_poly)
                sub_points = list(sub_poly.exterior.coords)[:-1]
                all_region_points.append(sub_points)
    
    # 处理门信息，将门作为区域添加
    door_regions = []  # 存储门区域的索引
    if door_info:
        for door_id, door_data in door_info.items():
            if 'intersection_points' in door_data and len(door_data['intersection_points']) >= 2:
                # 获取门的交点
                door_points = door_data['intersection_points']
                door_points = [(round(pt[0], 2), round(pt[1], 2)) for pt in door_points]
                
                # 检查门的坐标是否都在多边形顶点列表中
                all_valid = True
                for point in door_points:
                    # 检查点是否在多边形顶点列表中（允许小误差）
                    found = False
                    for poly_point in polygon_coords:
                        dist = math.sqrt((point[0] - poly_point[0])**2 + (point[1] - poly_point[1])**2)
                        if dist < 0.1:  # 允许小误差
                            found = True
                            break
                    if not found:
                        all_valid = False
                        break
                
                if not all_valid:
                    print(f"门 {door_id} 的坐标不在多边形顶点列表中，将被丢弃")
                    continue
                
                # 确保点是顺时针排序的
                if compute_signed_area(door_points) > 0:
                    door_points = list(reversed(door_points))
                
                try:
                    # 创建门的多边形
                    door_poly = Polygon(door_points)
                    
                    # 检查多边形是否有效且面积大于0
                    if door_poly.is_valid and door_poly.area > 0:
                        # 添加到多边形列表和区域点列表
                        all_polygons.append(door_poly)
                        all_region_points.append(door_points)
                        # 记录这个区域是门
                        door_regions.append(len(all_polygons) - 1)
                    else:
                        print(f"门 {door_id} 创建的多边形无效或面积为0，将被丢弃")
                except Exception as e:
                    print(f"处理门 {door_id} 时发生错误: {e}")
    
    # 构建区域邻接图
    G = nx.Graph()
    
    # 添加所有区域作为节点
    for i, poly in enumerate(all_polygons):
        G.add_node(i, geometry=poly, area=poly.area)
    
    # 添加邻接关系
    for i in range(len(all_polygons)):
        for j in range(i+1, len(all_polygons)):
            if all_polygons[i].touches(all_polygons[j]):
                intersection = all_polygons[i].intersection(all_polygons[j])
                if intersection.geom_type == 'LineString' or \
                   (intersection.geom_type == 'MultiLineString' and len(intersection.geoms) > 0):
                    G.add_edge(i, j)
    
    # 定义距离计算函数
    def distance_to_collector(poly, collector_point):
        centroid = poly.centroid
        return ((centroid.x - collector_point.x)**2 + (centroid.y - collector_point.y)**2)**0.5
    
    # 定义分配质量评分函数
    def evaluate_assignment(collector_regions, collector_areas):
        if not collector_regions:
            return float('-inf')
            
        # 计算总面积和理想面积
        total_area = sum(collector_areas.values())
        ideal_area = total_area / len(collectors)
        
        # 1. 面积平衡性得分 (0-1)
        area_imbalance = sum(abs(area - ideal_area) for area in collector_areas.values()) / total_area
        balance_score = 1 - area_imbalance
        
        # 2. 连通性得分 (0-1)
        connectivity_score = 1.0
        for collector_id, regions in collector_regions.items():
            if not regions:
                connectivity_score = 0
                break
            # 检查该集水器的区域是否连通
            subG = G.subgraph(regions)
            if nx.number_connected_components(subG) > 1:
                connectivity_score = 0
                break
        
        # 3. 距离得分 (0-1)
        distance_score = 0
        total_distance = 0
        for collector_id, regions in collector_regions.items():
            collector_point = collector_points[collector_id]
            for region_id in regions:
                poly = all_polygons[region_id]
                total_distance += distance_to_collector(poly, collector_point)
        # 归一化距离得分
        max_possible_distance = total_area * 2  # 一个估计值
        distance_score = 1 - (total_distance / max_possible_distance)
        
        # 综合得分 (权重可调)
        weights = {
            'balance': 0.4,
            'connectivity': 0.4,
            'distance': 0.2
        }
        
        return (balance_score * weights['balance'] +
                connectivity_score * weights['connectivity'] +
                distance_score * weights['distance'])
    
    # 使用模拟退火算法优化分配
    def simulated_annealing(initial_assignment, initial_areas, temperature=1.0, cooling_rate=0.95, iterations=1000):
        current_assignment = initial_assignment.copy()
        current_areas = initial_areas.copy()
        best_assignment = current_assignment.copy()
        best_score = evaluate_assignment(current_assignment, current_areas)
        
        for _ in range(iterations):
            # 随机选择一个区域和两个集水器
            region_id = random.randint(0, len(all_polygons) - 1)
            collector1 = random.randint(0, len(collectors) - 1)
            collector2 = random.randint(0, len(collectors) - 1)
            
            if collector1 == collector2:
                continue
                
            # 找到当前区域所属的集水器
            current_collector = None
            for c, regions in current_assignment.items():
                if region_id in regions:
                    current_collector = c
                    break
            
            if current_collector is None:
                continue
            
            # 尝试移动区域
            region_area = all_polygons[region_id].area
            new_assignment = current_assignment.copy()
            new_areas = current_areas.copy()
            
            # 从当前集水器移除区域
            new_assignment[current_collector].remove(region_id)
            new_areas[current_collector] -= region_area
            
            # 添加到新集水器
            new_assignment[collector2].append(region_id)
            new_areas[collector2] += region_area
            
            # 计算新得分
            new_score = evaluate_assignment(new_assignment, new_areas)
            
            # 决定是否接受新解
            if new_score > best_score or random.random() < math.exp((new_score - best_score) / temperature):
                current_assignment = new_assignment
                current_areas = new_areas
                if new_score > best_score:
                    best_assignment = new_assignment.copy()
                    best_score = new_score
            
            temperature *= cooling_rate
        
        return best_assignment, new_areas
    
    # 初始分配：将所有区域分配给最近的集水器
    initial_assignment = {i: [] for i in range(len(collectors))}
    initial_areas = {i: 0 for i in range(len(collectors))}
    
    for i, poly in enumerate(all_polygons):
        collector_id = min(range(len(collectors)), 
                          key=lambda c: distance_to_collector(poly, collector_points[c]))
        initial_assignment[collector_id].append(i)
        initial_areas[collector_id] += poly.area
    
    # 使用模拟退火算法优化分配
    collector_regions, collector_areas = simulated_annealing(initial_assignment, initial_areas)
    
    # 后续代码保持不变...
    unique_points = list(polygon_coords)  # 全局唯一点列表
    point_to_idx = {}  # 点坐标到索引的映射
    for i, pt in enumerate(unique_points):
        rounded_pt = (round(pt[0], 2), round(pt[1], 2))
        point_to_idx[rounded_pt] = i
    
    # 收集区域中的所有点，并建立映射
    for region_points in all_region_points:
        # 判断顶点顺序，若是逆时针则反转为顺时针
        if compute_signed_area(region_points) > 0:
            region_points = list(reversed(region_points))
        for pt in region_points:
            rounded_pt = (round(pt[0], 2), round(pt[1], 2))
            if rounded_pt not in point_to_idx:
                point_to_idx[pt] = len(unique_points)
                unique_points.append(pt)

    # 为每个区域创建边界点索引列表
    region_info = []
    for region_points in all_region_points:
        # 判断顶点顺序，若是逆时针则反转为顺时针
        if compute_signed_area(region_points) > 0:
            region_points = list(reversed(region_points))
        region_indices = []
        for pt in region_points:
            rounded_pt = (round(pt[0], 2), round(pt[1], 2))
            idx = point_to_idx.get(rounded_pt)
            if idx is not None:
                region_indices.append(idx)
        if len(region_indices) >= 3:
            region_info.append(region_indices)

    def distance(x,y):
        return math.sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]))

    def is_collinear(p_prev, p_cur, p_next, epsilon=1e-3):
        return (abs((p_cur[0]-p_prev[0])*(p_next[1]-p_prev[1]) - \
            (p_next[0]-p_prev[0])*(p_cur[1]-p_prev[1])) < epsilon) or \
            (distance(p_cur, p_prev) < 1)

    # 把全局点在区域边界上的点加入到区域中
    new_region_info = []
    for region in region_info:
        for p in unique_points:
            i = 0
            while i < len(region):
                p1 = unique_points[region[i]]
                p2 = unique_points[region[(i + 1) % len(region)]]
                if (distance(p1, p) > 1e-3) and (distance(p, p2) > 1e-3) and \
                    (abs(distance(p1, p) + distance(p, p2) - distance(p1, p2)) < 1e-3):
                    region = region[:i + 1] + [unique_points.index(p)] + region[i + 1:]
                    break
                i += 1
        new_region_info.append(region)
    region_info = new_region_info
    
    freq = {}
    for region in region_info:
        for idx in region:
            freq[idx] = freq.get(idx, 0) + 1

    cleaned_region_info = []
    for region in region_info:
        if len(region) < 3:
            cleaned_region_info.append(region)
            continue
        cleaned = []
        n = len(region)
        for i in range(n):
            prev_idx = region[i-1]
            cur_idx = region[i]
            next_idx = region[(i+1) % n]
            p_prev = unique_points[prev_idx]
            p_cur = unique_points[cur_idx]
            p_next = unique_points[next_idx]
            # 若当前点与前后点共线且只属于本区域，则去除
            if is_collinear(p_prev, p_cur, p_next) and freq[cur_idx] == 1:
                continue
            cleaned.append(cur_idx)
        if len(cleaned) < 3:
            cleaned = region  # 保证至少有3个点
        cleaned_region_info.append(cleaned)
    region_info = cleaned_region_info
    # --- 清理结束 ---

    # --- 新增：同步更新 unique_points ---
    used_indices = set()
    for region in region_info:
        used_indices.update(region)
    new_mapping = {}
    new_unique_points = []
    for old_idx, pt in enumerate(unique_points):
        if old_idx in used_indices:
            new_mapping[old_idx] = len(new_unique_points)
            new_unique_points.append(pt)
    # 更新 region_info 中的索引
    region_info = [[new_mapping[idx] for idx in region] for region in region_info]
    unique_points = new_unique_points
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

    # 添加集水器点到全局点列表
    for collector in collectors:
        collector_point = (round(collector[0], 2), round(collector[1], 2))
        unique_points.append(collector_point)

    for p in unique_points:
        l = len(allp)
        for i in range(l):
            if (distance(allp[i], p) > 1e-3) and (distance(p, allp[(i + 1) % l]) > 1e-3) and \
                (abs(distance(allp[i], p) + distance(p, allp[(i + 1) % l]) - distance(allp[i], allp[(i + 1) % l])) < 1e-3):
                allp = allp[:i + 1] + [p] + allp[i + 1:]
                break
    allp = allp[::-1]
    num_of_nodes = len(allp)

    indices = []
    for p in unique_points:
        if p not in allp:
            allp.append(p)
        indices.append(allp.index(p))

    new_region_info = []
    for collector in collectors:
        p = (round(collector[0], 2), round(collector[1], 2))
        idx = unique_points.index(p)

        for r in region_info:
            l = len(r)
            for i in range(l):
                p1 = unique_points[r[i]]
                p2 = unique_points[r[(i + 1) % l]]
                if (distance(p1, p) > 1e-3) and (distance(p, p2) > 1e-3) and \
                    (abs(distance(p1, p) + distance(p, p2) - distance(p1, p2)) < 1e-3):
                    r = r[:i + 1] + [idx] + r[i + 1:]
                    break
            r = [indices[x] for x in r]
            new_region_info.append(r[::-1])
    region_info = new_region_info

    unique_points = allp
    # 添加集水器点
    collector_points_indices = []
    for collector in collectors:
        collector_point = (round(collector[0], 2), round(collector[1], 2))
        collector_points_indices.append(unique_points.index(collector_point))
    
    # 为集水器区域进行染色
    region_colors = color_collector_regions(collector_regions, G, collector_points_indices, unique_points, door_regions)
    
    # # 将门区域的颜色设置为0
    # for door_idx in door_regions:
    #     for collector_id, regions in collector_regions.items():
    #         if door_idx in regions:
    #             region_colors[door_idx] = 0
    
    # 更新集水器区域信息，包含颜色信息
    collector_region_info = {}
    for i in range(len(collectors)):
        regions = collector_regions[i]
        colors = [region_colors.get(region, 1) for region in regions]
        collector_region_info[i] = {
            'regions': regions,
            'colors': colors
        }

    # 创建墙路径
    wall_path = [i for i in range(num_of_nodes)]

    if is_debug:
        # 计算并打印统计信息
        total_area = sum(collector_areas.values())
        ideal_area_per_collector = total_area / len(collectors)
        imbalance = sum(abs(area - ideal_area_per_collector) for area in collector_areas.values()) / total_area
        
        print("\n集水器区域统计:")
        for i, area in collector_areas.items():
            polys = len(collector_regions[i])
            print(f"集水器 {i+1}: {polys} 个区域, 面积 {area:.2f} ({area/total_area*100:.1f}%)")
        print(f"总面积: {total_area:.2f}")
        print(f"理想面积: {ideal_area_per_collector:.2f}")
        print(f"不平衡度: {imbalance*100:.2f}%")
        
        # 计算连通性
        connectivity = all(1 == nx.number_connected_components(G.subgraph(regions)) 
                          for regions in collector_regions.values())
        print(f"区域连通性: {'是' if connectivity else '否'}")
        
        # 计算最终得分
        final_score = evaluate_assignment(collector_regions, collector_areas)
        print(f"最终分配得分: {final_score:.4f}")
        
        # 统计颜色使用情况
        color_usage = {}
        for color in region_colors.values():
            color_usage[color] = color_usage.get(color, 0) + 1
        print("\n颜色使用统计:")
        for color, count in sorted(color_usage.items()):
            print(f"颜色 {color}: {count} 个区域")
        
        # 统计每个集水器的颜色使用情况
        print("\n集水器颜色使用统计:")
        for i, info in collector_region_info.items():
            collector_colors = set(info['colors'])
            print(f"集水器 {i+1}: 使用颜色 {sorted(collector_colors)}")
        
        # 绘制结果
        plot_collector_regions(
            all_polygons,
            unique_points,
            collector_regions,
            collector_points_indices,
            title="Final Merged Polygons with Global Point Indices"
        )
   
    return all_polygons, unique_points, region_info, wall_path, collector_points_indices, collector_region_info


if __name__ == "__main__":
    # work(2)
    # for i in [0,1,2,3,5]:
    #     work(i)
    partition_work([
    (6, 54), (6, 48), (18, 48), (18, 38), (18, 0), (96, 0), 
    (168, 0), (238, 0), (238, 94), (238, 158), (168, 158), 
    (98, 158), (98, 94), (98, 38), (96, 38), (94, 38), 
    (94, 41), (94, 158), (58, 158), (58, 42), (72, 42), 
    (72, 41), (72, 38), (44, 38), (44, 60), (44, 144), 
    (18, 144), (18, 60), (6, 60)
],3,3)
