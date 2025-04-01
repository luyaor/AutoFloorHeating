from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

@dataclass
class Line:
    StartPoint: Point
    EndPoint: Point
    ColorIndex: int = 0
    CurveType: int = 0

@dataclass
class Size:
    Width: float
    Height: float
    Thickness: float = 0.0

@dataclass
class Door:
    Location: Point
    Size: Size
    FlipFaceNormal: Point
    BaseLine: Line = None
    DoorNo: str = ""
    DoorDesc: str = ""
    DoorType: str = ""
    Name: str = ""

@dataclass
class JCW:
    WallName: str
    Category: str
    Type: str
    FirstLine: Line
    SecondLine: Line
    Height: float
    Thickness: float
    JCWNo: str = ""
    JCWDesc: str = ""

@dataclass
class Room:
    Name: str
    Boundary: List[Line]
    RoomNo: str = ""
    RoomDesc: str = ""
    Area: float = 0.0
    Category: str = ""
    Position: str = ""

@dataclass

@dataclass
class Construction:
    Wall: List[JCW]
    Room: List[Room]
    Door: List[Door]

@dataclass
class Floor:
    Name: str
    Num: str
    LevelHeight: float
    Construction: Construction

@dataclass
class ARDesign:
    Floor: List[Floor]

# 添加独立房间类型的全局配置变量
INDEPENDENT_ROOM_TYPES = ["电梯","客梯", "前室","阳台", "风井", "设备井", "水暖井", "电井", "设备平台", "不上人屋面", "楼梯"]

def load_json_data(json_path: str) -> dict:
    """Load JSON data from file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_json_point(json_point: dict) -> Point:
    """Convert JSON point format to Point class"""
    return Point(
        x=float(json_point["x"]),
        y=float(json_point["y"]),
        z=float(json_point["z"])
    )

def convert_json_line(json_line: dict) -> Line:
    """Convert JSON line format to Line class"""
    return Line(
        StartPoint=convert_json_point(json_line["StartPoint"]),
        EndPoint=convert_json_point(json_line["EndPoint"]),
        ColorIndex=json_line.get("ColorIndex", 0),
        CurveType=json_line.get("CurveType", 0)
    )

def get_points_from_room(room: Room) -> List[Tuple[float, float]]:
    """Extract points from room boundary maintaining the original order"""
    points = []
    for line in room.Boundary:
        points.append((line.StartPoint.x, line.StartPoint.y))
    return points

def get_points_from_room_dict(room: dict) -> List[Tuple[float, float]]:
    """Extract points from room boundary dictionary maintaining the original order"""
    points = []
    for line in room["Boundary"]:
        points.append((line["StartPoint"]["x"], line["StartPoint"]["y"]))
    return points

def get_points_from_jcw(jcw: JCW) -> List[Tuple[float, float]]:
    """Extract points from JCW first and second lines"""
    return [
        (jcw.FirstLine.StartPoint.x, jcw.FirstLine.StartPoint.y),
        (jcw.FirstLine.EndPoint.x, jcw.FirstLine.EndPoint.y),
        (jcw.SecondLine.EndPoint.x, jcw.SecondLine.EndPoint.y),
        (jcw.SecondLine.StartPoint.x, jcw.SecondLine.StartPoint.y)
    ]

def get_door_points(door: Door) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get door start and end points"""
    if door.BaseLine:
        return (
            (door.BaseLine.StartPoint.x, door.BaseLine.StartPoint.y),
            (door.BaseLine.EndPoint.x, door.BaseLine.EndPoint.y)
        )
    else:
        start = (door.Location.x, door.Location.y)
        end = (
            door.Location.x + door.Size.Width * door.FlipFaceNormal.x,
            door.Location.y + door.Size.Width * door.FlipFaceNormal.y
        )
        return start, end

def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the centroid of a set of points"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return (center_x, center_y)

def sort_points_ccw(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort points in counter-clockwise order around their centroid"""
    # Get centroid
    center = get_centroid(points)
    
    # Sort points based on angle from centroid
    def get_angle(point: Tuple[float, float]) -> float:
        return math.atan2(point[1] - center[1], point[0] - center[0])
    
    return sorted(points, key=get_angle)

def is_clockwise(points: List[Tuple[float, float]]) -> bool:
    """Check if points are in clockwise order using shoelace formula"""
    if len(points) < 3:
        return False
    
    area = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return area < 0

def create_polygon_from_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Create a polygon from points, ensuring counter-clockwise order"""
    if not points:
        return points
        
    # Remove duplicate consecutive points
    unique_points = []
    for i in range(len(points)):
        if i == 0 or points[i] != points[i-1]:
            unique_points.append(points[i])
    
    # If first and last points are the same, remove the last one
    if unique_points and len(unique_points) > 1 and unique_points[0] == unique_points[-1]:
        unique_points.pop()
    
    # Check direction and reverse if clockwise
    if is_clockwise(unique_points):
        unique_points.reverse()
    
    # Close the polygon by adding the first point at the end
    if unique_points and unique_points[0] != unique_points[-1]:
        unique_points.append(unique_points[0])
    
    return unique_points

def create_door_rectangle(door: Door) -> List[Tuple[float, float]]:
    """Create a rectangle for the door based on its location, size and direction"""
    if door.BaseLine:
        # Get door direction from baseline
        dx = door.BaseLine.EndPoint.x - door.BaseLine.StartPoint.x
        dy = door.BaseLine.EndPoint.y - door.BaseLine.StartPoint.y
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return []
        
        # Normalize direction vector
        dx, dy = dx/length, dy/length
        
        # Get perpendicular vector for door thickness
        # Using significantly larger thickness to ensure connection
        # thickness = max(door.Size.Width, 100)  # Use larger thickness to ensure overlap
        thickness = door.Size.Width 
        
        # Keep original door length without extension
        extension = 0  # No extension to maintain original door length
        px, py = -dy, dx  # Perpendicular vector
        
        # Calculate four corners of the door rectangle with original length
        p1 = (door.BaseLine.StartPoint.x - dx * extension - px * thickness/2, 
              door.BaseLine.StartPoint.y - dy * extension - py * thickness/2)
        p2 = (door.BaseLine.EndPoint.x + dx * extension - px * thickness/2, 
              door.BaseLine.EndPoint.y + dy * extension - py * thickness/2)
        p3 = (door.BaseLine.EndPoint.x + dx * extension + px * thickness/2, 
              door.BaseLine.EndPoint.y + dy * extension + py * thickness/2)
        p4 = (door.BaseLine.StartPoint.x - dx * extension + px * thickness/2, 
              door.BaseLine.StartPoint.y - dy * extension + py * thickness/2)
        
        return [p1, p2, p3, p4, p1]  # Return closed polygon
    return []

def create_door_rectangle_dict(door: dict) -> List[Tuple[float, float]]:
    """Create a rectangle for the door based on its location, size and direction (dict version)"""
    if "BaseLine" in door:
        # Get door direction from baseline
        dx = door["BaseLine"]["EndPoint"]["x"] - door["BaseLine"]["StartPoint"]["x"]
        dy = door["BaseLine"]["EndPoint"]["y"] - door["BaseLine"]["StartPoint"]["y"]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return []
        
        # Normalize direction vector
        dx, dy = dx/length, dy/length
        
        # Get perpendicular vector for door thickness
        thickness = door["Size"]["Width"]
        
        # Keep original door length without extension
        extension = 0  # No extension to maintain original door length
        px, py = -dy, dx  # Perpendicular vector
        
        # Calculate four corners of the door rectangle with original length
        p1 = (door["BaseLine"]["StartPoint"]["x"] - dx * extension - px * thickness/2, 
              door["BaseLine"]["StartPoint"]["y"] - dy * extension - py * thickness/2)
        p2 = (door["BaseLine"]["EndPoint"]["x"] + dx * extension - px * thickness/2, 
              door["BaseLine"]["EndPoint"]["y"] + dy * extension - py * thickness/2)
        p3 = (door["BaseLine"]["EndPoint"]["x"] + dx * extension + px * thickness/2, 
              door["BaseLine"]["EndPoint"]["y"] + dy * extension + py * thickness/2)
        p4 = (door["BaseLine"]["StartPoint"]["x"] - dx * extension + px * thickness/2, 
              door["BaseLine"]["StartPoint"]["y"] - dy * extension + py * thickness/2)
        
        return [p1, p2, p3, p4, p1]  # Return closed polygon
    return []

def create_door_rectangle_with_options(door: Door, extension: float = 0) -> List[Tuple[float, float]]:
    """Create a rectangle for the door based on its location, size and direction with customizable extension"""
    if door.BaseLine:
        # Get door direction from baseline
        dx = door.BaseLine.EndPoint.x - door.BaseLine.StartPoint.x
        dy = door.BaseLine.EndPoint.y - door.BaseLine.StartPoint.y
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return []
        
        # Normalize direction vector
        dx, dy = dx/length, dy/length
        
        # Get perpendicular vector for door thickness
        thickness = door.Size.Width 
        
        # Use the provided extension parameter
        px, py = -dy, dx  # Perpendicular vector
        
        # Calculate four corners of the door rectangle with controlled extension
        p1 = (door.BaseLine.StartPoint.x - dx * extension - px * thickness/2, 
              door.BaseLine.StartPoint.y - dy * extension - py * thickness/2)
        p2 = (door.BaseLine.EndPoint.x + dx * extension - px * thickness/2, 
              door.BaseLine.EndPoint.y + dy * extension - py * thickness/2)
        p3 = (door.BaseLine.EndPoint.x + dx * extension + px * thickness/2, 
              door.BaseLine.EndPoint.y + dy * extension + py * thickness/2)
        p4 = (door.BaseLine.StartPoint.x - dx * extension + px * thickness/2, 
              door.BaseLine.StartPoint.y - dy * extension + py * thickness/2)
        
        return [p1, p2, p3, p4, p1]  # Return closed polygon
    return []

def merge_room_with_doors(room_points: List[Tuple[float, float]], 
                         door_rectangles: List[List[Tuple[float, float]]],
                         is_independent_room: bool = False) -> List[Tuple[float, float]]:
    """Merge room polygon with door rectangles to create connections between rooms"""
    # For independent rooms, skip door merging and return the original room points
    if is_independent_room:
        return room_points
        
    # Convert room points to Shapely polygon
    room_polygon = Polygon(room_points)
    
    # Convert door rectangles to Shapely polygons and add buffer
    door_polygons = []
    for rect in door_rectangles:
        door_poly = Polygon(rect)
        # Only consider doors that intersect with the room
        if room_polygon.intersects(door_poly):
            # Get the intersection between the room and the door
            intersection = room_polygon.intersection(door_poly)
            if not intersection.is_empty:
                # Use a significant buffer to ensure connection between rooms
                buffered = intersection.buffer(30)
                door_polygons.append(buffered)
    
    if not door_polygons:
        return room_points
    
    # Union all intersecting door areas
    door_union = unary_union(door_polygons)
    
    # Create the final polygon by unioning the room with the door areas
    merged = unary_union([room_polygon, door_union])
    
    # Simplify the resulting polygon with a smaller tolerance to preserve details
    merged = merged.simplify(0.01)
    
    # Convert back to list of points
    if isinstance(merged, MultiPolygon):
        # Take the largest polygon if multiple polygons are created
        largest = max(merged.geoms, key=lambda p: p.area)
        coords = list(largest.exterior.coords)
    else:
        coords = list(merged.exterior.coords)
    
    # Convert to list of tuples and ensure it's closed
    result = [(x, y) for x, y in coords]
    
    # Remove duplicate consecutive points
    unique_points = []
    for i in range(len(result)):
        if i == 0 or result[i] != result[i-1]:
            unique_points.append(result[i])
    
    # Ensure counter-clockwise orientation
    if is_clockwise(unique_points):
        unique_points.reverse()
    
    # Remove the last point if it's the same as the first (closing point)
    if unique_points and len(unique_points) > 1 and unique_points[0] == unique_points[-1]:
        unique_points = unique_points[:-1]
    
    return unique_points


def process_ar_design(design_floor_data: dict) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float]]], Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """Process AR design data from a file path and return points in the format similar to test_data.py"""
    
    # 初始化结果字典和信息
    result = {}
    polygons = {}
    room_info_map = {}  # 存储房间名称和位置信息
    fixtures_info = {}  # 空字典，保留这个参数以维持接口一致性
    
    # 调试信息：输出设计数据的顶级键
    print("\n🔍 调试 - 设计数据顶级键:")
    for key in design_floor_data.keys():
        print(f"  - {key}")
    
    # 查找可能包含洁具信息的字段
    if "Construction" in design_floor_data:
        print("\n🔍 调试 - Construction字段的键:")
        for key in design_floor_data["Construction"].keys():
            print(f"  - {key}")

        # 探索一些特殊的字段
        special_fields = ["ToiletAndKitchenConditionHole", "ToiletHole", "Toilet"]
        for field in special_fields:
            if field in design_floor_data:
                print(f"\n🔍 调试 - 发现特殊字段: {field}")
                print(f"  内容: {design_floor_data[field]}")
    
    # Convert rooms
    rooms = []
    for room_data in design_floor_data["Construction"]["Room"]:
        room = Room(
            Name=room_data["Name"],
            Boundary=[convert_json_line(line) for line in room_data["Boundary"]],
            Area=float(room_data["Area"]),
            Category=room_data["Category"],
            Position=room_data["Position"]
        )
        rooms.append(room)
    
    # Convert walls
    walls = []
    
    # Convert doors
    doors = []
    door_data_list = design_floor_data["Construction"].get("DoorAndWindow", [])
    for door_data in door_data_list:
        if door_data.get("Type") == "门":  # Only process door type
            door = Door(
                Location=convert_json_point(door_data["Location"]),
                Size=Size(
                    Width=float(door_data["Size"]["Width"]),
                    Height=float(door_data["Size"]["Height"]),
                    Thickness=float(door_data.get("Size", {}).get("Thickness", 0.0))
                ),
                FlipFaceNormal=convert_json_point(door_data["FlipFaceNormal"]),
                BaseLine=convert_json_line(door_data["BaseLine"]),
                Name=door_data.get("Name", ""),
                DoorType=door_data.get("DoorType", "")
            )
            doors.append(door)
    
    # First, collect all rooms and doors
    room_polygons_by_name = {}
    door_polygons = []
    
    from shapely.geometry import LineString, Point, box
    
    # 1. 先处理所有房间和门
    for floor in [design_floor_data]:  # 将ar_design.Floor改为包含design_floor_data的列表
        # 处理所有房间
        for i, room in enumerate(floor["Construction"]["Room"]):
            points = get_points_from_room_dict(room)
            # 确保点序列是闭合的，第一个点和最后一个点相同
            if points and points[0] != points[-1]:
                points.append(points[0])
            room_key = f"room_{floor['Num']}_{i}"
            result[room_key] = points
            
            # 计算房间中心点位置，用于标注房间名称
            centroid = get_centroid(points[:-1] if points and points[0] == points[-1] else points)
            room_info_map[room_key] = {
                'name': room["Name"],
                'centroid': centroid,
                'is_independent': False
            }
            
            # 检查是否为独立房间
            for independent_type in INDEPENDENT_ROOM_TYPES:
                if independent_type in room["Name"]:
                    room_info_map[room_key]['is_independent'] = True
                    break
            
            # 存储房间多边形，用于后续处理
            room_polygons_by_name[room_key] = {
                'poly': Polygon(points),
                'name': room["Name"],
                'original_points': points,
                'centroid': centroid,
                'is_independent': room_info_map[room_key]['is_independent']
            }
        
        # 处理所有门
        for j, door in enumerate(floor["Construction"]["Door"]):
            if door["BaseLine"]:
                # 创建门的矩形表示并添加更大的缓冲区确保连接
                rect = create_door_rectangle_dict(door)
                if rect:
                    door_poly = Polygon(rect)
                    
                    # 检查门是否与独立房间相交
                    touches_independent_room = False
                    for room_key, room_info in room_polygons_by_name.items():
                        if room_info.get('is_independent', False) and door_poly.intersects(room_info['poly']):
                            touches_independent_room = True
                            break
                    
                    # 如果门与独立房间相交，不添加缓冲区，否则添加缓冲区确保连接
                    if touches_independent_room:
                        # 对于与独立房间相邻的门，不添加缓冲区
                        buffered_door = door_poly
                    else:
                        # 对于普通门，添加更大的缓冲区
                        buffered_door = door_poly.buffer(0)
                        
                    # 添加一个标志表示门是否与独立房间相交
                    door_polygons.append((buffered_door, j, rect, touches_independent_room))
                    result[f"door_rect_{j}"] = rect
    
    # 2. 确定房间间的连接关系
    room_names = list(room_polygons_by_name.keys())
    from collections import defaultdict
    connections = defaultdict(set)
    
    # 记录独立房间
    independent_rooms = set()
    for room_name in room_names:
        room_info = room_polygons_by_name[room_name]
        room_actual_name = room_info.get('name', '')
        # 检查房间名称或类型是否在独立房间类型列表中
        for independent_type in INDEPENDENT_ROOM_TYPES:
            if independent_type in room_actual_name:
                independent_rooms.add(room_name)
                break
    
    # 检查每一个门是否连接两个房间
    for door_poly, door_idx, rect, touches_independent in door_polygons:
        # 如果门与独立房间相交，跳过这个门
        if touches_independent:
            continue
        
        connected_rooms = []
        
        # 检查哪些房间与这个门相交
        for room_name in room_names:
            # 跳过独立房间类型
            if room_name in independent_rooms:
                continue
                
            room_info = room_polygons_by_name[room_name]
            if door_poly.intersects(room_info['poly']):
                connected_rooms.append(room_name)
        
        # 如果门连接了两个或更多房间，记录连接关系
        if len(connected_rooms) >= 2:
            for i in range(len(connected_rooms)):
                for j in range(i+1, len(connected_rooms)):
                    connections[connected_rooms[i]].add(connected_rooms[j])
                    connections[connected_rooms[j]].add(connected_rooms[i])
    
    # 4. 寻找连通分量（连接房间的组）
    def find_connected_component(start, visited):
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.add(node)
                stack.extend(connections[node] - visited)
        return component
    
    visited = set()
    room_groups = []
    
    # 先将独立房间添加为单独的组
    for room_name in independent_rooms:
        if room_name not in visited:
            visited.add(room_name)
            room_groups.append({room_name})
    
    # 分组连接的房间
    for room_name in room_names:
        if room_name not in visited:
            component = find_connected_component(room_name, visited)
            if component:
                room_groups.append(component)
    # print(f"Room groups: {room_groups}")
    
    # 5. 合并每个组中的房间
    for i, group in enumerate(room_groups):
        if len(group) == 1:
            # 单个房间，直接使用其多边形
            room_name = next(iter(group))
            room_info = room_polygons_by_name[room_name]
            polygons[f"polygon_group_{i}"] = [p for p in room_info['original_points'][:-1]]  # 移除最后一个闭合点
            continue
        
        # 获取组中所有房间的多边形
        room_polys = []
        connecting_door_polys = []
        
        # 添加所有房间多边形
        for room_name in group:
            room_polys.append(room_polygons_by_name[room_name]['poly'])
        
        # 找出所有连接这些房间的门
        for door_poly, door_idx, rect, touches_independent in door_polygons:
            # 如果门与独立房间相交，跳过这个门
            if touches_independent:
                continue
            
            # 检查门是否连接组内的房间
            connected_to_group_rooms = []
            
            for room_name in group:
                room_info = room_polygons_by_name[room_name]
                # 检查该房间是否为独立房间，如果是则跳过
                if room_info.get('is_independent', False):
                    continue
                elif door_poly.intersects(room_info['poly']):
                    connected_to_group_rooms.append(room_name)
            
            # 如果门连接组内至少两个非独立房间，使用它进行连接
            if len(connected_to_group_rooms) >= 2:
                # 使用原始门形状，但添加更大的缓冲区以确保连接
                connecting_door_polys.append(door_poly.buffer(0))
        
        try:
            # 创建房间的并集
            all_geoms = room_polys + connecting_door_polys
            merged = unary_union(all_geoms)
            
            # 使用较小的容差，避免过度简化
            merged = merged.simplify(0.01)
            
            # 处理合并结果
            if isinstance(merged, MultiPolygon):
                # 如果生成多个多边形，选择最大的
                largest = max(merged.geoms, key=lambda p: p.area)
                coords = list(largest.exterior.coords)
            else:
                coords = list(merged.exterior.coords)
            
            # 确保没有重复的连续点
            unique_coords = []
            for i in range(len(coords)):
                if i == 0 or coords[i] != coords[i-1]:
                    unique_coords.append(coords[i])
            
            # 如果最后一个点与第一个点相同，移除最后一个点
            if unique_coords and len(unique_coords) > 1 and unique_coords[0] == unique_coords[-1]:
                unique_coords = unique_coords[:-1]
            
            points = [(x, y) for x, y in unique_coords]
            polygons[f"polygon_group_{i}"] = points
            
        except Exception as e:
            print(f"Error merging room group {i}: {e}")
            # 合并失败时，使用组内第一个房间的多边形
            if group:
                first_room = next(iter(group))
                points = room_polygons_by_name[first_room]['original_points']
                if points and len(points) > 1 and points[0] == points[-1]:
                    points = points[:-1]
                polygons[f"polygon_group_{i}"] = points

    # 6. 处理多边形，确保逆时针顺序
    processed_polygons = {}
    polygon_info_map = {}  # 存储多边形分组的名称和位置信息
    
    # for key, points in polygons.items():
    for idx, (key, points) in enumerate(polygons.items()):
        if key.startswith("polygon"):
            # 确保点序列是逆时针方向
            if is_clockwise(points):
                points = points[::-1]
            
            processed_polygons[key] = points
            
            # 为每个分组多边形计算中心点，用于标注名称
            centroid = get_centroid(points)
            
            # 查找该多边形所属的房间组
            group_idx = int(key.split('_')[-1])
            group_rooms = room_groups[group_idx] if group_idx < len(room_groups) else []
            
            # 收集组内所有房间的名称
            group_names = []
            room_infos = {}
            for room_name in room_groups[idx]:
                if room_name in room_polygons_by_name:
                    room_info = room_polygons_by_name[room_name]
                    if 'name' in room_info and room_info['name']:
                        group_names.append(room_info['name'])
                    room_basic_info = {
                        'points': room_info['original_points'][:-1],
                        'centroid': room_info['centroid']
                    }
                    room_infos[room_name] = room_basic_info
            
            # 存储多边形分组的名称和位置信息
            polygon_info_map[key] = {
                'names': group_names if len(group_names) == 1 else [],
                'centroid': centroid,
                'room_infos': room_infos
            }
    
    
    # 返回处理的数据、多边形信息、房间信息和多边形信息
    return result, processed_polygons, room_info_map, polygon_info_map, fixtures_info

def get_example_data() -> ARDesign:
    """Load and convert real JSON data to ARDesign"""
    # Load JSON data
    json_path = os.path.join("data", "ARDesign.json")
    data = load_json_data(json_path)
    
    floors = []
    for floor_data in data["Floor"]:
        # Convert rooms
        rooms = []
        for room_data in floor_data["Construction"]["Room"]:
            room = Room(
                Name=room_data["Name"],
                Boundary=[convert_json_line(line) for line in room_data["Boundary"]],
                Area=float(room_data["Area"]),
                Category=room_data["Category"],
                Position=room_data["Position"]
            )
            rooms.append(room)
        
        # Convert walls
        walls = []
        # for wall_data in floor_data["Construction"]["Wall"]:
        #     wall = JCW(
        #         WallName=wall_data["WallName"],
        #         Category=wall_data["Category"],
        #         Type=wall_data["Type"],
        #         FirstLine=convert_json_line(wall_data["FirstLine"]),
        #         SecondLine=convert_json_line(wall_data["SecondLine"]),
        #         Height=float(wall_data["Height"]),
        #         Thickness=float(wall_data["Thickness"])
        #     )
        #     walls.append(wall)
        
        # Convert doors
        doors = []
        door_data_list = floor_data["Construction"].get("DoorAndWindow", [])
        for door_data in door_data_list:
            if door_data.get("Type") == "门":  # Only process door type
                door = Door(
                    Location=convert_json_point(door_data["Location"]),
                    Size=Size(
                        Width=float(door_data["Size"]["Width"]),
                        Height=float(door_data["Size"]["Height"]),
                        Thickness=float(door_data.get("Size", {}).get("Thickness", 0.0))
                    ),
                    FlipFaceNormal=convert_json_point(door_data["FlipFaceNormal"]),
                    BaseLine=convert_json_line(door_data["BaseLine"]),
                    Name=door_data.get("Name", ""),
                    DoorType=door_data.get("DoorType", "")
                )
                doors.append(door)
        
        # Create construction
        construction = Construction(
            Wall=walls,
            Room=rooms,
            Door=doors
        )
        
        # Create floor
        floor = Floor(
            Name=floor_data["Name"],
            Num=floor_data["Num"],
            LevelHeight=float(floor_data["LevelHeight"]),
            Construction=construction
        )
        floors.append(floor)
        break
    
    return ARDesign(Floor=floors)

def plot_comparison(original_data: Dict[str, List[Tuple[float, float]]], 
                   polygons: Dict[str, List[Tuple[float, float]]], 
                   collectors: List[dict] = None,
                   room_info: Dict[str, dict] = None,
                   polygon_info: Dict[str, dict] = None,
                   fixtures_info: Dict[str, dict] = None):
    """Plot original points and processed polygons side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original points
    ax1.set_title('Original Points')
    for key, points in original_data.items():
        if key.startswith("room"):
            points_array = np.array(points)
            ax1.scatter(points_array[:, 0], points_array[:, 1], label=key)
            for i in range(len(points)):
                j = (i + 1) % len(points)
                ax1.plot([points[i][0], points[j][0]], 
                        [points[i][1], points[j][1]], 
                        'b-', alpha=0.5)
            
            # 添加房间名称标注
            if room_info and key in room_info and 'centroid' in room_info[key]:
                centroid = room_info[key]['centroid']
                room_name = room_info[key]['name']
                if room_name:  # 只有当房间名称存在时才添加标注
                    ax1.text(centroid[0], centroid[1], room_name, 
                            fontsize=9, ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        elif key.startswith("door_rect"):
            points_array = np.array(points)
            ax1.plot(points_array[:, 0], points_array[:, 1], 
                    'r-', alpha=0.7, linewidth=2)
    
    # Plot collectors if provided
    if collectors:
        for idx, collector in enumerate(collectors):
            if 'Borders' in collector:
                # 绘制集水器边界
                borders = collector['Borders']
                x_coords = []
                y_coords = []
                for border in borders:
                    start = border['StartPoint']
                    end = border['EndPoint']
                    x_coords.extend([start['x'], end['x']])
                    y_coords.extend([start['y'], end['y']])
                    
                # 绘制边界线
                ax1.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 
                        'r-', linewidth=2, label=f'Collector {idx+1}')
                # 填充集水器区域
                ax1.fill(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 
                        color='red', alpha=0.2)
    
    # Plot processed polygons
    ax2.set_title('Processed Polygons with Merged Rooms')
    colors = plt.cm.tab20(np.linspace(0, 1, len(polygons)))
    for (key, points), color in zip(polygons.items(), colors):
        if key.startswith("polygon"):
            points_array = np.array(points)
            ax2.fill(points_array[:, 0], points_array[:, 1], 
                    color=color, alpha=0.3, label=key)
            ax2.plot(points_array[:, 0], points_array[:, 1], 
                    color=color, linewidth=2)
            
            # 添加多边形分组的房间名称标注
            if polygon_info and key in polygon_info and 'centroid' in polygon_info[key]:
                centroid = polygon_info[key]['centroid']
                group_names = polygon_info[key]['names']
                if group_names:  # 只有当有房间名称时才添加标注
                    # 将组内所有房间名称合并为一个字符串
                    name_text = "\n".join(group_names)
                    ax2.text(centroid[0], centroid[1], name_text,
                            fontsize=9, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Plot collectors in the second subplot as well
    if collectors:
        for idx, collector in enumerate(collectors):
            if 'Borders' in collector:
                # 绘制集水器边界
                borders = collector['Borders']
                x_coords = []
                y_coords = []
                for border in borders:
                    start = border['StartPoint']
                    end = border['EndPoint']
                    x_coords.extend([start['x'], end['x']])
                    y_coords.extend([start['y'], end['y']])
                    
                # 绘制边界线
                ax2.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 
                        'r-', linewidth=2, label=f'Collector {idx+1}')
                # 填充集水器区域
                ax2.fill(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 
                        color='red', alpha=0.2)
    
    # Set equal aspect ratio and grid for both subplots
    for ax in [ax1, ax2]:
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add legend without duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Process the real data from file
    json_path = os.path.join("data", "ARDesign.json")
    processed_data, polygons, room_info, polygon_info, fixtures_info = process_ar_design(json_path)
    
    # Print the merged polygons points
    print("\nMerged Polygons Points:")
    for key, points in polygons.items():
        if key.startswith("polygon"):
            print(f"\n{key}:")
            print("Points = [")
            for x, y in points:
                print(f"    ({x:.2f}, {y:.2f}),")
            print("]")
            
            # Verify counter-clockwise order
            area = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
            print(f"Area (should be positive for CCW): {area/2:.2f}")
    
    # Comment out plotting code
    # plot_comparison(processed_data, polygons, [], room_info, polygon_info)
