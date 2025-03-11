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
class Fixture:
    """洁具类，如马桶、洗手台等"""
    Type: str
    Location: Point
    Size: Size
    Name: str = ""
    FixtureNo: str = ""
    FixtureDesc: str = ""

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

def create_fixture_rectangle(fixture: Fixture) -> List[Tuple[float, float]]:
    """创建洁具的矩形表示"""
    # 从洁具数据中提取位置和尺寸信息
    x = fixture.Location.x
    y = fixture.Location.y
    width = fixture.Size.Width
    height = fixture.Size.Height
    
    # 创建矩形四个角的坐标
    half_width = width / 2
    half_height = height / 2
    
    return [
        (x - half_width, y - half_height),
        (x + half_width, y - half_height),
        (x + half_width, y + half_height),
        (x - half_width, y + half_height),
        (x - half_width, y - half_height)  # 闭合多边形
    ]

def create_fixture_rectangle_dict(fixture_location, fixture_size) -> List[Tuple[float, float]]:
    """创建洁具的矩形表示 (字典版本)"""
    # 从洁具数据中提取位置和尺寸信息
    x = fixture_location[0]
    y = fixture_location[1]
    width = fixture_size[0]
    height = fixture_size[1]
    
    # 创建矩形四个角的坐标
    half_width = width / 2
    half_height = height / 2
    
    return [
        (x - half_width, y - half_height),
        (x + half_width, y - half_height),
        (x + half_width, y + half_height),
        (x - half_width, y + half_height),
        (x - half_width, y - half_height)  # 闭合多边形
    ]

def extend_fixture_to_nearest_wall(fixture: Fixture, walls: List[JCW]) -> List[Tuple[float, float]]:
    """
    选取洁具距离墙面最近的一个边延伸到墙面，扩大洁具的覆盖面积
    
    Args:
        fixture: 洁具对象
        walls: 墙面列表
        
    Returns:
        List[Tuple[float, float]]: 扩展后的洁具矩形点列表
    """
    import math
    
    # 1. 获取洁具原始矩形
    fixture_rect = create_fixture_rectangle(fixture)
    if not fixture_rect or len(fixture_rect) < 5:  # 确保有足够的点
        return fixture_rect
    
    # 2. 提取洁具的四条边
    fixture_edges = [
        (fixture_rect[0], fixture_rect[1]),  # 下边
        (fixture_rect[1], fixture_rect[2]),  # 右边
        (fixture_rect[2], fixture_rect[3]),  # 上边
        (fixture_rect[3], fixture_rect[0])   # 左边
    ]
    
    # 3. 遍历所有墙面，获取所有墙面线段
    wall_lines = []
    for wall in walls:
        # 获取墙的两条边线
        if wall.FirstLine and wall.SecondLine:
            first_line = (
                (wall.FirstLine.StartPoint.x, wall.FirstLine.StartPoint.y),
                (wall.FirstLine.EndPoint.x, wall.FirstLine.EndPoint.y)
            )
            second_line = (
                (wall.SecondLine.StartPoint.x, wall.SecondLine.StartPoint.y),
                (wall.SecondLine.EndPoint.x, wall.SecondLine.EndPoint.y)
            )
            wall_lines.append(first_line)
            wall_lines.append(second_line)
    
    if not wall_lines:
        return fixture_rect  # 如果没有墙面，直接返回原始矩形
    
    # 4. 找到洁具的每条边到墙面的最小距离
    min_distance = float('inf')
    nearest_fixture_edge_index = -1
    nearest_wall_line = None
    nearest_proj_point = None
    
    # 洁具中心点
    fixture_center = (fixture.Location.x, fixture.Location.y)
    
    for i, (edge_start, edge_end) in enumerate(fixture_edges):
        # 计算洁具边的中点
        edge_mid_x = (edge_start[0] + edge_end[0]) / 2
        edge_mid_y = (edge_start[1] + edge_end[1]) / 2
        edge_mid_point = (edge_mid_x, edge_mid_y)
        
        # 计算边的方向向量
        edge_dx = edge_end[0] - edge_start[0]
        edge_dy = edge_end[1] - edge_start[1]
        edge_length = math.sqrt(edge_dx**2 + edge_dy**2)
        
        if edge_length < 1e-6:  # 避免除以零
            continue
        
        # 边的单位方向向量
        edge_unit_dx = edge_dx / edge_length
        edge_unit_dy = edge_dy / edge_length
        
        # 计算洁具中心点到边的垂直方向向量（指向外部）
        perp_dx = -edge_unit_dy  # 垂直向量
        perp_dy = edge_unit_dx
        
        # 判断垂直向量是否指向洁具外部
        center_to_mid_dx = edge_mid_x - fixture_center[0]
        center_to_mid_dy = edge_mid_y - fixture_center[1]
        
        # 如果点积为负，需要反转垂直向量方向
        if perp_dx * center_to_mid_dx + perp_dy * center_to_mid_dy < 0:
            perp_dx = -perp_dx
            perp_dy = -perp_dy
        
        # 对于每条墙面线段，计算最短距离
        for wall_line in wall_lines:
            wall_start, wall_end = wall_line
            
            # 使用点到线段的投影计算距离
            def calculate_point_to_edge_projection(point, line_start, line_end):
                x, y = point
                x1, y1 = line_start
                x2, y2 = line_end
                
                # 计算线段长度的平方
                line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
                
                # 如果线段长度为0，返回起点和点到起点的距离
                if line_length_sq < 1e-6:
                    return line_start, math.sqrt((x - x1)**2 + (y - y1)**2)
                
                # 计算投影比例 t
                t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
                
                # 计算投影点坐标
                proj_x = x1 + t * (x2 - x1)
                proj_y = y1 + t * (y2 - y1)
                
                # 计算点到投影点的距离
                distance = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                
                return (proj_x, proj_y), distance
            
            proj_point, distance = calculate_point_to_edge_projection(edge_mid_point, wall_start, wall_end)
            
            # 检查投影的方向是否与洁具朝外的方向一致
            proj_dx = proj_point[0] - edge_mid_point[0]
            proj_dy = proj_point[1] - edge_mid_point[1]
            
            # 如果投影方向与垂直向量方向一致（点积为正）
            direction_match = (proj_dx * perp_dx + proj_dy * perp_dy) > 0
            
            if direction_match and distance < min_distance:
                min_distance = distance
                nearest_fixture_edge_index = i
                nearest_wall_line = wall_line
                nearest_proj_point = proj_point
    
    # 5. 如果找到最近的墙面，延伸洁具边到墙面
    if nearest_fixture_edge_index != -1 and nearest_wall_line and nearest_proj_point:
        # 获取需要延伸的边
        edge_start, edge_end = fixture_edges[nearest_fixture_edge_index]
        
        # 计算边的方向向量
        edge_dx = edge_end[0] - edge_start[0]
        edge_dy = edge_end[1] - edge_start[1]
        
        # 计算洁具中心点
        center_x = (fixture_rect[0][0] + fixture_rect[2][0]) / 2
        center_y = (fixture_rect[0][1] + fixture_rect[2][1]) / 2
        
        # 计算从边中点到投影点的向量
        edge_mid_x = (edge_start[0] + edge_end[0]) / 2
        edge_mid_y = (edge_start[1] + edge_end[1]) / 2
        
        to_wall_dx = nearest_proj_point[0] - edge_mid_x
        to_wall_dy = nearest_proj_point[1] - edge_mid_y
        
        # 创建新的点，将边延伸到墙面
        if nearest_fixture_edge_index == 0:  # 下边
            extended_rect = [
                (nearest_proj_point[0] - edge_dx/2, nearest_proj_point[1]),
                (nearest_proj_point[0] + edge_dx/2, nearest_proj_point[1]),
                fixture_rect[2],
                fixture_rect[3],
                (nearest_proj_point[0] - edge_dx/2, nearest_proj_point[1])
            ]
        elif nearest_fixture_edge_index == 1:  # 右边
            extended_rect = [
                fixture_rect[0],
                (nearest_proj_point[0], nearest_proj_point[1] - edge_dy/2),
                (nearest_proj_point[0], nearest_proj_point[1] + edge_dy/2),
                fixture_rect[3],
                fixture_rect[0]
            ]
        elif nearest_fixture_edge_index == 2:  # 上边
            extended_rect = [
                fixture_rect[0],
                fixture_rect[1],
                (nearest_proj_point[0] + edge_dx/2, nearest_proj_point[1]),
                (nearest_proj_point[0] - edge_dx/2, nearest_proj_point[1]),
                fixture_rect[0]
            ]
        else:  # 左边
            extended_rect = [
                fixture_rect[0],
                fixture_rect[1],
                fixture_rect[2],
                (nearest_proj_point[0], nearest_proj_point[1] + edge_dy/2),
                (nearest_proj_point[0], nearest_proj_point[1] - edge_dy/2),
                fixture_rect[0]
            ]
        
        return extended_rect
    
    # 如果找不到合适的墙面或边，返回原始矩形
    return fixture_rect

def process_ar_design(design_floor_data: dict) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float]]], Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """Process AR design data from a file path and return points in the format similar to test_data.py"""
    # # Load and convert JSON data to ARDesign
    # data = load_json_data(file_path)
    
    # 初始化结果字典和信息
    result = {}
    polygons = {}
    room_info_map = {}  # 存储房间名称和位置信息
    fixtures_info = {}  # 存储洁具信息
    
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
    
    # 处理洁具数据
    fixtures = []
    fixture_data_list = design_floor_data["Construction"].get("Fixture", [])
    
    # 输出调试信息
    print(f"\n🔍 调试 - 直接的Fixture字段存在: {bool(fixture_data_list)}")
    
    # 检查可能包含洁具信息的其他字段
    possible_fixture_fields = [
        "FurnitureAndAccessories", 
        "SanitaryFixture", 
        "Furniture", 
        "Accessories",
        "Equipment",
        "SanitaryAppliance",    # 卫生器具
        "Appliance",            # 器具
        "Fitting",              # 装置
        "Plumbing"              # 管道设备
    ]
    
    print("\n🔍 调试 - 可能包含洁具信息的字段:")
    for field in possible_fixture_fields:
        has_field = field in design_floor_data["Construction"]
        count = len(design_floor_data["Construction"].get(field, []))
        print(f"  - {field}: {'存在' if has_field else '不存在'} (项目数: {count})")
        
        # 如果字段存在并且有数据，输出第一个项目的信息
        if has_field and count > 0:
            first_item = design_floor_data["Construction"][field][0]
            print(f"    示例数据键: {list(first_item.keys())}")
            if "Name" in first_item:
                print(f"    名称: {first_item['Name']}")
    
    # 从各种可能的字段中查找洁具信息
    for field in possible_fixture_fields:
        if field in design_floor_data["Construction"]:
            items = design_floor_data["Construction"][field]
            
            # 输出调试信息
            if items:
                print(f"\n🔍 调试 - 从 {field} 字段查找洁具 (共{len(items)}项)")
                # 输出前几个项目的名称，以便了解数据结构
                for i, item in enumerate(items[:min(5, len(items))]):
                    name = item.get("Name", "未知")
                    print(f"  - 项目 {i+1}: {name}")
            
            for item in items:
                # 根据名称判断是否为洁具
                name = item.get("Name", "")
                fixture_keywords = ["马桶", "座便器", "洗手台", "洗脸盆", "浴缸", "淋浴", "洁具", "坐便器"]
                is_fixture = any(keyword in name for keyword in fixture_keywords)
                
                if is_fixture:
                    print(f"  ✅ 找到洁具: {name}")
                    try:
                        fixture = Fixture(
                            Type="洁具",
                            Location=convert_json_point(item["Location"]),
                            Size=Size(
                                Width=float(item["Size"]["Width"]),
                                Height=float(item["Size"]["Height"]),
                                Thickness=float(item.get("Size", {}).get("Thickness", 0.0))
                            ),
                            Name=name
                        )
                        fixtures.append(fixture)
                    except Exception as e:
                        print(f"  ❌ 处理洁具数据出错: {e}")
    
    # 特殊场景：检查卫生间房间，看是否可以在其中添加虚拟洁具
    bathroom_rooms = []
    
    # 在设计数据中查找卫生间
    for room in design_floor_data["Construction"]["Room"]:
        if any(keyword in room["Name"] for keyword in ["卫生间", "厕所", "洗手间", "toilet", "bathroom"]):
            bathroom_rooms.append(room)
    
    print(f"\n🔍 调试 - 找到 {len(bathroom_rooms)} 间卫生间")
    
    # 如果有卫生间但没找到洁具，我们可以在卫生间中放置虚拟洁具
    if bathroom_rooms and not fixtures:
        print("  在卫生间中放置虚拟洁具")
        for i, bathroom in enumerate(bathroom_rooms):
            # 计算卫生间中心点作为虚拟洁具位置
            boundary_points = get_points_from_room_dict(bathroom)
            
            if boundary_points:
                # 计算中心点
                centroid = get_centroid(boundary_points)
                x, y = centroid
                
                # 创建虚拟马桶
                print(f"  ✅ 添加虚拟马桶在卫生间: {bathroom['Name']}, 位置: ({x}, {y})")
                
                # 使用字典创建矩形点
                rect_points = [
                    (x - 300, y - 350),  # 左下
                    (x + 300, y - 350),  # 右下
                    (x + 300, y + 350),  # 右上
                    (x - 300, y + 350),  # 左上
                    (x - 300, y - 350),  # 闭合
                ]
                
                # 获取墙体列表
                walls = []
                if "Construction" in design_floor_data and "Wall" in design_floor_data["Construction"]:
                    for wall_data in design_floor_data["Construction"]["Wall"]:
                        if all(key in wall_data for key in ["FirstLine", "SecondLine", "Height", "Thickness"]):
                            try:
                                wall = JCW(
                                    WallName=wall_data.get("WallName", ""),
                                    Category=wall_data.get("Category", ""),
                                    Type=wall_data.get("Type", ""),
                                    FirstLine=convert_json_line(wall_data["FirstLine"]),
                                    SecondLine=convert_json_line(wall_data["SecondLine"]),
                                    Height=float(wall_data.get("Height", 0)),
                                    Thickness=float(wall_data.get("Thickness", 0))
                                )
                                walls.append(wall)
                            except Exception as e:
                                print(f"  ❌ 处理墙体数据出错: {e}")
                
                # 找到距离墙面最近的矩形边并延伸它
                # 创建临时点函数来计算位置而非使用Point类
                def extend_rectangle_to_nearest_wall(rect_points, walls, center_point):
                    """简化版的洁具延伸，直接处理矩形点"""
                    import math
                    
                    if not walls:
                        return rect_points
                    
                    # 提取矩形的四条边
                    rect_edges = [
                        (rect_points[0], rect_points[1]),  # 下边
                        (rect_points[1], rect_points[2]),  # 右边
                        (rect_points[2], rect_points[3]),  # 上边
                        (rect_points[3], rect_points[0])   # 左边
                    ]
                    
                    # 获取所有墙面线段
                    wall_lines = []
                    for wall in walls:
                        # 获取墙的两条边线
                        if wall.FirstLine and wall.SecondLine:
                            first_line = (
                                (wall.FirstLine.StartPoint.x, wall.FirstLine.StartPoint.y),
                                (wall.FirstLine.EndPoint.x, wall.FirstLine.EndPoint.y)
                            )
                            second_line = (
                                (wall.SecondLine.StartPoint.x, wall.SecondLine.StartPoint.y),
                                (wall.SecondLine.EndPoint.x, wall.SecondLine.EndPoint.y)
                            )
                            wall_lines.append(first_line)
                            wall_lines.append(second_line)
                    
                    # 定义一个交接阈值，如果洁具与墙面距离小于此值，认为已经交接
                    INTERSECTION_THRESHOLD = 5.0  # 单位通常是毫米
                    
                    cx, cy = center_point
                    
                    # 首先检查洁具是否已经与墙面交接，如果是则不需要延伸
                    for i, (edge_start, edge_end) in enumerate(rect_edges):
                        # 计算边的中点
                        edge_mid_x = (edge_start[0] + edge_end[0]) / 2
                        edge_mid_y = (edge_start[1] + edge_end[1]) / 2
                        edge_mid_point = (edge_mid_x, edge_mid_y)
                        
                        # 计算边的方向向量
                        edge_dx = edge_end[0] - edge_start[0]
                        edge_dy = edge_end[1] - edge_start[1]
                        edge_length = math.sqrt(edge_dx**2 + edge_dy**2)
                        
                        if edge_length < 1e-6:  # 避免除以零
                            continue
                        
                        # 边的单位方向向量
                        edge_unit_dx = edge_dx / edge_length
                        edge_unit_dy = edge_dy / edge_length
                        
                        # 计算矩形中心点到边的垂直方向向量（指向外部）
                        perp_dx = -edge_unit_dy  # 垂直向量
                        perp_dy = edge_unit_dx
                        
                        # 判断垂直向量是否指向矩形外部
                        center_to_mid_dx = edge_mid_x - cx
                        center_to_mid_dy = edge_mid_y - cy
                        
                        # 如果点积为负，需要反转垂直向量方向
                        if perp_dx * center_to_mid_dx + perp_dy * center_to_mid_dy < 0:
                            perp_dx = -perp_dx
                            perp_dy = -perp_dy
                        
                        # 对于每条墙面线段，检查是否已经交接
                        for wall_line in wall_lines:
                            wall_start, wall_end = wall_line
                            
                            # 使用点到线段的投影计算距离
                            def calculate_point_to_edge_projection(point, line_start, line_end):
                                x, y = point
                                x1, y1 = line_start
                                x2, y2 = line_end
                                
                                # 计算线段长度的平方
                                line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
                                
                                # 如果线段长度为0，返回起点和点到起点的距离
                                if line_length_sq < 1e-6:
                                    return line_start, math.sqrt((x - x1)**2 + (y - y1)**2)
                                
                                # 计算投影比例 t
                                t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
                                
                                # 计算投影点坐标
                                proj_x = x1 + t * (x2 - x1)
                                proj_y = y1 + t * (y2 - y1)
                                
                                # 计算点到投影点的距离
                                distance = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                                
                                return (proj_x, proj_y), distance
                            
                            proj_point, distance = calculate_point_to_edge_projection(edge_mid_point, wall_start, wall_end)
                            
                            # 检查投影的方向是否与矩形朝外的方向一致
                            proj_dx = proj_point[0] - edge_mid_point[0]
                            proj_dy = proj_point[1] - edge_mid_point[1]
                            direction_match = (proj_dx * perp_dx + proj_dy * perp_dy) > 0
                            
                            # 如果方向匹配且距离小于阈值，认为已经与墙面交接
                            if direction_match and distance < INTERSECTION_THRESHOLD:
                                print(f"  🔍 检测到洁具已与墙面交接，距离: {distance:.2f}，不需要延伸")
                                return rect_points  # 已经交接，直接返回原始矩形
                    
                    # 4. 如果没有检测到交接，找到矩形的每条边到墙面的最小距离
                    min_distance = float('inf')
                    nearest_rect_edge_index = -1
                    nearest_wall_line = None
                    nearest_proj_point = None
                    
                    for i, (edge_start, edge_end) in enumerate(rect_edges):
                        # 计算洁具边的中点
                        edge_mid_x = (edge_start[0] + edge_end[0]) / 2
                        edge_mid_y = (edge_start[1] + edge_end[1]) / 2
                        edge_mid_point = (edge_mid_x, edge_mid_y)
                        
                        # 计算边的方向向量
                        edge_dx = edge_end[0] - edge_start[0]
                        edge_dy = edge_end[1] - edge_start[1]
                        edge_length = math.sqrt(edge_dx**2 + edge_dy**2)
                        
                        if edge_length < 1e-6:  # 避免除以零
                            continue
                        
                        # 边的单位方向向量
                        edge_unit_dx = edge_dx / edge_length
                        edge_unit_dy = edge_dy / edge_length
                        
                        # 计算矩形中心点到边的垂直方向向量（指向外部）
                        perp_dx = -edge_unit_dy  # 垂直向量
                        perp_dy = edge_unit_dx
                        
                        # 判断垂直向量是否指向矩形外部
                        center_to_mid_dx = edge_mid_x - cx
                        center_to_mid_dy = edge_mid_y - cy
                        
                        # 如果点积为负，需要反转垂直向量方向
                        if perp_dx * center_to_mid_dx + perp_dy * center_to_mid_dy < 0:
                            perp_dx = -perp_dx
                            perp_dy = -perp_dy
                        
                        # 对于每条墙面线段，计算最短距离
                        for wall_line in wall_lines:
                            wall_start, wall_end = wall_line
                            
                            proj_point, distance = calculate_point_to_edge_projection(edge_mid_point, wall_start, wall_end)
                            
                            # 检查投影的方向是否与矩形朝外的方向一致
                            proj_dx = proj_point[0] - edge_mid_point[0]
                            proj_dy = proj_point[1] - edge_mid_point[1]
                            
                            # 如果投影方向与垂直向量方向一致（点积为正）
                            direction_match = (proj_dx * perp_dx + proj_dy * perp_dy) > 0
                            
                            if direction_match and distance < min_distance:
                                min_distance = distance
                                nearest_rect_edge_index = i
                                nearest_wall_line = wall_line
                                nearest_proj_point = proj_point
                    
                    # 5. 如果找到最近的墙面，延伸矩形边到墙面
                    if nearest_rect_edge_index != -1 and nearest_wall_line and nearest_proj_point:
                        # 获取需要延伸的边
                        edge_start, edge_end = rect_edges[nearest_rect_edge_index]
                        
                        # 计算边的方向向量
                        edge_dx = edge_end[0] - edge_start[0]
                        edge_dy = edge_end[1] - edge_start[1]
                        
                        # 计算矩形中心点
                        center_x = (rect_points[0][0] + rect_points[2][0]) / 2
                        center_y = (rect_points[0][1] + rect_points[2][1]) / 2
                        
                        # 计算从边中点到投影点的向量
                        edge_mid_x = (edge_start[0] + edge_end[0]) / 2
                        edge_mid_y = (edge_start[1] + edge_end[1]) / 2
                        
                        to_wall_dx = nearest_proj_point[0] - edge_mid_x
                        to_wall_dy = nearest_proj_point[1] - edge_mid_y
                        
                        # 创建新的点，将边延伸到墙面
                        if nearest_rect_edge_index == 0:  # 下边
                            # 保持垂直关系，确保左右两点的x坐标与原矩形一致
                            left_x = rect_points[0][0]
                            right_x = rect_points[1][0]
                            # 延伸的y坐标使用投影点的y坐标
                            new_y = nearest_proj_point[1]
                            
                            extended_rect = [
                                (left_x, new_y),   # 新的左下角
                                (right_x, new_y),  # 新的右下角
                                rect_points[2],     # 原来的右上角
                                rect_points[3],     # 原来的左上角
                                (left_x, new_y)     # 闭合回新的左下角
                            ]
                        elif nearest_rect_edge_index == 1:  # 右边
                            # 保持垂直关系，确保上下两点的y坐标与原矩形一致
                            bottom_y = rect_points[1][1]
                            top_y = rect_points[2][1]
                            # 延伸的x坐标使用投影点的x坐标
                            new_x = nearest_proj_point[0]
                            
                            extended_rect = [
                                rect_points[0],     # 原来的左下角
                                (new_x, bottom_y),  # 新的右下角
                                (new_x, top_y),     # 新的右上角
                                rect_points[3],     # 原来的左上角
                                rect_points[0]      # 闭合回原来的左下角
                            ]
                        elif nearest_rect_edge_index == 2:  # 上边
                            # 保持垂直关系，确保左右两点的x坐标与原矩形一致
                            left_x = rect_points[3][0]
                            right_x = rect_points[2][0]
                            # 延伸的y坐标使用投影点的y坐标
                            new_y = nearest_proj_point[1]
                            
                            extended_rect = [
                                rect_points[0],     # 原来的左下角
                                rect_points[1],     # 原来的右下角
                                (right_x, new_y),   # 新的右上角
                                (left_x, new_y),    # 新的左上角
                                rect_points[0]      # 闭合回原来的左下角
                            ]
                        else:  # 左边 (nearest_rect_edge_index == 3)
                            # 保持垂直关系，确保上下两点的y坐标与原矩形一致
                            bottom_y = rect_points[0][1]
                            top_y = rect_points[3][1]
                            # 延伸的x坐标使用投影点的x坐标
                            new_x = nearest_proj_point[0]
                            
                            extended_rect = [
                                (new_x, bottom_y),  # 新的左下角
                                rect_points[1],     # 原来的右下角
                                rect_points[2],     # 原来的右上角
                                (new_x, top_y),     # 新的左上角
                                (new_x, bottom_y)   # 闭合回新的左下角
                            ]
                        
                        return extended_rect
                    
                    # 如果找不到合适的墙面或边，返回原始矩形
                    return rect_points
                
                # 延伸虚拟马桶到最近的墙面
                extended_rect_points = extend_rectangle_to_nearest_wall(rect_points, walls, (x, y))
                
                fixture_key = f"fixture_rect_{len(fixtures)}"
                result[fixture_key] = extended_rect_points
                
                # 存储洁具信息
                fixtures_info[fixture_key] = {
                    "name": f"虚拟马桶 {i+1} (在{bathroom['Name']})",
                    "type": "虚拟洁具",
                    "centroid": (x, y)
                }
                
                # 添加虚拟洗手台（位置稍微偏移）
                sink_x = x + 800
                sink_y = y
                print(f"  ✅ 添加虚拟洗手台在卫生间: {bathroom['Name']}, 位置: ({sink_x}, {sink_y})")
                
                # 创建虚拟洗手台的矩形点
                sink_rect_points = [
                    (sink_x - 300, sink_y - 225),  # 左下
                    (sink_x + 300, sink_y - 225),  # 右下
                    (sink_x + 300, sink_y + 225),  # 右上
                    (sink_x - 300, sink_y + 225),  # 左上
                    (sink_x - 300, sink_y - 225),  # 闭合
                ]
                
                # 延伸虚拟洗手台到最近的墙面
                extended_sink_rect_points = extend_rectangle_to_nearest_wall(sink_rect_points, walls, (sink_x, sink_y))
                
                sink_fixture_key = f"fixture_rect_{len(fixtures) + 1}"
                result[sink_fixture_key] = extended_sink_rect_points
                
                # 存储洁具信息
                fixtures_info[sink_fixture_key] = {
                    "name": f"虚拟洗手台 {i+1} (在{bathroom['Name']})",
                    "type": "虚拟洁具",
                    "centroid": (sink_x, sink_y)
                }
                
                # 增加fixtures列表的计数，以便下一个洁具编号正确
                fixtures.extend([1, 1])  # 添加两个占位符
    
    # 输出找到的洁具数量
    print(f"\n🔍 调试 - 总共找到 {len(fixtures)} 个洁具")
    
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
    
    # # 3. 打印连接关系进行调试
    # print("Connections between rooms:")
    # for room_name, connected_rooms in connections.items():
    #     if connected_rooms:
    #         print(f"{room_name} is connected to: {', '.join(connected_rooms)}")
            
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
    
    for key, points in polygons.items():
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
            for room_name in group_rooms:
                if room_name in room_polygons_by_name:
                    room_info = room_polygons_by_name[room_name]
                    if 'name' in room_info and room_info['name']:
                        group_names.append(room_info['name'])
            
            # 存储多边形分组的名称和位置信息
            polygon_info_map[key] = {
                'names': group_names,
                'centroid': centroid
            }
    
    # 添加所有找到的洁具的矩形到结果中
    fixtures_list = []  # 创建一个新的列表来存储洁具信息，避免将整数放入fixtures
    print(f"\n🔍 调试 - 总共找到 {len(fixtures)} 个洁具")
    
    for k, fixture in enumerate(fixtures):
        # 检查是否是Fixture对象实例
        if hasattr(fixture, 'Location') and hasattr(fixture, 'Size'):
            # 获取墙体列表，用于延伸洁具到墙面
            walls = []
            if "Construction" in design_floor_data and "Wall" in design_floor_data["Construction"]:
                for wall_data in design_floor_data["Construction"]["Wall"]:
                    if all(key in wall_data for key in ["FirstLine", "SecondLine", "Height", "Thickness"]):
                        try:
                            wall = JCW(
                                WallName=wall_data.get("WallName", ""),
                                Category=wall_data.get("Category", ""),
                                Type=wall_data.get("Type", ""),
                                FirstLine=convert_json_line(wall_data["FirstLine"]),
                                SecondLine=convert_json_line(wall_data["SecondLine"]),
                                Height=float(wall_data.get("Height", 0)),
                                Thickness=float(wall_data.get("Thickness", 0))
                            )
                            walls.append(wall)
                        except Exception as e:
                            print(f"  ❌ 处理墙体数据出错: {e}")
            
            # 创建洁具的矩形表示
            x = fixture.Location.x
            y = fixture.Location.y
            width = fixture.Size.Width
            height = fixture.Size.Height
            
            half_width = width / 2
            half_height = height / 2
            
            rect_points = [
                (x - half_width, y - half_height),
                (x + half_width, y - half_height),
                (x + half_width, y + half_height),
                (x - half_width, y + half_height),
                (x - half_width, y - half_height)  # 闭合多边形
            ]
            
            # 延伸洁具到最近的墙面
            if 'extend_rectangle_to_nearest_wall' in locals():
                rect = extend_rectangle_to_nearest_wall(rect_points, walls, (x, y))
            else:
                # 如果函数未定义，直接使用原始矩形
                rect = rect_points
            
            result[f"fixture_rect_{k}"] = rect
            fixtures_list.append({
                "name": fixture.Name,
                "type": fixture.Type,
                "location": (fixture.Location.x, fixture.Location.y),
                "rect": rect
            })
            # 存储洁具信息
            fixtures_info[f"fixture_rect_{k}"] = {
                "name": fixture.Name,
                "type": fixture.Type,
                "centroid": (fixture.Location.x, fixture.Location.y)
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
        # 绘制洁具
        elif key.startswith("fixture_rect"):
            points_array = np.array(points)
            ax1.fill(points_array[:, 0], points_array[:, 1], 
                    color='green', alpha=0.5)
            ax1.plot(points_array[:, 0], points_array[:, 1], 
                    'g-', alpha=0.7, linewidth=2)
            
            # 注释掉添加洁具名称标注的代码
            # if fixtures_info and key in fixtures_info and 'centroid' in fixtures_info[key]:
            #     centroid = fixtures_info[key]['centroid']
            #     fixture_name = fixtures_info[key]['name']
            #     if fixture_name:  # 只有当洁具名称存在时才添加标注
            #         ax1.text(centroid[0], centroid[1], fixture_name, 
            #                 fontsize=8, ha='center', va='center', 
            #                 bbox=dict(facecolor='lightgreen', alpha=0.7, boxstyle='round,pad=0.5'))
    
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
    
    # 在第二个子图中也绘制洁具
    if fixtures_info:
        for key in fixtures_info:
            if key in original_data:
                points = original_data[key]
                points_array = np.array(points)
                ax2.fill(points_array[:, 0], points_array[:, 1], 
                        color='green', alpha=0.5)
                ax2.plot(points_array[:, 0], points_array[:, 1], 
                        'g-', alpha=0.7, linewidth=2)
                
                # 注释掉添加洁具名称标注的代码
                # if 'centroid' in fixtures_info[key]:
                #     centroid = fixtures_info[key]['centroid']
                #     fixture_name = fixtures_info[key]['name']
                #     if fixture_name:
                #         ax2.text(centroid[0], centroid[1], fixture_name, 
                #                 fontsize=8, ha='center', va='center', 
                #                 bbox=dict(facecolor='lightgreen', alpha=0.7, boxstyle='round,pad=0.5'))
    
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