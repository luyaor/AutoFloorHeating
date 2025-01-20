from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import os

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
    """Extract points from room boundary"""
    return [(line.StartPoint.x, line.StartPoint.y) for line in room.Boundary]

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

def process_ar_design(ar_design: ARDesign) -> Dict[str, List[Tuple[float, float]]]:
    """Process AR design data and return points in the format similar to test_data.py"""
    result = {}
    
    for floor in ar_design.Floor:
        # Process rooms
        for i, room in enumerate(floor.Construction.Room):
            points = get_points_from_room(room)
            result[f"room_{floor.Num}_{i}"] = points
        
        # Process walls (JCWs)
        for i, wall in enumerate(floor.Construction.Wall):
            points = get_points_from_jcw(wall)
            result[f"wall_{floor.Num}_{i}"] = points
        
        # Process doors
        for i, door in enumerate(floor.Construction.Door):
            start, end = get_door_points(door)
            result[f"door_{floor.Num}_{i}"] = [start, end]
    
    return result

def get_example_data() -> ARDesign:
    """Load and convert real JSON data to ARDesign"""
    # Load JSON data
    json_path = os.path.join("example", "ARDesign.json")
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

if __name__ == "__main__":
    # Load and process the real data
    ar_design = get_example_data()
    processed_data = process_ar_design(ar_design)
    
    # Print the processed data for verification
    print("Processed AR Design Data:")
    for key, points in processed_data.items():
        print(f"\n{key}:")
        print(f"Points: {points}")
        
    # # Print floor information
    # print("\nFloor Information:")
    # for floor in ar_design.Floor:
    #     print(f"\nFloor {floor.Name} (Number: {floor.Num}):")
    #     print(f"Level Height: {floor.LevelHeight}")
    #     print(f"Number of Rooms: {len(floor.Construction.Room)}")
    #     print(f"Number of Walls: {len(floor.Construction.Wall)}")
    #     print(f"Number of Doors: {len(floor.Construction.Door)}")
        
    #     # Print room details
    #     print("\nRooms:")
    #     for room in floor.Construction.Room:
    #         print(f"- {room.Name} (Category: {room.Category}, Area: {room.Area})")
            
    #     # Print door details
    #     print("\nDoors:")
    #     for door in floor.Construction.Door:
    #         print(f"- {door.Name} (Type: {door.DoorType})")