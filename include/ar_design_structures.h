#ifndef AR_DESIGN_STRUCTURES_H
#define AR_DESIGN_STRUCTURES_H

#include <vector>
#include <string>
#include "data_structures.h"

// Structures for ARDesign.json
// 尺寸结构
struct Size {
    double Height;
    double Width;
    double Thickness;
};

// 门结构
struct Door {
    std::string Guid;
    std::string FamilyName;
    std::string Name;
    int DoorType;  // 门类型
    std::string HostWall;  // 所属墙体的ID
    Point Location;  // 位置
    Size DoorSize;  // 尺寸
    Point FlipFaceNormal;  // 门开启方向
    Point FlipHandNormal;  // 把手方向
};

// JCW (家具等) 结构
struct JCW {
    std::string Guid;
    std::string Name;
    std::string Type;
    Point CenterPoint;
    std::vector<Point> BoundaryLines;  // 边界线
};

// 房间结构
struct Room {
    std::string Guid;
    std::string Name;
    std::string NameType;
    std::vector<std::string> DoorIds;  // 门ID
    std::vector<std::string> JCWGuidNames;  // 家具等ID
    std::vector<std::string> WallNames;  // 墙体ID
    std::vector<Point> Boundary;  // 边界
    bool IsRecreationalRoom;  // 是否真实房间
};

// 房型结构
struct HouseType {
    std::string houseName;
    std::vector<std::string> RoomNames;  // 包含的房间ID
    std::vector<Point> Boundary;  // 户型边界
};

// Construction（结构）结构
struct Construction {
    std::vector<HouseType> houseTypes;
    std::vector<Room> rooms;
    std::vector<JCW> jcws;
    std::vector<Door> doors;
};

// 楼层结构
struct Floor {
    std::string Name;  // 楼层名称
    std::string Num;  // 楼层号
    double LevelHeight;  // 楼层高度
    Construction construction;  // 楼层的建筑结构
};

// 建筑设计数据结构
struct ARDesign {
    std::vector<Floor> Floors;
};

#endif // AR_DESIGN_STRUCTURES_H