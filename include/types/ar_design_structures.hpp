#ifndef AR_DESIGN_STRUCTURES_H
#define AR_DESIGN_STRUCTURES_H

#include <vector>
#include <string>
#include "data_structures.hpp"
#include <map>

// Structures for ARDesign.json
// 尺寸结构
struct Size {
    double Height; // 高度
    double Width; // 宽度
    double Thickness; // 厚度
};


// JCW (家具等) 结构
struct JCW {
    std::string GuidName; // 全局唯一标识符
    int Type; // 类型编号
    std::string Name; // 名称
    Point CenterPoint; // 中心点
    std::vector<CurveInfo> BoundaryLines; // 边界线列表
};

// 房间结构
struct Room {
    std::string Guid;                // 全局唯一标识符
    std::string Name;               // 名称
    std::string NameType;           // 名称类型
    std::vector<std::string> DoorIds;      // 门ID列表
    std::vector<std::string> JCWGuidNames; // 家具等ID列表
    std::vector<std::string> WallNames; // 墙体id名称列表
    std::vector<CurveInfo> Boundary; // 边界线列表
    int BlCreateRoom;               // 是否真实房间
};

// 房型结构
struct HouseType {
    std::string houseName; // 户型名称 唯一id
    std::vector<std::string> RoomNames; // 包含的房间ID列表
    std::vector<Point> Boundary; // 户型边界
};


// 门结构
struct Door {
    std::string Guid; // 全局唯一标识符
    std::string FamilyName;    // 族名称
    std::string DoorType;             // 门类型
    std::string HostWall;      // 所属墙体
    Point Location;            // 位置
    Size Size;          // 尺寸
    Point FlipFaceNormal;      // 门开启方向
    Point FlipHandNormal;      // 门把手方向
};

// 建筑结构
struct Construction {
    std::vector<HouseType> houseTypes;     // 户型列表
    std::vector<Room> rooms;               // 房间列表
    std::vector<JCW> jcws;                 // 家具列表
    std::vector<Door> door;  // 门窗列表
};

// 楼层结构
struct Floor {
    std::string Name; // 楼层名称
    std::string Num; // 楼层号
    double LevelHeight; // 层高度
    Construction construction; // 楼层的建筑结构
};

// 建筑设计数据结构
struct ARDesign {
    std::vector<Floor> Floor; // 楼层列表
};
#endif // AR_DESIGN_STRUCTURES_H
