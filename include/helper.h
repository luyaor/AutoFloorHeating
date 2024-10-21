#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <json/json.h>

// Structures for ARDesign.json
// 坐标点结构
struct Point {
    double x;
    double y;
    double z;
};

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



// Structures for inputData.json
// 边界结构
struct Border {
    Point StartPoint;
    Point EndPoint;
    int ColorIndex;
    int CurveType;
};

// 辅助采集器结构
struct AssistCollector {
    std::string Id;
    Point Loc;
    std::string LevelName;
    struct Boundary {
        std::vector<Border> Borders;
        double Offset;
    };
    std::vector<Boundary> Boundaries;
};

// 辅助数据结构
struct AssistData {
    std::vector<AssistCollector> AssistCollectors;
};

// 回路跨度结构
struct LoopSpan {
    std::string TypeName;
    double MinSpan;
    double MaxSpan;
    double Curvity;
};

// 障碍物跨度结构
struct ObstacleSpan {
    std::string ObsName;
    double MinSpan;
    double MaxSpan;
};

// 管道跨度结构
struct PipeSpanSet {
    std::string LevelDesc;
    std::string FuncName;
    std::vector<std::string> Directions;
    int ExterWalls;
    double PipeSpan;
};

// 弹性跨度结构
struct ElasticSpan {
    std::string FuncName;
    double PriorSpan;
    double MinSpan;
    double MaxSpan;
};

// 功能房间结构
struct FuncRoom {
    std::string FuncName;
    std::vector<std::string> RoomNames;
};

// Web 数据结构
struct WebData {
    int ImbalanceRatio;
    double JointPipeSpan;
    double DenseAreaWallSpan;
    double DenseAreaSpanLess;
    std::vector<LoopSpan> LoopSpanSet;
    std::vector<ObstacleSpan> ObsSpanSet;
    std::vector<ObstacleSpan> DeliverySpanSet;
    std::vector<PipeSpanSet> PipeSpanSet;
    std::vector<ElasticSpan> ElasticSpanSet;
    std::vector<FuncRoom> FuncRooms;
};

// 总的数据结构
struct InputData {
    AssistData assistData;
    WebData webData;
};


// Combined structure for both JSON files
struct CombinedData {
    ARDesign arDesign;
    InputData inputData;
};

// Function declarations
CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson);
std::vector<cv::Point> generatePipePlan(const CombinedData& combinedData);
std::string planToJson(const std::vector<cv::Point>& plan);

#endif // HELPER_H
