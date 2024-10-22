#ifndef INPUT_DATA_STRUCTURES_H
#define INPUT_DATA_STRUCTURES_H

#include "data_structures.hpp"

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


#endif // INPUT_DATA_STRUCTURES_H
