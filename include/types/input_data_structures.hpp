#ifndef INPUT_DATA_STRUCTURES_H
#define INPUT_DATA_STRUCTURES_H

#include <string>
#include <vector>
#include "data_structures.hpp"
#include "ar_design_structures.hpp"

// Collector structure
struct AssistCollector {
    Point Location;
    std::vector<CurveInfo> Borders;
};

// Construction structure
struct AssistConstruction {
    std::vector<AssistCollector> AssistCollector;
};

// Floor structure
struct AssistFloor {
    std::string Name;
    std::string Num;
    Point BasePoint;
    double LevelHeight;
    double LevelElevation;
    AssistConstruction Construction;
};

// AssistData structure
struct AssistData {
    std::vector<AssistFloor> Floor;
};

// Loop span structure
struct LoopSpan {
    std::string TypeName;
    double MinSpan;
    double MaxSpan;
    double Curvity;
};

// Obstacle span structure
struct ObstacleSpan {
    std::string ObsName;
    double MinSpan;
    double MaxSpan;
};

// Pipe span structure
struct PipeSpanSet {
    std::string LevelDesc;
    std::string FuncName;
    std::vector<std::string> Directions;
    int ExterWalls;
    double PipeSpan;
};

// Elastic span structure
struct ElasticSpan {
    std::string FuncName;
    double PriorSpan;
    double MinSpan;
    double MaxSpan;
};

// Function room structure
struct FuncRoom {
    std::string FuncName;
    std::vector<std::string> RoomNames;
};

// Web data structure
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

// Input data structure
struct InputData {
    AssistData AssistData;
    WebData WebData;
};

#endif // INPUT_DATA_STRUCTURES_H
