#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <json/json.h>

// Structures for ARDesign.json
struct Point3D {
    double x, y, z;
};

struct Grid {
    std::string GridTextNote;
    Point3D StartPoint;
    Point3D EndPoint;
    int EP1Visible;
    int EP2Visible;
    int IsSub;
};

struct Construction {
    std::vector<Grid> Grid;
};

struct Floor {
    std::string Name;
    std::string Num;
    std::string AllFloor;
    double LevelHeight;
    double LevelElevation;
    Construction Construction;
};

struct ARDesign {
    std::vector<Floor> Floor;
};

// Structures for inputData.json
struct LoopSpan {
    std::string TypeName;
    int MinSpan;
    int MaxSpan;
    int Curvity;
};

struct ObsSpan {
    std::string ObsName;
    int MinSpan;
    int MaxSpan;
};

struct PipeSpan {
    std::string LevelDesc;
    std::string FuncName;
    std::vector<std::string> Directions;
    int ExterWalls;
    double PipeSpan;
};

struct ElasticSpan {
    std::string FuncName;
    double PriorSpan;
    double MinSpan;
    double MaxSpan;
};

struct FuncRoom {
    std::string FuncName;
    std::vector<std::string> RoomNames;
};

struct WebData {
    int ImbalanceRatio;
    int JointPipeSpan;
    int DenseAreaWallSpan;
    int DenseAreaSpanLess;
    std::vector<LoopSpan> LoopSpanSet;
    std::vector<ObsSpan> ObsSpanSet;
    std::vector<ObsSpan> DeliverySpanSet;
    std::vector<PipeSpan> PipeSpanSet;
    std::vector<ElasticSpan> ElasticSpanSet;
    std::vector<FuncRoom> FuncRooms;
};

struct InputData {
    WebData WebData;
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
