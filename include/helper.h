#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <json/json.h>

struct JPoint {
    double x, y, z;
};

struct JLine {
    JPoint startPoint, endPoint;
};

struct Room {
    std::vector<cv::Point> outline;
    std::vector<cv::Point> obstacles;
};

struct PlanningParams {
    double pipeWidth;
    bool isSnakePattern;
    double minDistance;
};

void drawLines(const std::vector<JLine>& lines, cv::Mat& image);
std::vector<JLine> parseDataFromJson(const std::string& jsonData);
Room parseRoomFromJson(const std::string& jsonData);
PlanningParams parsePlanningParams(const std::string& jsonData);
std::vector<cv::Point> generatePipePlan(const Room& room, const PlanningParams& params);
std::string planToJson(const std::vector<cv::Point>& plan);

#endif // HELPER_H
