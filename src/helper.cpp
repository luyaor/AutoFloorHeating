//
// Created by Wang Tai on 2024/10/20.
//

#include "../include/helper.h"

void drawLines(const std::vector<JLine>& lines, cv::Mat& image) {
    for (const auto& line : lines) {
        cv::Point p1(line.startPoint.x, line.startPoint.y);
        cv::Point p2(line.endPoint.x, line.endPoint.y);
        cv::line(image, p1, p2, cv::Scalar(255, 0, 0), 2); // Blue lines
    }
}

std::vector<JLine> parseDataFromJson(const std::string& jsonData) {
    std::vector<JLine> lines;
    Json::Value root;
    Json::Reader reader;
    if (reader.parse(jsonData, root)) {
        const Json::Value jsonLines = root["lines"];
        for (const auto& jsonLine : jsonLines) {
            JLine line;
            line.startPoint.x = jsonLine["StartPoint"]["X"].asDouble();
            line.startPoint.y = jsonLine["StartPoint"]["Y"].asDouble();
            line.startPoint.z = jsonLine["StartPoint"]["Z"].asDouble();
            line.endPoint.x = jsonLine["EndPoint"]["X"].asDouble();
            line.endPoint.y = jsonLine["EndPoint"]["Y"].asDouble();
            line.endPoint.z = jsonLine["EndPoint"]["Z"].asDouble();
            lines.push_back(line);
        }
    }
    return lines;
}

Room parseRoomFromJson(const std::string& jsonData) {
    // 实现从JSON解析房间数据的逻辑
    Room room;
    // ... 填充 room 结构
    return room;
}

PlanningParams parsePlanningParams(const std::string& jsonData) {
    // 实现从JSON解析规划参数的逻辑
    PlanningParams params;
    // ... 填充 params 结构
    return params;
}

std::vector<cv::Point> generatePipePlan(const Room& room, const PlanningParams& params) {
    std::vector<cv::Point> pipePlan;
    // 实现管线规划算法
    // ... 生成 pipePlan
    return pipePlan;
}

std::string planToJson(const std::vector<cv::Point>& plan) {
    // 实现将规划结果转换为JSON的逻辑
    Json::Value root;
    // ... 填充 root
    Json::FastWriter writer;
    return writer.write(root);
}
