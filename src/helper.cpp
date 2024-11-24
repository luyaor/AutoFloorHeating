//
// Created by Wang Tai on 2024/10/20.
//

#include "helper.hpp"
#include <json/json.h>
#include "pipe_layout_generator.hpp"
#include "parsers/ar_design_parser.hpp"
#include "parsers/input_data_parser.hpp"
#include <opencv2/opencv.hpp>


namespace iad {


// Main parsing function that combines both
CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson) {
    CombinedData combinedData;
    
    try {
        combinedData.arDesign = parsers::ARDesignParser::parse(arDesignJson);
        printARDesign(combinedData.arDesign, std::cout);
        combinedData.inputData = parsers::InputDataParser::parse(inputDataJson);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error parsing JSON: ") + e.what());
    }

    return combinedData;
}

HeatingDesign generatePipePlan(const CombinedData& combinedData){
    HeatingDesign heatingDesign;

    // Iterate through each floor
    for (const auto& floor : combinedData.arDesign.Floor) {
        HeatingCoil heatingCoil;
        heatingCoil.LevelName = floor.Name;
        heatingCoil.LevelNo = std::stoi(floor.Num);
        heatingCoil.LevelDesc = "Floor " + floor.Num;

        // Iterate through each house type
        for (const auto& houseType : floor.construction.houseTypes) {
            heatingCoil.HouseName = houseType.houseName;

            // Call the pipe layout generation function
            CollectorCoil collectorCoil = generatePipeLayout(houseType, combinedData.inputData.webData);

            // Add the generated CollectorCoil to HeatingCoil
            heatingCoil.CollectorCoils.push_back(collectorCoil);
        }

        // Add the generated HeatingCoil to HeatingDesign
        heatingDesign.HeatingCoils.push_back(heatingCoil);
    }

    return heatingDesign;
}

void iad::printARDesign(const ARDesign& design, std::ostream& out) {
    out << "ARDesign Summary:\n";
    for (const auto& floor : design.Floor) {
        out << "Floor: " << floor.Name << " (Num: " << floor.Num << ")\n";
        out << "  Height: " << floor.LevelHeight << "\n";
        out << "  HouseTypes: " << floor.construction.houseTypes.size() << "\n";
        out << "  Rooms: " << floor.construction.rooms.size() << "\n";
        out << "  JCWs: " << floor.construction.jcws.size() << "\n";
        out << "  Doors: " << floor.construction.door.size() << "\n";
    }
}

void drawARDesign(const ARDesign& design, const std::string& outputPath) {
    // 创建一个白色背景的图像
    cv::Mat image(2000, 2000, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 定义颜色
    const cv::Scalar ROOM_COLOR(200, 200, 200);    // 灰色填充房间
    const cv::Scalar WALL_COLOR(0, 0, 0);          // 黑色墙体
    const cv::Scalar DOOR_COLOR(0, 0, 255);        // 红色门
    const cv::Scalar JCW_COLOR(0, 255, 0);         // 绿色家具

    // 找到整体边界以进行缩放
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    // 遍历所有楼层找到边界
    for (const auto& floor : design.Floor) {
        for (const auto& room : floor.construction.rooms) {
            for (const auto& curve : room.Boundary) {
                minX = std::min({minX, curve.StartPoint.x, curve.EndPoint.x});
                minY = std::min({minY, curve.StartPoint.y, curve.EndPoint.y});
                maxX = std::max({maxX, curve.StartPoint.x, curve.EndPoint.x});
                maxY = std::max({maxY, curve.StartPoint.y, curve.EndPoint.y});
            }
        }
    }

    // 计算缩放因子
    double scale = std::min(1800.0 / (maxX - minX), 1800.0 / (maxY - minY));
    
    // 坐标转换函数
    auto transformPoint = [&](const Point& p) -> cv::Point {
        return cv::Point(
            static_cast<int>((p.x - minX) * scale) + 100,
            static_cast<int>((p.y - minY) * scale) + 100
        );
    };

    // 绘制每个楼层
    for (const auto& floor : design.Floor) {
        // 绘制房间
        for (const auto& room : floor.construction.rooms) {
            std::vector<cv::Point> roomPoints;
            for (const auto& curve : room.Boundary) {
                cv::Point start = transformPoint(curve.StartPoint);
                cv::Point end = transformPoint(curve.EndPoint);
                
                if (curve.CurveType == 0) { // 直线
                    cv::line(image, start, end, WALL_COLOR, 2);
                } else { // 弧线
                    cv::Point center = transformPoint(curve.Center);
                    double radius = cv::norm(center - start);
                    cv::ellipse(image, center, cv::Size(radius, radius),
                              0, curve.StartAngle, curve.EndAngle, WALL_COLOR, 2);
                }
                roomPoints.push_back(start);
            }
            
            // 填充房间
            if (!roomPoints.empty()) {
                std::vector<std::vector<cv::Point>> contours = {roomPoints};
                cv::fillPoly(image, contours, ROOM_COLOR);
            }
        }

        // 绘制门
        for (const auto& door : floor.construction.door) {
            cv::Point location = transformPoint(door.Location);
            cv::circle(image, location, 5, DOOR_COLOR, -1);
        }

        // 绘制家具
        for (const auto& jcw : floor.construction.jcws) {
            for (const auto& line : jcw.BoundaryLines) {
                cv::Point start = transformPoint(line.StartPoint);
                cv::Point end = transformPoint(line.EndPoint);
                cv::line(image, start, end, JCW_COLOR, 1);
            }
        }
    }

    // 保存图像
    cv::imwrite(outputPath, image);
}
}