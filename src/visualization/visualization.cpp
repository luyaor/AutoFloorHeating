#include "visualization/visualization.hpp"
#include <opencv2/freetype.hpp>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace iad {

void printARDesign(const ARDesign& design, std::ostream& out) {
    out << "ARDesign Summary:\n";
    for (const auto&[Name, Num, LevelHeight, construction] : design.Floor) {
        out << "Floor: " << Name << " (Num: " << Num << ")\n";
        out << "  Height: " << LevelHeight << "\n";
        out << "  HouseTypes: " << construction.houseTypes.size() << "\n";
        out << "  Rooms: " << construction.rooms.size() << "\n";
        out << "  JCWs: " << construction.jcws.size() << "\n";
        out << "  Doors: " << construction.door.size() << "\n";
    }
}

// 添加新的辅助函数用于绘制中文文本
void putTextZH(cv::Mat& dst, const std::string& str, const cv::Point org,
               const cv::Scalar& color, const int fontSize,
               const int thickness = 1, const std::string& fontPath = "") {
    // 尝试多个字体路径
    std::vector<std::string> fontPaths;
    
    #ifdef _WIN32
        fontPaths = {
            "C:/Windows/Fonts/simhei.ttf",     // 黑体
            "C:/Windows/Fonts/msyh.ttc",       // 微软雅黑
            "C:/Windows/Fonts/simsun.ttc"      // 宋体
        };
    #elif __APPLE__
        fontPaths = {
            "/System/Library/Fonts/STHeiti Light.ttc",    // 把已知可用的字体放在首位
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc"          // 把可能有问题的字体放在后面
        };
    #else
        fontPaths = {
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        };
    #endif

    // 如果提供了自定义字体路径，将其添加到列表开头
    if (!fontPath.empty()) {
        fontPaths.insert(fontPaths.begin(), fontPath);
    }

    // 创建FreeType字体
    const cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    bool fontLoaded = false;

    // 尝试加载每个字体，直到成功
    for (const auto& path : fontPaths) {
        try {
            ft2->loadFontData(path, 0);
            fontLoaded = true;
            // 只在调试模式下输出成功信息
            #ifdef DEBUG
            std::cout << "Successfully loaded font: " << path << std::endl;
            #endif
            break;
        } catch ([[maybe_unused]] const cv::Exception& e) {
            // 只在调试模式下输出详细错误信息
            #ifdef DEBUG
            std::cerr << "Failed to load font " << path << ": " << e.what() << std::endl;
            #endif
        }
    }

    if (!fontLoaded) {
        std::cerr << "Warning: Failed to load any Chinese font, falling back to default font" << std::endl;
        putText(dst, str, org, cv::FONT_HERSHEY_SIMPLEX, fontSize/30.0,
                    color, thickness, cv::LINE_AA);
        return;
    }

    try {
        ft2->putText(dst, str, org, fontSize, color, thickness, cv::LINE_AA, true);
    } catch (const cv::Exception& e) {
        std::cerr << "Error rendering text: " << e.what() << std::endl;
        cv::putText(dst, str, org, cv::FONT_HERSHEY_SIMPLEX, fontSize/30.0, 
                    color, thickness, cv::LINE_AA);
    }
}

void drawARDesign(const ARDesign& design, const std::string& outputPath) {
    std::cout << "Input outputPath: " << outputPath << std::endl;

    // 使用 filesystem 处理路径
    std::filesystem::path path(outputPath);
    std::string directory = path.parent_path().string();
    std::string name_without_ext = path.stem().string();
    std::string extension = path.extension().string();
    
    std::cout << "Parsed path components:" << std::endl
              << "- Directory: '" << directory << "'" << std::endl
              << "- File name (no ext): '" << name_without_ext << "'" << std::endl
              << "- Extension: '" << extension << "'" << std::endl;

    // 如果目录不为空，确保它存在
    if (!directory.empty()) {
        try {
            std::cout << "Attempting to create directory: " << directory << std::endl;
            bool created = std::filesystem::create_directories(directory);
            std::cout << "Directory creation " 
                      << (created ? "successful" : "not needed (already exists)") 
                      << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to create directory: " << e.what() << std::endl;
            throw;
        }
    } else {
        std::cout << "No directory to create (empty path)" << std::endl;
    }

    // 在drawARDesign函数开头添加字母生成器
    auto generateLabel = [](int index) -> std::string {
        return std::to_string(index + 1);  // 从1开始编号
    };

    // 在drawARDesign函数开头添加辅助函数，用于检查坐标是否重叠
    auto isPointNear = [](const Point& p1, const Point& p2, double tolerance = 1.0) -> bool {
        return std::abs(p1.x - p2.x) < tolerance && std::abs(p1.y - p2.y) < tolerance;
    };

    // 在绘制房间之前，添加坐标映射表的数据结构
    struct CoordInfo {
        std::string label;
        Point originalPoint;
        cv::Point transformedPoint;
    };
    std::vector<CoordInfo> coordMapping;

    // 添加辅助函数，用于查找已存在的坐标点
    auto findExistingPoint = [&coordMapping, &isPointNear](const Point& point) -> int {
        for (size_t i = 0; i < coordMapping.size(); ++i) {
            if (isPointNear(coordMapping[i].originalPoint, point)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    // 为每个楼层绘制图像
    for (const auto& floor : design.Floor) {
        cv::Mat image(4000, 4000, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // 定义颜色
        const cv::Scalar ROOM_COLOR(200, 200, 200);    // 灰色填充房间
        const cv::Scalar WALL_COLOR(0, 0, 0);          // 黑色墙体
        const cv::Scalar DOOR_COLOR(0, 0, 255);        // 红色门
        const cv::Scalar JCW_COLOR(0, 255, 0);         // 绿色家具
        const cv::Scalar TEXT_COLOR(0, 0, 0);          // 黑色文字
        const cv::Scalar DIM_COLOR(100, 100, 100);     // 灰色尺寸线

        // 找到当前楼层的边界
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double maxY = std::numeric_limits<double>::lowest();

        // 遍历当前楼层的所有房间找到边界
        for (const auto& room : floor.construction.rooms) {
            for (const auto& curve : room.Boundary) {
                minX = std::min({minX, curve.StartPoint.x, curve.EndPoint.x});
                minY = std::min({minY, curve.StartPoint.y, curve.EndPoint.y});
                maxX = std::max({maxX, curve.StartPoint.x, curve.EndPoint.x});
                maxY = std::max({maxY, curve.StartPoint.y, curve.EndPoint.y});
            }
        }

        // 计算缩放因子，留出右侧空间给坐标表
        double scale = std::min(2800.0 / (maxX - minX), 3800.0 / (maxY - minY));
        
        // 坐标转换函数，向左偏移以留出右侧空间
        auto transformPoint = [&](const Point& p) -> cv::Point {
            return {
                static_cast<int>((p.x - minX) * scale) + 100,  // 左边距改为100
                static_cast<int>((p.y - minY) * scale) + 100   // 上边距保持100
            };
        };

        // 绘制房间
        for (const auto& room : floor.construction.rooms) {
            std::vector<cv::Point> roomPoints;
            std::vector<std::pair<cv::Point, cv::Point>> walls;
            
            for (const auto& curve : room.Boundary) {
                cv::Point start = transformPoint(curve.StartPoint);
                cv::Point end = transformPoint(curve.EndPoint);
                
                if (curve.CurveType == 0) { // 直线
                    cv::line(image, start, end, WALL_COLOR, 3);
                    walls.emplace_back(start, end);
                    
                    // 检查起点是否已存在
                    int startIndex = findExistingPoint(curve.StartPoint);
                    std::string startLabel;
                    if (startIndex == -1) {
                        // 点不存在，添加新点
                        startLabel = generateLabel(coordMapping.size());
                        coordMapping.push_back({startLabel, curve.StartPoint, start});
                    } else {
                        // 点已存在，使用已有标签
                        startLabel = coordMapping[startIndex].label;
                    }
                    
                    // 检查终点是否已存在
                    int endIndex = findExistingPoint(curve.EndPoint);
                    std::string endLabel;
                    if (endIndex == -1) {
                        // 点不存在，添加新点
                        endLabel = generateLabel(coordMapping.size());
                        coordMapping.push_back({endLabel, curve.EndPoint, end});
                    } else {
                        // 点已存在，使用已有标签
                        endLabel = coordMapping[endIndex].label;
                    }
                    
                    // 绘制标签
                    putTextZH(image, startLabel, start + cv::Point(-20, -20), TEXT_COLOR, 32, 2);
                    putTextZH(image, endLabel, end + cv::Point(-20, -20), TEXT_COLOR, 32, 2);
                } else { // 弧线
                    cv::Point center = transformPoint(curve.Center);
                    double radius = cv::norm(center - start);
                    cv::ellipse(image, center, cv::Size(radius, radius),
                              0, curve.StartAngle, curve.EndAngle, WALL_COLOR, 3);
                }
                roomPoints.push_back(start);
            }
            
            // 填充房间
            if (!roomPoints.empty()) {
                std::vector<std::vector<cv::Point>> contours = {roomPoints};
                cv::fillPoly(image, contours, ROOM_COLOR);
            }

            // 添加房间名称
            if (!roomPoints.empty()) {
                // 计算房间中心点
                cv::Point center(0, 0);
                for (const auto& p : roomPoints) {
                    center.x += p.x;
                    center.y += p.y;
                }
                center.x /= roomPoints.size();
                center.y /= roomPoints.size();

                // 使用新的函数绘制房间名称
                putTextZH(image, room.Name, center, TEXT_COLOR, 24, 1);
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

        // 添加楼层信息
        std::string floorInfo = "楼层 " + floor.Num + " (" + floor.Name + ")";
        putTextZH(image, floorInfo, cv::Point(50, 50), TEXT_COLOR, 32, 2);

        // 在绘制完所有内容后，添加坐标映射表
        const int TABLE_X = 3000;  // 第一列起始X坐标
        const int TABLE_X2 = 3500; // 第二列起始X坐标
        const int TABLE_Y = 200;   // 表格起始Y坐标
        const int LINE_HEIGHT = 40; // 行高

        // 添加表头
        putTextZH(image, "点位对照表", cv::Point(TABLE_X, TABLE_Y), TEXT_COLOR, 32, 2);
        putTextZH(image, "标记    坐标", cv::Point(TABLE_X, TABLE_Y + LINE_HEIGHT), TEXT_COLOR, 24, 1);
        putTextZH(image, "标记    坐标", cv::Point(TABLE_X2, TABLE_Y + LINE_HEIGHT), TEXT_COLOR, 24, 1);

        // 创建一个有序集合来存储唯一的坐标信息
        std::set<std::string> processedLabels;
        std::vector<std::pair<std::string, std::string>> uniqueCoords; // 存储标签和坐标文本

        // 首先收集所有唯一的坐标
        for (const auto& coord : coordMapping) {
            if (processedLabels.find(coord.label) != processedLabels.end()) {
                continue;
            }
            processedLabels.insert(coord.label);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(0) 
               << "(" << coord.originalPoint.x << "," << coord.originalPoint.y << ")";
            
            uniqueCoords.push_back({coord.label, ss.str()});
        }

        // 计算每列应显示的行数
        int totalCoords = uniqueCoords.size();
        int rowsPerColumn = (totalCoords + 1) / 2; // 向上取整

        // 绘制坐标列表
        for (int i = 0; i < totalCoords; ++i) {
            const auto& coord = uniqueCoords[i];
            bool isSecondColumn = i >= rowsPerColumn;
            
            int x = isSecondColumn ? TABLE_X2 : TABLE_X;
            int row = isSecondColumn ? i - rowsPerColumn : i;
            cv::Point textPos(x, TABLE_Y + LINE_HEIGHT * (row + 2));
            
            putTextZH(image, coord.first + "      " + coord.second, 
                     textPos, TEXT_COLOR, 24, 1);
        }

        // 在绘制完所有内容后，添加坐标系
        const int AXIS_LENGTH = 200;  // 坐标轴长度
        const int ARROW_LENGTH = 20;  // 箭头长度
        const cv::Point ORIGIN(200, image.rows - 200);  // 原点位置
        
        // 绘制X轴
        cv::line(image, ORIGIN, 
                 cv::Point(ORIGIN.x + AXIS_LENGTH, ORIGIN.y), 
                 TEXT_COLOR, 2);
        // X轴箭头
        cv::line(image, 
                 cv::Point(ORIGIN.x + AXIS_LENGTH, ORIGIN.y),
                 cv::Point(ORIGIN.x + AXIS_LENGTH - ARROW_LENGTH, ORIGIN.y - ARROW_LENGTH/2),
                 TEXT_COLOR, 2);
        cv::line(image, 
                 cv::Point(ORIGIN.x + AXIS_LENGTH, ORIGIN.y),
                 cv::Point(ORIGIN.x + AXIS_LENGTH - ARROW_LENGTH, ORIGIN.y + ARROW_LENGTH/2),
                 TEXT_COLOR, 2);
        putTextZH(image, "X", 
                  cv::Point(ORIGIN.x + AXIS_LENGTH + 10, ORIGIN.y + 20), 
                  TEXT_COLOR, 24, 2);
        
        // 绘制Y轴
        cv::line(image, ORIGIN, 
                 cv::Point(ORIGIN.x, ORIGIN.y - AXIS_LENGTH), 
                 TEXT_COLOR, 2);
        // Y轴箭头
        cv::line(image, 
                 cv::Point(ORIGIN.x, ORIGIN.y - AXIS_LENGTH),
                 cv::Point(ORIGIN.x - ARROW_LENGTH/2, ORIGIN.y - AXIS_LENGTH + ARROW_LENGTH),
                 TEXT_COLOR, 2);
        cv::line(image, 
                 cv::Point(ORIGIN.x, ORIGIN.y - AXIS_LENGTH),
                 cv::Point(ORIGIN.x + ARROW_LENGTH/2, ORIGIN.y - AXIS_LENGTH + ARROW_LENGTH),
                 TEXT_COLOR, 2);
        putTextZH(image, "Y", 
                  cv::Point(ORIGIN.x - 20, ORIGIN.y - AXIS_LENGTH - 10), 
                  TEXT_COLOR, 24, 2);
        
        // 标注原点
        putTextZH(image, "O(0,0)", 
                  cv::Point(ORIGIN.x - 60, ORIGIN.y + 30), 
                  TEXT_COLOR, 24, 2);

        // 生成当前楼层的输出文件名
        std::filesystem::path floor_path = 
            path.parent_path() / 
            (name_without_ext + "_floor_" + floor.Num + extension);
            
        // 生成对应的坐标文件名（将扩展名改为.txt）
        std::filesystem::path coord_path = 
            path.parent_path() / 
            (name_without_ext + "_floor_" + floor.Num + "_coords.txt");
        
        std::cout << "Saving floor image to: " << floor_path.string() << std::endl;
        std::cout << "Saving coordinates to: " << coord_path.string() << std::endl;
        
        // 保存图像
        try {
            bool success = cv::imwrite(floor_path.string(), image);
            if (!success) {
                std::cerr << "Failed to save image: " << floor_path.string() << std::endl;
            } else {
                std::cout << "Successfully saved image: " << floor_path.string() << std::endl;
            }
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV error while saving image: " << e.what() << std::endl;
            throw;
        }

        // 保存坐标信息到txt文件
        try {
            std::ofstream coordFile(coord_path);
            if (!coordFile.is_open()) {
                std::cerr << "Failed to open coordinate file: " << coord_path.string() << std::endl;
                throw std::runtime_error("Failed to open coordinate file");
            }

            // 写入表头
            coordFile << "点位对照表\n";
            coordFile << "标记    坐标\n";
            coordFile << "-------------------\n";
            coordFile << "\n墙体坐标:\n";

            // 写入墙体坐标信息（保持两列格式）
            std::vector<std::pair<std::string, std::string>> uniqueCoords;
            std::set<std::string> processedLabels;

            // 收集唯一的墙体坐标信息
            for (const auto& coord : coordMapping) {
                if (processedLabels.find(coord.label) != processedLabels.end()) {
                    continue;
                }
                processedLabels.insert(coord.label);
                
                std::stringstream ss;
                ss << std::fixed << std::setprecision(0) 
                   << "(" << coord.originalPoint.x << "," << coord.originalPoint.y << ")";
                
                uniqueCoords.push_back({coord.label, ss.str()});
            }

            // 写入墙体坐标
            int totalCoords = uniqueCoords.size();
            int rowsPerColumn = (totalCoords + 1) / 2;
            for (int row = 0; row < rowsPerColumn; ++row) {
                coordFile << std::left << std::setw(20) 
                         << (uniqueCoords[row].first + "  " + uniqueCoords[row].second);
                
                if (row + rowsPerColumn < totalCoords) {
                    coordFile << std::left 
                             << (uniqueCoords[row + rowsPerColumn].first + "  " 
                                + uniqueCoords[row + rowsPerColumn].second);
                }
                coordFile << "\n";
            }

            // 写入门的信息
            coordFile << "\n门的信息:\n";
            coordFile << "编号    起点坐标         终点坐标\n";
            coordFile << "----------------------------------------\n";
            
            int doorIndex = 1;
            for (const auto& door : floor.construction.door) {
                // 计算门的起点和终点坐标
                Point doorStart = door.Location;
                Point doorEnd = {
                    doorStart.x + door.Size.Width * door.FlipFaceNormal.x,
                    doorStart.y + door.Size.Width * door.FlipFaceNormal.y
                };

                std::stringstream ss;
                ss << std::fixed << std::setprecision(0) 
                   << "D" << doorIndex << "      "
                   << "(" << doorStart.x << "," << doorStart.y << ")"
                   << "      "
                   << "(" << doorEnd.x << "," << doorEnd.y << ")";
                
                coordFile << ss.str() << "\n";
                doorIndex++;
            }

            coordFile.close();
            std::cout << "Successfully saved coordinates to: " << coord_path.string() << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error saving coordinates: " << e.what() << std::endl;
            throw;
        }
    }
}

} 