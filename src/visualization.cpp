#include "visualization.hpp"
#include <opencv2/freetype.hpp>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <iostream>

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

    // 为每个楼层绘制图像
    for (const auto& floor : design.Floor) {
        cv::Mat image(2000, 2000, CV_8UC3, cv::Scalar(255, 255, 255));
        
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

        // 计算缩放因子
        double scale = std::min(1800.0 / (maxX - minX), 1800.0 / (maxY - minY));
        
        // 坐标转换函数
        auto transformPoint = [&](const Point& p) -> cv::Point {
            return {
                static_cast<int>((p.x - minX) * scale) + 100,
                static_cast<int>((p.y - minY) * scale) + 100
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
                    cv::line(image, start, end, WALL_COLOR, 2);
                    walls.emplace_back(start, end);
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

            // 添加墙体尺寸
            for (const auto& wall : walls) {
                cv::Point start = wall.first;
                cv::Point end = wall.second;
                
                // 计算实际距离（以毫米为单位）
                double realDistance = cv::norm(cv::Point2d(
                    (end.x - start.x) / scale,
                    (end.y - start.y) / scale
                ));
                
                // 格式化距离文本（转换为米，保留两位小数）
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << (realDistance / 1000.0) << "m";
                std::string dimText = ss.str();

                // 计算文字位置（墙体中点上方或右侧）
                cv::Point textPos((start.x + end.x) / 2, (start.y + end.y) / 2);
                cv::Point dir(end.x - start.x, end.y - start.y);
                double length = cv::norm(dir);
                if (length > 0) {
                    // 标注线的偏移距离
                    constexpr int offset = 20;
                    cv::Point normal(-dir.y / length, dir.x / length);
                    
                    // 绘制尺寸线和文字
                    cv::Point dimStart = start + normal * offset;
                    cv::Point dimEnd = end + normal * offset;
                    cv::line(image, dimStart, dimEnd, DIM_COLOR, 1);
                    cv::line(image, start, dimStart, DIM_COLOR, 1);
                    cv::line(image, end, dimEnd, DIM_COLOR, 1);
                    
                    // 文字位置调整
                    cv::Point textOffset = normal * (offset + 10);
                    textPos += textOffset;
                    
                    // 获取文字大小以便居中放置
                    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                    double fontScale = 0.4;
                    int thickness = 1;
                    int baseline;
                    cv::Size textSize = cv::getTextSize(dimText, fontFace, 
                                                       fontScale, thickness, &baseline);
                    
                    // 调整文字位置使其居中
                    textPos.x -= textSize.width / 2;
                    textPos.y += textSize.height / 2;
                    
                    // 使用新的函数绘制尺寸标注
                    putTextZH(image, dimText, textPos, DIM_COLOR, 20, 1);
                }
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

        // 生成当前楼层的输出文件名
        std::filesystem::path floor_path = 
            path.parent_path() / 
            (name_without_ext + "_floor_" + floor.Num + extension);
        
        std::cout << "Saving floor image to: " << floor_path.string() << std::endl;
        
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
    }
}

} 