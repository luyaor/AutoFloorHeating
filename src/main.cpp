#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <json/json.h> // Assuming JSON input is used.
#include "helper.h"

int main() {
    // 从标准输入读取 JSON 数据
    std::string inputJson;
    std::string line;
    while (std::getline(std::cin, line)) {
        inputJson += line + "\n";
    }
    
    // 解析房间数据和规划参数
    Room room = parseRoomFromJson(inputJson);
    PlanningParams params = parsePlanningParams(inputJson);
    
    // 生成管线规划
    std::vector<cv::Point> pipePlan = generatePipePlan(room, params);
    
    // 将结果转换为JSON并输出
    std::string outputJson = planToJson(pipePlan);
    std::cout << outputJson << std::endl;
    
    return 0;
}
