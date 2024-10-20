#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <json/json.h> // Assuming JSON input is used.
#include <fstream>
#include "helper.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <ARDesign.json> <inputData.json>" << std::endl;
        return 1;
    }

    // 读取 ARDesign.json 文件
    std::ifstream arDesignFile(argv[1]);
    if (!arDesignFile.is_open()) {
        std::cerr << "Error: Unable to open ARDesign.json file." << std::endl;
        return 1;
    }
    std::string arDesignJson((std::istreambuf_iterator<char>(arDesignFile)),
                              std::istreambuf_iterator<char>());

    // 读取 inputData.json 文件
    std::ifstream inputDataFile(argv[2]);
    if (!inputDataFile.is_open()) {
        std::cerr << "Error: Unable to open inputData.json file." << std::endl;
        return 1;
    }
    std::string inputDataJson((std::istreambuf_iterator<char>(inputDataFile)),
                               std::istreambuf_iterator<char>());

    // 解析输入数据
    CombinedData combinedData = parseJsonData(arDesignJson, inputDataJson);
    
    // 生成管线规划
    std::vector<cv::Point> pipePlan = generatePipePlan(combinedData);
    
    // 将结果转换为JSON并输出
    std::string outputJson = planToJson(pipePlan);
    std::cout << outputJson << std::endl;
    
    return 0;
}
