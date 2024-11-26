#include <iostream>
#include <string>
#include <fstream>
#include "core/parsing/output_parser.hpp"
#include "core/parsing/input_parser.hpp"
#include "visualization/visualization.hpp"
#include "core/parsing/parser/heating_design_parser.hpp"

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
    iad::CombinedData combinedData = iad::input_parser::parseJsonData(arDesignJson, inputDataJson);
    iad::printARDesign(combinedData.arDesign, std::cout);
    
    // 生成管线规划
    HeatingDesign pipePlan = iad::output_parser::generatePipePlan(combinedData);
    
    // 将结果转换为JSON并输出
    std::string outputJson = iad::parsers::HeatingDesignParser::toJson(pipePlan);
    std::cout << outputJson << std::endl;
    
    return 0;
}
