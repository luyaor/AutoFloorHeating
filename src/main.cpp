#include <iostream>
#include <string>
#include <fstream>
#include "core/parsing/output_parser.hpp"
#include "core/parsing/input_parser.hpp"
#include "visualization/visualization.hpp"
#include "core/parsing/parser/heating_design_parser.hpp"
#include "core/export/dwg_exporter.hpp"

void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <ARDesign.json> <inputData.json> [--export-dxf <output.dxf>]" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::string dxfOutputPath;
    if (argc == 5) {
        if (std::string(argv[3]) != "--export-dxf") {
            printUsage(argv[0]);
            return 1;
        }
        dxfOutputPath = argv[4];
    }

    try {
        // 读取 ARDesign.json 文件
        std::ifstream arDesignFile(argv[1]);
        if (!arDesignFile.is_open()) {
            std::cerr << "Error: Unable to open ARDesign.json file: " << argv[1] << std::endl;
            return 1;
        }
        std::string arDesignJson((std::istreambuf_iterator<char>(arDesignFile)),
                                std::istreambuf_iterator<char>());

        // 读取 inputData.json 文件
        std::ifstream inputDataFile(argv[2]);
        if (!inputDataFile.is_open()) {
            std::cerr << "Error: Unable to open inputData.json file: " << argv[2] << std::endl;
            return 1;
        }
        std::string inputDataJson((std::istreambuf_iterator<char>(inputDataFile)),
                                std::istreambuf_iterator<char>());

        // 解析输入数据
        std::cout << "Processing input files..." << std::endl;
        iad::CombinedData combinedData = iad::input_parser::parseJsonData(arDesignJson, inputDataJson);
        
        // 生成管线规划
        std::cout << "Generating pipe layout..." << std::endl;
        HeatingDesign pipePlan = iad::output_parser::generatePipePlan(combinedData);
        
        // 将结果转换为JSON并输出
        std::string outputJson = iad::parsers::HeatingDesignParser::toJson(pipePlan);
        std::cout << "Generated pipe layout:" << std::endl;
        std::cout << outputJson << std::endl;

        // 如果指定了导出DXF，则导出
        if (!dxfOutputPath.empty()) {
            std::cout << "Exporting to DXF file: " << dxfOutputPath << "..." << std::endl;
            DwgExporter exporter;
            if (!exporter.exportToFile(dxfOutputPath, pipePlan)) {
                std::cerr << "Error: Failed to export to DXF file" << std::endl;
                return 1;
            }
            std::cout << "Successfully exported to DXF file: " << dxfOutputPath << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
