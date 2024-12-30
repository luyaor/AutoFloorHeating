#include <gtest/gtest.h>
#include "core/export/dwg_exporter.hpp"
#include "core/parsing/parser/heating_design_parser.hpp"
#include <filesystem>
#include <fstream>
#include <json/json.h>

class DwgExporterTest : public ::testing::Test {
protected:
    std::string getTestDataPath(const std::string& filename) {
        return std::filesystem::current_path().string() + "/../tests/test_data/" + filename;
    }

    HeatingDesign loadDesignFromJson(const std::string& filename) {
        std::string fullPath = getTestDataPath(filename);
        std::ifstream file(fullPath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + fullPath);
        }
        
        Json::Value root;
        Json::Reader reader;
        if (!reader.parse(file, root)) {
            throw std::runtime_error("Failed to parse JSON file: " + fullPath);
        }
        
        return iad::parsers::HeatingDesignParser::fromJson(root);
    }

    void SetUp() override {
        // 创建输出目录
        std::filesystem::create_directories("test_output");
    }

    void TearDown() override {
        // 清理生成的 DXF 文件
        for (const auto& file : {"test_output/basic.dxf", "test_output/complex.dxf", 
                                "test_output/empty.dxf", "test_output/curves.dxf"}) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        }
    }
};

TEST_F(DwgExporterTest, ExportBasicDesignTest) {
    DwgExporter exporter;
    std::string filename = "test_output/basic.dxf";
    
    try {
        // 加载基本设计测试用例
        HeatingDesign design = loadDesignFromJson("basic_heating_design.json");
        
        // 测试导出
        EXPECT_TRUE(exporter.exportToFile(filename, design));
        EXPECT_TRUE(std::filesystem::exists(filename));
    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
}

TEST_F(DwgExporterTest, ExportComplexDesignTest) {
    DwgExporter exporter;
    std::string filename = "test_output/complex.dxf";
    
    try {
        // 加载复杂设计测试用例
        HeatingDesign design = loadDesignFromJson("complex_heating_design.json");
        
        // 测试导出
        EXPECT_TRUE(exporter.exportToFile(filename, design));
        EXPECT_TRUE(std::filesystem::exists(filename));
    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
}

TEST_F(DwgExporterTest, ExportEmptyDesignTest) {
    DwgExporter exporter;
    std::string filename = "test_output/empty.dxf";
    
    try {
        // 加载空设计测试用例
        HeatingDesign design = loadDesignFromJson("empty_heating_design.json");
        
        // 测试导出
        EXPECT_TRUE(exporter.exportToFile(filename, design));
        EXPECT_TRUE(std::filesystem::exists(filename));
    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
}

TEST_F(DwgExporterTest, ExportToInvalidPathTest) {
    DwgExporter exporter;
    std::string filename = "/invalid/path/test.dxf";
    
    try {
        HeatingDesign design = loadDesignFromJson("basic_heating_design.json");
        // 测试导出到无效路径
        EXPECT_FALSE(exporter.exportToFile(filename, design));
    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
}

// 测试不同类型的曲线
TEST_F(DwgExporterTest, ExportCurveTypesTest) {
    DwgExporter exporter;
    std::string filename = "test_output/curves.dxf";
    
    try {
        // 加载包含不同类型曲线的测试用例
        HeatingDesign design = loadDesignFromJson("curves_heating_design.json");
        
        // 测试导出
        EXPECT_TRUE(exporter.exportToFile(filename, design));
        EXPECT_TRUE(std::filesystem::exists(filename));
    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
} 