#include <gtest/gtest.h>
#include "core/export/dwg_exporter.hpp"
#include <filesystem>

class DwgExporterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试数据
        design.HeatingCoils.push_back(createSampleHeatingCoil());
    }

    HeatingCoil createSampleHeatingCoil() {
        HeatingCoil coil;
        coil.LevelName = "1F";
        coil.LevelNo = 1;
        coil.LevelDesc = "First Floor";
        coil.HouseName = "A101";

        // 添加伸缩缝
        CurveInfo expansion;
        expansion.StartPoint.x = 0.0;
        expansion.StartPoint.y = 0.0;
        expansion.StartPoint.z = 0.0;
        coil.Expansions.push_back(expansion);

        expansion.StartPoint.x = 100.0;
        expansion.StartPoint.y = 0.0;
        coil.Expansions.push_back(expansion);

        // 添加分集水器回路
        CollectorCoil collector;
        collector.CollectorName = "C1";
        collector.Loops = 1;

        // 添加盘管回路
        CoilLoop loop;
        loop.Length = 100.0;
        loop.Curvity = 20;

        // 添加回路路径点
        CurveInfo point;
        point.StartPoint.x = 0.0;
        point.StartPoint.y = 0.0;
        point.StartPoint.z = 0.0;
        loop.Path.push_back(point);

        point.StartPoint.x = 50.0;
        point.StartPoint.y = 0.0;
        loop.Path.push_back(point);

        point.StartPoint.x = 50.0;
        point.StartPoint.y = 50.0;
        loop.Path.push_back(point);

        point.StartPoint.x = 0.0;
        point.StartPoint.y = 50.0;
        loop.Path.push_back(point);

        collector.CoilLoops.push_back(loop);
        coil.CollectorCoils.push_back(collector);

        return coil;
    }

    HeatingDesign design;
};

TEST_F(DwgExporterTest, ExportToFileTest) {
    DwgExporter exporter;
    std::string filename = "test_output.dwg";

    // 测试导出
    EXPECT_TRUE(exporter.exportToFile(filename, design));

    // 验证文件是否创建
    EXPECT_TRUE(std::filesystem::exists(filename));

    // 清理测试文件
    if (std::filesystem::exists(filename)) {
        std::filesystem::remove(filename);
    }
} 