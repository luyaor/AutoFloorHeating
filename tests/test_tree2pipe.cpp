// tests/test_pipe_layout_generator.cpp
#include "../include/tree2pipe.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <Eigen/Core>

using namespace tree2pipe;
using namespace std;
using Eigen::Vector2d;

TEST(Tree2PipeTest, Test1) {
    double w = 0.15;
    double dx = -0.999;
    Vector2d ovr_dir(dx, sqrt(1 - dx * dx));
    auto m1_rt = make_shared<NodeM1>(
        Vector2d(2, -1),
        CfgM1 { Vector2d::Zero(), w },
        vector<shared_ptr<M1>> { make_shared<RectM1>(
            Vector2d(2, 0),
            Vector2d(0, 1),
            CfgM1 { ovr_dir, w * 0.7 },
            vector<RectM1Son> {
                { 2, make_shared<NodeM1>(Vector2d(0.5, 2.5), CfgM1 { Vector2d::Zero(), w }) },
                { 2, make_shared<NodeM1>(Vector2d(1.0, 2.5), CfgM1 { Vector2d::Zero(), w }) },
                { 2, make_shared<NodeM1>(Vector2d(0.2, 2.5), CfgM1 { Vector2d::Zero(), w }) },
                { 3, make_shared<NodeM1>(Vector2d(3, 0.6), CfgM1 { Vector2d::Zero(), w }) },
                { 1, make_shared<NodeM1>(Vector2d(-2, 0.6), CfgM1 { Vector2d::Zero(), w }) },
                { 0,
                  make_shared<RectM1>(
                      Vector2d(0.5, -1.0),
                      Vector2d(-1.2, -3.0),
                      CfgM1 { std::nullopt, std::nullopt }
                  ) },
                { 2,
                  make_shared<RectM1>(
                      Vector2d(1.3, 2.5),
                      Vector2d(2.5, 3.6),
                      CfgM1 { Vector2d(5, 0.7).normalized(), w * 1.4 }
                  ) },

            }
        ) }
    );
    try {
        auto m0_rt = m1_rt->to_tree(CfgM1 { std::nullopt, std::nullopt });
        auto pts = tree_to_points(m0_rt, Vector2d::Zero());
        EXPECT_EQ(pts.size(), 90);
        plot_points_linked(pts);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}


TEST(Tree2PipeTest, Test2) {
    // 定义一个五边形
    std::vector<Vector2d> pentagon_vertices = {
        {0, 0}, {1, 0}, {1.5, 1}, {0.5, 1.5}, {-0.5, 1}
    };
    auto polygon_m1 = std::make_shared<PolygonM1>(pentagon_vertices);
    // 转换为树
    auto root_node = polygon_m1->to_tree(CfgM1{std::nullopt, 0.1});
    // 获取用于可视化的点
    auto points = tree_to_points(root_node);
    // 绘制
    plot_points_linked(points);
}

TEST(Tree2PipeTest, TestSharedPlotting) {
    // Test case 1: Simple triangle
    std::vector<Vector2d> triangle = {
        Vector2d(0, 0),
        Vector2d(1, 0),
        Vector2d(0.5, 1),
        Vector2d(0, 0)  // Close the shape
    };
    
    // Test case 2: Square
    std::vector<Vector2d> square = {
        Vector2d(0.2, 0.2),
        Vector2d(0.8, 0.2),
        Vector2d(0.8, 0.8),
        Vector2d(0.2, 0.8),
        Vector2d(0.2, 0.2)  // Close the shape
    };
    
    // Test case 3: Pentagon
    std::vector<Vector2d> pentagon = {
        Vector2d(0.5, 0),
        Vector2d(1, 0.4),
        Vector2d(0.8, 1),
        Vector2d(0.2, 1),
        Vector2d(0, 0.4),
        Vector2d(0.5, 0)  // Close the shape
    };

    // Plot all shapes and save to a single file
    plot_points_linked_shared(triangle, false, "b");  // Blue triangle
    plot_points_linked_shared(square, false, "r");    // Red square
    plot_points_linked_shared(pentagon, true, "g");   // Green pentagon and save plot

    // Verify that the file was created
    std::ifstream file("combined_plot_1.png");
    EXPECT_TRUE(file.good());
    file.close();
}

TEST(Tree2PipeTest, TestSharedPlottingWithM1) {
    // 创建临时目录
    std::string temp_dir = "test_output";
    std::filesystem::create_directories(temp_dir);
    
    double w = 0.15;
    
    // Create first structure
    auto m1_first = make_shared<NodeM1>(
        Vector2d(0, 0),
        CfgM1 { Vector2d::Zero(), w },
        vector<shared_ptr<M1>> { 
            make_shared<RectM1>(
                Vector2d(0, 0),
                Vector2d(1, 1),
                CfgM1 { Vector2d(1, 1).normalized(), w * 0.7 }
            )
        }
    );

    // Create second structure
    auto m1_second = make_shared<NodeM1>(
        Vector2d(2, 0),
        CfgM1 { Vector2d::Zero(), w },
        vector<shared_ptr<M1>> { 
            make_shared<RectM1>(
                Vector2d(2, 0),
                Vector2d(3, 1),
                CfgM1 { Vector2d(1, 1).normalized(), w * 0.7 }
            )
        }
    );

    try {
        // Convert both structures to trees and get points
        auto points1 = tree_to_points(m1_first->to_tree(CfgM1 { std::nullopt, std::nullopt }));
        auto points2 = tree_to_points(m1_second->to_tree(CfgM1 { std::nullopt, std::nullopt }));

        // Plot both structures and save to a single file
        plot_points_linked_shared(points1, false, "b");  // First structure in blue
        plot_points_linked_shared(points2, true, "r");   // Second structure in red and save plot

        // 检查文件是否存在
        std::string filename = temp_dir + "/combined_plot_1.png";
        std::ifstream file(filename);
        EXPECT_TRUE(file.good()) << "Failed to find file at: " << filename;
        file.close();

        EXPECT_GT(points1.size(), 0);
        EXPECT_GT(points2.size(), 0);
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST(Tree2PipeTest, TestMultipleShapesAtDifferentPositions) {
    // 创建输出目录
    std::string output_dir = "test_output";
    std::filesystem::create_directories(output_dir);
    
    // 创建几个不同位置的图形
    
    // 图形1：原点附近的正方形
    std::vector<Vector2d> square1 = {
        Vector2d(0, 0),
        Vector2d(1, 0),
        Vector2d(1, 1),
        Vector2d(0, 1),
        Vector2d(0, 0)  // 闭合图形
    };
    
    // 图形2：右上方的正方形
    std::vector<Vector2d> square2 = {
        Vector2d(5, 5),
        Vector2d(6, 5),
        Vector2d(6, 6),
        Vector2d(5, 6),
        Vector2d(5, 5)  // 闭合图形
    };
    
    // 图形3：左下方的三角形
    std::vector<Vector2d> triangle = {
        Vector2d(-2, -2),
        Vector2d(-1, -2),
        Vector2d(-1.5, -1),
        Vector2d(-2, -2)  // 闭合图形
    };
    
    // 图形4：右下方的矩形
    std::vector<Vector2d> rectangle = {
        Vector2d(3, -3),
        Vector2d(5, -3),
        Vector2d(5, -2),
        Vector2d(3, -2),
        Vector2d(3, -3)  // 闭合图形
    };
    
    // 依次绘制所有图形，使用不同的颜色
    plot_points_linked_shared(square1, false, "r");    // 红色
    plot_points_linked_shared(square2, false, "g");    // 绿色
    plot_points_linked_shared(triangle, false, "b");   // 蓝色
    plot_points_linked_shared(rectangle, true, "r");   // 红色，并保存图像
    
    // 验证最终图像是否生成
    std::string filename = output_dir + "/combined_plot_1.png";
    std::ifstream f(filename);
    EXPECT_TRUE(f.good()) << "Failed to find file at: " << filename;
    f.close();
}

TEST(Tree2PipeTest, TestFourCornersShapes) {
    // 创建输出目录
    std::string output_dir = "test_output";
    std::filesystem::create_directories(output_dir);
    
    // 左上角：三角形 (大小约1x1)
    std::vector<Vector2d> triangle = {
        Vector2d(-5, 4),
        Vector2d(-4, 4),
        Vector2d(-4.5, 5),
        Vector2d(-5, 4)  // 闭合图形
    };
    
    // 右上角：正方形 (大小1x1)
    std::vector<Vector2d> square = {
        Vector2d(4, 4),
        Vector2d(5, 4),
        Vector2d(5, 5),
        Vector2d(4, 5),
        Vector2d(4, 4)  // 闭合图形
    };
    
    // 左下角：五边形 (大小约1x1)
    std::vector<Vector2d> pentagon = {
        Vector2d(-5, -5),
        Vector2d(-4, -5),
        Vector2d(-4, -4),
        Vector2d(-4.5, -3.5),
        Vector2d(-5, -4),
        Vector2d(-5, -5)  // 闭合图形
    };
    
    // 右下角：矩形 (大小1x1)
    std::vector<Vector2d> rectangle = {
        Vector2d(4, -5),
        Vector2d(5, -5),
        Vector2d(5, -4),
        Vector2d(4, -4),
        Vector2d(4, -5)  // 闭合图形
    };
    
    // 依次绘制所有图形，使用不同的颜色
    plot_points_linked_shared(triangle, false, "b");   // 左上角蓝色三角形
    plot_points_linked_shared(square, false, "r");     // 右上角红色正方形
    plot_points_linked_shared(pentagon, false, "g");   // 左下角绿色五边形
    plot_points_linked_shared(rectangle, true, "r");   // 右下角红色矩形
    
    // 验证最终图像是否生成
    std::string filename = output_dir + "/combined_plot_1.png";
    std::ifstream f(filename);
    EXPECT_TRUE(f.good()) << "Failed to find file at: " << filename;
    f.close();
}
