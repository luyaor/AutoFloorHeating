// tests/test_pipe_layout_generator.cpp
#include "../include/tree2pipe.hpp"
#include <gtest/gtest.h>

using namespace tree2pipe;

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
