#include "tree2pipe.hpp"

#include <iostream>
#include <memory>
#include <optional>

#include <Eigen/Core>
#include <matplot/matplot.h>

namespace tree2pipe {

using std::array;
using std::cout;
using std::enable_shared_from_this;
using std::endl;
using std::make_shared;
using std::optional;
using std::pair;
using std::shared_ptr;
using std::vector;

using Eigen::Vector2d;
using Eigen::Vector3d;

const double EPS = 1e-8;

// Line 使用 (A, B, C) 表示 Ax + By + C = 0

bool same_line(const Vector3d& p, const Vector3d& q) {
    return abs(p[0] * q[1] - p[1] * q[0]) < EPS && abs(p[0] * q[2] - p[2] * q[0]) < EPS;
}

// def up_left(pos: np.array, dir: np.array, fa_width, width):
//     # dir should be unit
//     left_dir = np.array([-dir[1], dir[0]])
//     return pos + dir * fa_width + left_dir * width
Vector2d up_left(const Vector2d& pos, const Vector2d& dir, double fa_width, double width) {
    Vector2d left_dir(-dir[1], dir[0]);
    return pos + dir * fa_width + left_dir * width;
}

Vector2d up_right(const Vector2d& pos, const Vector2d& dir, double fa_width, double width) {
    Vector2d right_dir(dir[1], -dir[0]);
    return pos + dir * fa_width + right_dir * width;
}

Vector2d dir_left(const Vector2d& dir) {
    return Vector2d(-dir[1], dir[0]);
}

Vector2d dir_right(const Vector2d& dir) {
    return Vector2d(dir[1], -dir[0]);
}

Vector2d normalized(const Vector2d& v) {
    return v / v.norm();
}

Vector3d line_at_dir(const Vector2d& at, const Vector2d& dir) {
    const double a = -dir[1];
    const double b = dir[0];
    const double c = -a * at[0] - b * at[1];
    return Vector3d(a, b, c);
}

/// 两直线交点
Vector2d line_cross(const Vector3d& a, const Vector3d& b) {
    if (abs(a[0] * b[1] - a[1] * b[0]) < EPS) {
        throw std::runtime_error(
            "Parallel lines that can't reach each other when calculating cross point"
        );
    }
    const double x = (a[1] * b[2] - a[2] * b[1]) / (a[0] * b[1] - a[1] * b[0]);
    const double y = (a[2] * b[0] - a[0] * b[2]) / (a[0] * b[1] - a[1] * b[0]);
    return Vector2d(x, y);
}

/// 将 Node 树转换为管道结点，存入 lines
void dfs(
    shared_ptr<Node> u,
    const double up_width,
    Vector2d up_dir,
    vector<Vector3d>& lines,
    bool root = false
) {
    // 根方向特判
    if (root && up_dir.isZero()) {
        assert(u->sons.size() == 1);
        up_dir = (u->sons[0]->pos - u->pos).normalized();
    }

    const double width = u->subtree_width ? u->subtree_width : up_width;

    auto lines_append = [&](const Vector3d& line) {
        if (!lines.empty() && same_line(lines.back(), line))
            return;
        lines.push_back(line);
    };

    if (u->sons.size() > 1) {
        sort(
            u->sons.begin(),
            u->sons.end(),
            [&](const shared_ptr<Node>& v1, const shared_ptr<Node>& v2) {
                const Vector2d dir1 = normalized(v1->pos - u->pos);
                const Vector2d dir2 = normalized(v2->pos - u->pos);
                const double angle1 = atan2(dir1.dot(dir_left(up_dir)), dir1.dot(up_dir));
                const double angle2 = atan2(dir2.dot(dir_left(up_dir)), dir2.dot(up_dir));
                return angle1 < angle2;
            }
        );
    }

    for (auto& v: u->sons) {
        const Vector2d dir = normalized(v->pos - u->pos);
        const Vector3d line = line_at_dir(u->pos, dir);
        lines_append(Vector3d(line[0], line[1], line[2] + width / 2.0)); // dir right
        dfs(v, width, dir, lines);
        lines_append(Vector3d(line[0], line[1], line[2] - width / 2.0)); // dir left
    }

    if (u->sons.empty()) {
        lines_append(line_at_dir(u->pos + up_dir * width / 2.0, dir_left(up_dir)));
    }

    if (root) {
        lines_append(line_at_dir(u->pos, dir_left(up_dir)));
    }
}

/// pub
vector<Vector2d> lines_to_points(const vector<Vector3d>& lines, bool verbose) {
    vector<Vector2d> points;
    for (size_t i = 0; i < lines.size(); ++i) {
        int pre = i == 0 ? lines.size() - 1 : i - 1;
        points.push_back(line_cross(lines[pre], lines[i]));
        if (verbose) {
            std::cout << "cross 2" << lines[pre].transpose() << " and " << lines[i].transpose()
                      << " = " << points.back().transpose() << std::endl;
        }
    }
    return points;
}

/*
def tree_to_points(root: Node, up_dir=None) -> List[np.array]:
    lines = []
    dfs(root, None, up_dir, lines, root=True, verbose=False)
    return lines_to_points(lines, verbose=False)
*/
vector<Vector2d> tree_to_points(shared_ptr<Node> root, Vector2d up_dir) {
    vector<Vector3d> lines;
    dfs(root, 0, up_dir, lines, true);
    return lines_to_points(lines, false);
}

// [M1]
shared_ptr<Node> RectM1::to_tree(const CfgM1& anc_recommend) {
    const auto cfg = CfgM1 { this->cfg.dir.has_value() ? this->cfg.dir : anc_recommend.dir.value(),
                             this->cfg.w.has_value() ? this->cfg.w : anc_recommend.w.value() };
    const auto dir = cfg.dir.value();
    const auto w = cfg.w.value();
    std::function<Vector2d(const Vector2d&)> nxt_dir =
        dir_left(dir).dot(p3 - p1) > 0 ? dir_left : dir_right;
    auto nxt_internal_pt_dir = dir_left(dir).dot(p3 - p1) > 0 ? up_left : up_right;
    double rect_width = w * 2.0;
    Vector2d p2 = line_cross(line_at_dir(p1, dir), line_at_dir(p3, nxt_dir(dir)));
    Vector2d p4 = line_cross(line_at_dir(p3, dir), line_at_dir(p1, nxt_dir(dir)));
    Vector2d p5 = p1 + nxt_dir(dir) * rect_width;

    vector<pair<Vector2d, Vector2d>> q = { { p1, dir },
                                           { p2, nxt_dir(dir) },
                                           { p3, nxt_dir(nxt_dir(dir)) },
                                           { p4, nxt_dir(nxt_dir(nxt_dir(dir))) },
                                           { p5, dir } };

    while (true) {
        auto [u, dir] = q.back();
        auto [same_corner, s_dir] = q[q.size() - 4];
        Vector2d v = nxt_internal_pt_dir(same_corner, s_dir, rect_width, rect_width);
        if ((v - u).norm() > rect_width / 2.0) {
            q.emplace_back(v, s_dir);
        }
        if ((v - u).norm() < rect_width) {
            break;
        }
    }

    auto rt = make_shared<Node>(q[0].first, w);
    auto current = rt;
    for (size_t i = 1; i < q.size(); ++i) {
        current->sons.push_back(make_shared<Node>(q[i].first, w));
        current = current->sons[0];
    }
    // 此时均只有一个孩子，已成为 Node 类型
    array<shared_ptr<Node>, 5> key_nodes = { rt, // p1
                                             rt->sons[0],
                                             rt->sons[0]->sons[0],
                                             rt->sons[0]->sons[0]->sons[0],
                                             rt->sons[0]->sons[0]->sons[0]->sons[0] };
    array<vector<shared_ptr<M1>>, 4> e_sons;
    for (const auto& [eid, m1]: sons) {
        e_sons[eid].push_back(m1);
    }
    for (int eid = 0; eid < 4; eid++) {
        auto dir = (key_nodes[eid + 1]->pos - key_nodes[eid]->pos).normalized();
        sort(
            e_sons[eid].begin(),
            e_sons[eid].end(),
            [&](const shared_ptr<M1>& m1_1, const shared_ptr<M1>& m1_2) {
                return (m1_1->get_pos() - key_nodes[eid]->pos).dot(dir)
                    < (m1_2->get_pos() - key_nodes[eid]->pos).dot(dir);
            }
        );
        key_nodes[eid]->sons.clear(); // 清除指向下一个关键点的边，
        auto cur_fa = key_nodes[eid];
        for (const auto& m1: e_sons[eid]) {
            const double len = (m1->get_pos() - key_nodes[eid]->pos).dot(dir);
            auto on_edge_node_m0 =
                make_shared<Node>(key_nodes[eid]->pos + dir * len, w, vector<shared_ptr<Node>> {});
            on_edge_node_m0->sons.push_back(
                m1->to_tree(CfgM1 { (m1->get_pos() - on_edge_node_m0->pos).normalized(), w })
            );
            cur_fa->sons.push_back(on_edge_node_m0);
            cur_fa = on_edge_node_m0;
        }
        // 把关键点加回来
        cur_fa->sons.push_back(key_nodes[eid + 1]);
    }
    return rt;
}

shared_ptr<Node> NodeM1::to_tree(const CfgM1& anc_recommend) {
    vector<shared_ptr<Node>> son_trees;
    CfgM1 cfg = { std::nullopt, this->cfg.w.has_value() ? this->cfg.w : anc_recommend.w };
    const double w = cfg.w.value();
    for (const auto& son: sons) {
        son_trees.push_back(son->to_tree(CfgM1 { (son->get_pos() - this->pos).normalized(), w }));
    }
    return make_shared<Node>(pos, w, son_trees);
}

std::shared_ptr<Node> PolygonM1::to_tree(const CfgM1& anc_recommend) {
    const auto cfg = CfgM1{
        this->cfg.dir.has_value() ? this->cfg.dir : anc_recommend.dir,
        this->cfg.w.has_value() ? this->cfg.w : anc_recommend.w
    };
    const auto w = cfg.w.value();// 宽度参数

    double offset_distance = w * 2.0; // 偏移距离
    const double min_area_threshold = 0.1; // 最小面积阈值
    const double min_distance_threshold = w; // 最小顶点间距离

    // 初始化多边形的顶点序列
    std::vector<std::vector<Vector2d>> polygons;
    polygons.push_back(vertices);

    int max_iterations = 100; // 防止死循环的最大迭代次数
    int iteration = 0;

    // 开始生成回旋的多边形结构
    while (iteration < max_iterations) {
        iteration++;
        const auto& current_vertices = polygons.back();
        size_t n = current_vertices.size();
        std::vector<Vector3d> offset_lines(n);

        // 计算每条边的偏移直线
        bool valid = true;
        for (size_t i = 0; i < n; ++i) {
            Vector2d v1 = current_vertices[i];
            Vector2d v2 = current_vertices[(i + 1) % n];
            Vector2d edge_dir = (v2 - v1).normalized();

            Vector2d normal = dir_left(edge_dir); // 内法线方向

            // 计算偏移直线上的新点
            Vector2d offset_point = v1 + offset_distance * normal;

            // 用偏移点和原方向生成偏移直线
            Vector3d line = line_at_dir(offset_point, edge_dir);

            offset_lines[i] = line;
        }

        // 计算相邻偏移直线的交点
        std::vector<Vector2d> offset_vertices;
        for (size_t i = 0; i < n; ++i) {
            std::cout << "i: " << i << std::endl;
            Vector3d line1 = offset_lines[(i + n - 1) % n];
            Vector3d line2 = offset_lines[i % n];
            try {
                Vector2d intersection = line_cross(line1, line2);
                offset_vertices.push_back(intersection);
            } catch (const std::runtime_error& e) {
                // 平行线，无法计算交点，结束回旋
                valid = false;
                break;
            }
        }

        if (!valid || offset_vertices.size() < 2) {
            break; // 无法继续偏移
        }

        // 计算面积，终止条件
        double area = 0.0;
        for (size_t i = 0; i < offset_vertices.size(); ++i) {
            const auto& v1 = offset_vertices[i];
            const auto& v2 = offset_vertices[(i + 1) % offset_vertices.size()];
            area += v1.x() * v2.y() - v1.y() * v2.x();
        }
        area = std::abs(area) / 2.0;

        if (area < min_area_threshold) {
            break; // 面积过小
        }

        polygons.push_back(offset_vertices); // 添加新的回旋多边形
    }

    // 构建节点树
    std::vector<std::vector<std::shared_ptr<Node>>> node_layers;
    for (const auto& polygon : polygons) {
        std::vector<std::shared_ptr<Node>> nodes;
        for (const auto& v : polygon) {
            nodes.push_back(std::make_shared<Node>(v, w));
        }
        std::cout << std::endl;
        node_layers.push_back(nodes);
    }

    // 构建线形连接
    auto cur_node = node_layers[0][0]; // 当前节点

    for (size_t layer = 0; layer < node_layers.size(); ++layer) {
        auto& current_layer = node_layers[layer];
        for (size_t i = 1; i < current_layer.size(); ++i) {
            cur_node->sons.push_back(current_layer[i]); // 连接当前节点
            cur_node = current_layer[i]; // 更新当前节点
        }

        // 层间连接：当前层最后一个点连接到下一层第一个点
        if (layer + 1 < node_layers.size()) {
            cur_node->sons.push_back(node_layers[layer + 1][0]);
            cur_node = node_layers[layer + 1][0]; // 更新到下一层第一个点
        }
    }

    // return root_node;
    return node_layers[0][0];
}


void plot_points_linked(const std::vector<Vector2d>& pts) {
    using namespace matplot;
    std::vector<double> x, y;
    for (const auto& pt: pts) {
        x.push_back(pt.x());
        y.push_back(pt.y());
    }

    double x_max = *std::max_element(x.begin(), x.end());
    double x_min = *std::min_element(x.begin(), x.end());
    double y_max = *std::max_element(y.begin(), y.end());
    double y_min = *std::min_element(y.begin(), y.end());

    double x_mid = (x_max + x_min) / 2.0;
    double y_mid = (y_max + y_min) / 2.0;
    double width = std::max(x_max - x_min, y_max - y_min) * 1.2;

    auto fig = figure(true);
    fig->size(600, 600);
    plot(x, y)->line_width(2).color("b");
    xlabel("x");
    ylabel("y");
    title("Points Plot");
    xlim({ x_mid - width / 2, x_mid + width / 2 });
    ylim({ y_mid - width / 2, y_mid + width / 2 });
    grid(true);
    show();
}

//
} // namespace tree2pipe
