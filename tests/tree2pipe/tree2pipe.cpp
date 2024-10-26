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

class Node {
public:
    Vector2d pos;
    double subtree_width;
    vector<shared_ptr<Node>> sons;

    Node(
        const Vector2d& pos,
        const double subtree_width = 0,
        const vector<shared_ptr<Node>>& sons = {}
    ):
        pos(pos),
        subtree_width(subtree_width),
        sons(sons) {}
};

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

vector<Vector2d> lines_to_points(const vector<Vector3d>& lines, bool verbose = false) {
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
vector<Vector2d> tree_to_points(shared_ptr<Node> root, Vector2d up_dir = Vector2d::Zero()) {
    vector<Vector3d> lines;
    dfs(root, 0, up_dir, lines, true);
    return lines_to_points(lines, false);
}

// [M1]

struct CfgM1 {
    optional<Vector2d> dir;
    optional<double> w;
};

// Abstract base class
class M1: public enable_shared_from_this<M1> {
public:
    virtual Vector2d get_pos() const = 0;
    virtual shared_ptr<Node> to_tree(const CfgM1& anc_recommend) = 0;
    virtual ~M1() = default;
};

struct RectM1Son {
    int eid;
    shared_ptr<M1> m1;
};

class RectM1: public M1 {
private:
    Vector2d p1, p3;
    CfgM1 cfg;
    vector<RectM1Son> sons;

public:
    RectM1(
        const Vector2d& p1,
        const Vector2d& p3,
        const CfgM1& cfg = {},
        const vector<RectM1Son>& sons = {}
    ):
        p1(p1),
        p3(p3),
        cfg(cfg),
        sons(sons) {}

    Vector2d get_pos() const override {
        return this->p1;
    }

    /// 输入数据需保证至少要能合法生成外围 5 个点
    /// 本函数非只读函数，由于必须知道宽度
    shared_ptr<Node> to_tree(const CfgM1& anc_recommend) override {
        const auto cfg =
            CfgM1 { this->cfg.dir.has_value() ? this->cfg.dir : anc_recommend.dir.value(),
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
                auto on_edge_node_m0 = make_shared<Node>(
                    key_nodes[eid]->pos + dir * len,
                    w,
                    vector<shared_ptr<Node>> {}
                );
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
};

class NodeM1: public M1 {
private:
    Vector2d pos;
    CfgM1 cfg;
    vector<shared_ptr<M1>> sons;

public:
    NodeM1(const Vector2d& pos, const CfgM1& cfg = {}, const vector<shared_ptr<M1>>& sons = {}):
        pos(pos),
        cfg(cfg),
        sons(sons) {}

    Vector2d get_pos() const override {
        return pos;
    }

    shared_ptr<Node> to_tree(const CfgM1& anc_recommend) override {
        vector<shared_ptr<Node>> son_trees;
        CfgM1 cfg = { std::nullopt, this->cfg.w.has_value() ? this->cfg.w : anc_recommend.w };
        const double w = cfg.w.value();
        for (const auto& son: sons) {
            son_trees.push_back(son->to_tree(CfgM1 { (son->get_pos() - this->pos).normalized(), w })
            );
        }
        return make_shared<Node>(pos, w, son_trees);
    }
};

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

int main() {
    /*
n2 = Node([1, 1], None, [])
n1 = Node([0, 0], 1, [n2])
print(f'len n1.sons {len(n1.sons)}')
lines = []
dfs(n1, None, None, lines, root=True)
print(lines)
D = 3
*/
    // auto n2 = make_shared<Node>(Vector2d(1, 1));
    // auto n1 = make_shared<Node>(Vector2d(0, 0), 1, vector<shared_ptr<Node>> { n2 });
    // vector<Vector3d> lines;
    // dfs(n1, 0, Vector2d::Zero(), lines, true);
    // for (const auto& line: lines) {
    //     std::cout << line.transpose() << std::endl;
    // }

    // Vector2d p1(2, 0), p3(0, 1);
    // double dx = -1;
    // Vector2d ovr_dir(dx, sqrt(1 - dx * dx));
    // auto root = make_shared<RectM1>(p1, p3, 0.07, vector<RectM1Son> {}, ovr_dir)->to_tree();
    // auto lines = vector<Vector3d> {};
    // dfs(root, 0, Vector2d::Zero(), lines, true);
    // for (const auto& line: lines) {
    //     std::cout << line.transpose() << std::endl;
    // }
    // auto pts = lines_to_points(lines, true);
    // for (const auto& pt: pts) {
    //     std::cout << pt.transpose() << std::endl;
    // }
    // plot_points_linked(pts);

    // Visualization and further processing would go here
    // (e.g., plotting lines or points similar to the Python code)

    /*
w = 0.15
m1_rt = NodeM1(pos=np.array([2, -1]), w=w, sons=[
    RectM1(np.array([2, 0]), np.array([0, 1]),
           w=w * 0.7, ovr_dir=np.array([dx, (1 - dx ** 2) ** 0.5])),
])
m0_rt = m1_rt.to_tree()
plot_points_linked(tree_to_points(m0_rt, None))
*/
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
        plot_points_linked(pts);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}

//
} // namespace tree2pipe

int main() {
    return tree2pipe::main();
}
