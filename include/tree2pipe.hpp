#ifndef TREE2PIPE_H
#define TREE2PIPE_H

#include <iostream>
#include <memory>
#include <optional>

#include<vector>
#include<algorithm>

#include <Eigen/Dense>

#include <chrono>
#include <thread>

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

struct Node {
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

    shared_ptr<Node> to_tree(const CfgM1& anc_recommend) override;
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

    shared_ptr<Node> to_tree(const CfgM1& anc_recommend) override;
};

vector<Vector2d> lines_to_points(const vector<Vector3d>& lines, bool verbose = false);

void plot_points_linked(const std::vector<Vector2d>& pts);

void plot_points_linked_shared(const std::vector<Vector2d>& pts, bool show_plot = false, const std::string& color = "b");

vector<Vector2d> tree_to_points(shared_ptr<Node> root, Vector2d up_dir = Vector2d::Zero());

class PolygonM1 : public M1 {
private:
    std::vector<Vector2d> vertices; // Polygon vertices
    CfgM1 cfg;
    std::vector<std::pair<int, std::shared_ptr<M1>>> sons; // Edge id and M1 object

public:
    PolygonM1(
        const std::vector<Vector2d>& vertices,
        const CfgM1& cfg = {},
        const std::vector<std::pair<int, std::shared_ptr<M1>>>& sons = {}
    ) : vertices(vertices), cfg(cfg), sons(sons) {}

    Vector2d get_pos() const override {
        return vertices[0];
    }

    std::shared_ptr<Node> to_tree(const CfgM1& anc_recommend) override;
};

} // namespace tree2pipe

#endif
