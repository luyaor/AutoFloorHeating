#ifndef HEATING_DESIGN_STRUCTURES_H
#define HEATING_DESIGN_STRUCTURES_H

#include <vector>
#include <string>
#include "data_structures.hpp"

// Structures for HeatingDesign.json

// 回路区域结构
struct CoilArea {
    std::string AreaName;  
};

// 户型盘管回路结构
struct CoilLoop {
    float Length;  // 回路总长度
    std::vector<CoilArea> Areas;  // 回路区域
    std::vector<CurveInfo> Path;  // 回路路由
    int Curvity;  // 管道曲率半径
};

// 分集水器回路结构
struct CollectorCoil {
    std::string CollectorName;  // 集分水器编号
    int Loops;  // 回路数量
    std::vector<CoilLoop> CoilLoops;  // 户型盘管集合
    std::vector<std::vector<CurveInfo>> Deliverys;  // 入户管道集合
};

// 地暖盘管结构
struct HeatingCoil {
    std::string LevelName;  // 楼层名称
    int LevelNo;  // 楼号
    std::string LevelDesc;  // 楼层描述
    std::string HouseName;  // 户型编号
    std::vector<CurveInfo> Expansions;  // 伸缩缝集合
    std::vector<CollectorCoil> CollectorCoils;  // 分集水器回路集合
};

// 地暖系统设计结果
struct HeatingDesign {
    std::vector<HeatingCoil> HeatingCoils;  // 地暖盘管集合
};

#endif // HEATING_DESIGN_STRUCTURES_H
