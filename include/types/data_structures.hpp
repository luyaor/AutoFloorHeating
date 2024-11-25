#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

// 坐标点结构
struct Point {
    double x;
    double y;
    double z;
};

// 曲线结构（直线或弧线）
struct CurveInfo {
    Point StartPoint;        // 起点
    Point EndPoint;          // 终点
    Point MidPoint;         // 中点（弧线时使用）
    Point Normal;           // 法线方向（弧线时使用）
    Point Center;           // 圆心（弧线时使用）
    double Radius;          // 半径（弧线时使用）
    double StartAngle;      // 起始角度（弧线时使用）
    double EndAngle;        // 结束角度（弧线时使用）
    int ColorIndex;         // 颜色索引
    int CurveType;          // 曲线类型：0-直线，1-圆弧
};

#endif // DATA_STRUCTURES_H
