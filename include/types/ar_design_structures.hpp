#ifndef AR_DESIGN_STRUCTURES_H
#define AR_DESIGN_STRUCTURES_H

#include <vector>
#include <string>
#include "data_structures.hpp"
#include <map>

struct Window;
struct ToiletAndKitchenConditionHole;
struct Pipe;
struct StairArea;
struct ArcWall;
struct AirConditionHoles;
struct WallHole;
struct CurveInfo;
struct Grid;
// Structures for ARDesign.json
// 尺寸结构
struct Size {
    double Height; // 高度
    double Width; // 宽度
    double Thickness; // 厚度
};

// 门结构
struct Door {
    std::string Guid; // 全局唯一标识符
    std::string FamilyName; // 族名称
    std::string Name; // 名称
    int DoorType; // 门类型
    std::string HostWall; // 所属墙体的ID
    Point Location; // 位置
    Size DoorSize; // 尺寸
    Point FlipFaceNormal; // 门开启方向
    Point FlipHandNormal; // 把手方向
    std::string Number; // 编号
    std::string NumberForModeling; // 建模编号
    bool IsHaveShutter; // 是否有百叶
    std::vector<Size> Sizes; // 尺寸数组
    int ElementId; // 元素ID
    double MortarThickness; // 砂浆厚度
    double AirArea; // 通风面积
    std::string OpenDirection; // 开启方向
    std::string FamilyType; // 族类型
    std::string SectionSymbolName; // 剖面符号名称
    bool IsHaveHole; // 是否有洞
    bool IsDimension; // 是否标注
    int WindowBorderType; // 窗框类型
    bool IsFireWindow; // 是否防火窗
    std::string FireClass; // 防火等级
    std::string DoorTypeForNum; // 门类型编号
    std::string WindowType; // 窗类型
    std::string ShutterType; // 百叶类型
    std::string GroupNo; // 组号
    bool IsCross; // 是否交叉
    double Elevation; // 标高
};

// JCW (家具等) 结构
struct JCW {
    std::string GuidName; // 全局唯一标识符
    int Type; // 类型编号
    std::string Name; // 名称
    Point CenterPoint; // 中心点
    std::vector<CurveInfo> ShowCurves; // 显示曲线列表
    std::vector<CurveInfo> MaxBoundaryCurves; // 最大边界曲线列表
    std::vector<CurveInfo> BoundaryLines; // 边界线列表
    std::map<std::string, std::string> Parameters; // 参数列表
    bool IsBlockLayer; // 是否为块图层
};

// 房间结构
struct Room {
    std::string Guid; // 全局唯一标识符
    std::string Name; // 名称
    std::string NameType; // 名称类型
    std::vector<std::string> DoorIds; // 门ID列表
    std::vector<std::string> DoorNums; // 门编号列表
    std::vector<std::string> WindowIds; // 窗户ID列表
    std::vector<std::string> DoorAndWindowIds; // 门窗ID列表
    std::vector<std::string> DoorWayIds; // 门道ID列表
    std::vector<std::string> JCWGuidNames; // 家具等ID列表
    std::vector<std::string> JCWGuidNs; // 家具编号列表
    std::vector<Point> FloorDrainPoints; // 地漏点列表
    double ShowArea; // 显示面积
    Point AnnotationPoint; // 标注点
    double LevelOffset; // 标高偏移
    int LevelOffsetType; // 标高偏移类型
    double ArchThickness; // 建筑厚度
    double STOffSet; // 结构偏移
    double LightArea; // 采光面积
    double AirArea; // 通风面积
    double Area; // 面积
    int RoomElementId; // 房间元素ID
    std::string SectionSymbolName; // 剖面符号名称
    std::vector<std::string> WallNames; // 墙体名称列表
    bool IsOpen; // 是否开放
    std::string RoomNumber; // 房间编号
    int Number; // 编号
    std::vector<Point> Boundary; // 边界点列表
    bool IsRecreationalRoom; // 是否真实房间
};

// 房型结构
struct HouseType {
    std::string houseName; // 户型名称
    std::vector<std::string> RoomNames; // 包含的房间ID列表
    std::vector<Point> Boundary; // 户型边界点列表
};

// 建筑结构
struct Construction {
    std::vector<HouseType> houseTypes; // 户型列表
    std::vector<Room> rooms; // 房间列表
    std::vector<JCW> jcws; // 家具列表
    std::vector<Door> doors; // 门列表
    std::vector<Grid> Grid; // 轴网列表
    std::vector<Elevation> Elevation; // 标高列表
    std::vector<WallHole> WallHole; // 墙洞列表
    std::vector<ToiletAndKitchenConditionHole> ToiletAndKitchenConditionHole; // 卫生间和厨房的空调洞列表
    std::vector<CurveInfo> DrainageDitch; // 排水沟列表
    std::vector<CurveInfo> StepAndRampRailing; // 台阶和坡道栏杆列表
    std::vector<CurveInfo> RegionBreak; // 区域打断列表
    std::vector<ArcWall> ArcWall; // 弧形墙列表
    std::vector<StairArea> StairArea; // 楼梯区域列表
    std::vector<Pipe> Pipe; // 管道列表
    std::vector<CurveInfo> SourceWallLines; // 源墙线列表
    std::vector<Window> windows;  // 窗户列表
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

// 轴网结构
struct Grid {
    std::string GridTextNote; // 轴网文本注释
    Point StartPoint; // 起点
    Point EndPoint; // 终点
    int EP1Visible; // 端点1可见性
    int EP2Visible; // 端点2可见性
    int IsSub; // 是否为次轴网
    std::string BottomLevel; // 底部标高
    std::string TopLevel; // 顶部标高
    std::string Type; // 类型
    std::string Sign; // 标记
    std::string TerminalPattern; // 端点样式
    int TerminalWidth; // 端点宽度
    struct {
        // 中心段设置
        std::string Type; // 类型
        int LineWidth; // 线宽
        std::string Pattern; // 样式
        int TerminalLength; // 端点长度
    } CenterSection;

    std::vector<std::string> ShowFloorNames; // 显示楼层名称列表
    int ElementId; // 元素ID
    std::string Coordinate; // 坐标
    int Mode; // 模式
    std::vector<CurveInfo> CurveArray; // 曲线数组
};

// 标高结构
struct Elevation {
    std::string DimentionText;     // 尺寸文本
    CurveInfo DimentionLine;       // 尺寸线
    std::vector<CurveInfo> TriangleLines;  // 三角形线条
    Point TrianglePt;             // 三角形点
    Point TextLocation;           // 文本位置
    Point DimDirection;           // 尺寸方向
    int ElementId;                // 元素ID
    std::string TextType;         // 文本类型
    std::string Id;               // 标识符
    std::string DescribeText;     // 描述文本
    bool IsUnknownElevation;      // 是否未知标高
};

// 墙洞结构
struct WallHole {
    // 待补充字段
};

// 空调洞结构
struct ToiletAndKitchenConditionHole {
    std::string AirConditionName; // 空调名称
    std::vector<WallHole> AirConditionHoles; // 空调洞列表
};

// 楼层结构
struct Floor {
    std::string Name; // 楼层名称
    std::string Num; // 楼层号
    std::string AllFloor; // 所有楼层
    double LevelHeight; // 层高度
    double LevelElevation; // 楼层标高
    Construction construction; // 楼层的建筑结构
    Point BasePoint; // 基准点
    std::string RNum; // 房间编号
    std::string DrawingFrameNo; // 图框编号
};

// 标高结构
struct Level {
    std::string Name; // 楼层名称
    std::string Num; // 楼层号
    double Elevation; // 标高
    double AbsElevation; // 绝对标高
    double LevelHeight; // 层高
    int LineWidth; // 线宽
    int EP1Visible; // EP1可见性
    int EP2Visible; // EP2可见性
    std::string Type; // 类型
    std::string Sign; // 标记
    std::string Pattern; // 图案
    double LevelOffSet; // 偏移量
    int ElementId; // 元素ID
    std::vector<double> OffSet; // 偏移数组
};

// 标准信息结构
struct StandardInfo {
    std::string AllFloorNums; // 所有楼层号
    std::string Num; // 编号
};

// 建筑设计数据结构
struct ARDesign {
    std::vector<Floor> Floor; // 楼层列表
    std::vector<Level> Level; // 标高列表
    std::vector<Grid> Grid; // 顶层轴网列表
    std::vector<StandardInfo> StandardInfo; // 标准信息列表
};

// 弧形墙结构
struct ArcWall {
    std::string WallName;    // 墙体名称
    std::string Category;    // 类别
    std::string Type;        // 类型
    CurveInfo Curve;         // 中心线
    CurveInfo FirstLine;     // 第一条边线
    CurveInfo SecondLine;    // 第二条边线
    double Height;           // 高度
    double Thickness;        // 厚度
    int RoomBoundary;        // 房间边界
    Point Normal;            // 法线方向
    double BottomHeight;     // 底部高度
    int ElevationType;       // 标高类型
    std::string MaterialType; // 材料类型
    std::string ElevationId; // 标高ID，可以为 null
    bool IsUnknownElevation; // 是否未知标高，可以为 null
};

// 管道结构
struct Pipe {
    std::string Guid; // 全局唯一标识符
    Point Center; // 中心点
    double Radius; // 半径
    std::string Type; // 类型（如"雨水管"）
    std::string Location; // 位置
    std::string LevelName; // 标高名称
};

// 楼梯区域结构
struct StairArea {
    std::vector<CurveInfo> Borders; // 边界线列表
    double Offset; // 偏移量
};

// 窗户样式单元格结构
struct StyleCell {
    std::vector<Point> Vertices;  // 顶点列表
    bool IsErased;               // 是否被擦除
    std::map<std::string, int> Parameters;  // 参数列表，如"开启扇": 1
};

// 窗户视图结构
struct StyleView {
    std::vector<Point> Verticals;  // 垂直线列表
    std::vector<StyleCell> Cells;  // 单元格列表
    std::vector<CurveInfo> BorderLines;  // 边界线列表
};

// 窗户样式结构
struct WindowStyle {
    int StyleId;                // 样式ID
    std::string StyleName;      // 样式名称
    double SSMHeight;           // SSM高度
    double SSMWidth;            // SSM宽度
    double SSMBottomHeight;     // SSM底部高度
    std::vector<Point> Vertices;  // 顶点列表
    std::vector<StyleCell> Cells; // 单元格列表
    std::vector<StyleView> Views; // 视图列表
    std::string WindowsMaterial;  // 窗户材料
    std::string GlassMaterial;    // 玻璃材料
    std::map<std::string, bool> GlassConfig;    // 玻璃配置
    std::map<std::string, std::string> OtherConfig;  // 其他配置
};

// 窗户结构
struct Window {
    std::string Guid;           // 全局唯一标识符
    std::string FamilyName;     // 族名称
    std::string Name;           // 名称
    std::string Type;           // 类型
    int DoorType;              // 门类型
    std::string HostWall;      // 所属墙体
    Point Location;            // 位置
    double BottomHeight;       // 底部高度
    double TopHeight;          // 顶部高度
    Size Size;                 // 尺寸
    Point FlipFaceNormal;      // 翻转面法线
    Point FlipHandNormal;      // 翻转手法线
    bool IsVisible;            // 是否可见
    bool IsMirror;            // 是否镜像
    std::string NumberSign;    // 编号标记
    int VisibleElevationPlan; // 可见立面图
    double LevelOffset;        // 标高偏移
    CurveInfo BaseLine;        // 基准线
    bool IsFire;              // 是否防火
    WindowStyle Style;         // 窗户样式
    std::vector<double> TwoSideDepths;  // 两侧深度
    std::string GroupNo;       // 组号
    std::vector<std::string> Items;  // 项目列表
    bool IsSpecialShap;       // 是否特殊形状
};

#endif // AR_DESIGN_STRUCTURES_H
