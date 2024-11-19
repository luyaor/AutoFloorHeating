#ifndef AR_DESIGN_STRUCTURES_H
#define AR_DESIGN_STRUCTURES_H

#include <vector>
#include <string>
#include "data_structures.hpp"
#include <map>

// Structures for ARDesign.json
// 尺寸结构
struct Size {
    double Height; // 高度
    double Width; // 宽度
    double Thickness; // 厚度
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
    std::string Guid;                // 全局唯一标识符
    std::string MapGuid;            // 地图GUID
    std::string Category;           // 类别
    std::string Position;           // 位置
    int BlCreateRoom;               // 是否创建房间
    void* NewJBoundary;             // 新边界(暂时用void*表示，具体类型待定)
    std::vector<CurveInfo> Boundary; // 边界线列表
    std::string Name;               // 名称
    std::vector<std::string> Names; // 名称列表
    std::string NameType;           // 名称类型
    std::vector<std::string> DoorIds;      // 门ID列表
    std::vector<std::string> DoorNums;     // 门编号列表
    std::vector<std::string> WindowIds;    // 窗户ID列表
    std::vector<std::string> DoorAndWindowIds; // 门窗ID列表
    std::vector<std::string> DoorWayIds;   // 门道ID列表
    std::vector<std::string> JCWGuidNames; // 家具等ID列表
    std::vector<std::string> JCWGuidNs;    // 家具编号列表
    std::vector<Point> FloorDrainPoints;   // 地漏点列表
    double ShowArea;                // 显示面积
    Point AnnotationPoint;          // 标注点
    double LevelOffset;             // 标高偏移
    int LevelOffsetType;            // 标高偏移类型
    double ArchThickness;           // 建筑厚度
    double STOffSet;                // 结构偏移
    double LightArea;               // 采光面积
    double AirArea;                 // 通风面积
    double Area;                    // 面积
    int RoomElementId;              // 房间元素ID
    std::string SectionSymbolName;  // 剖面符号名称
    std::vector<std::string> WallNames; // 墙体名称列表
    bool IsOpen;                    // 是否开放
    std::string RoomNumber;         // 房间编号
    int Number;                     // 编号
    std::vector<Room> SubAreas;     // 子区域列表
    std::vector<Room> InsideRooms;  // 内部房间列表
};

// 房型结构
struct HouseType {
    std::string houseName; // 户型名称
    std::vector<std::string> RoomNames; // 包含的房间ID列表
    std::vector<Point> Boundary; // 户型边界点列表
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
    bool IsUnknownElevation;      // 否未知标高
};

// 墙洞结构
struct WallHole {
    std::string HoleName;           // 洞名称
    double HoleDisToFloor;          // 距地高度
    double HoleDiameter;            // 洞直径
    std::string HostWallName;       // 所属墙体名称
    Point JHolePosition;            // 洞位置
    Point JHostWallDir;             // 墙体方向
    Point JHostWallNormal;          // 墙体法线
    double HostWallThickness;       // 墙体厚度
    CurveInfo JHostWallLine;        // 墙体线
    int IsCreat;                    // 是否创建
    int IsInWindow;                 // 是否在窗内
    std::string WindowName;         // 窗户名称
    int HoleType;                   // 洞类型
    std::string HostJCW;            // 所属构件
};

// 空调洞结构
struct ToiletAndKitchenConditionHole {
    std::string AirConditionName; // 空调名称
    std::vector<WallHole> AirConditionHoles; // 空调洞列表
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
    double LevelOffSet; // 偏移
    int ElementId; // 元素ID
    std::vector<double> OffSet; // 偏移数组
};

// 标准信息结构
struct StandardInfo {
    std::string AllFloorNums; // 所有楼层号
    std::string Num; // 编号
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

// 墙体结构
struct Wall {
    std::string WallName;           // 墙体名称
    std::string Category;           // 类别（如"普通墙"）
    std::string Type;              // 类型（如"外墙"、"内墙"）
    std::string MaterialType;      // 材料类型
    bool HasUpWall;                // 是否有上墙
    bool IsParapetWall;            // 是否为女儿墙
    bool IsEdgeGutterWall;         // 是否为边沟墙
    Point Normal;                  // 法线方向
    CurveInfo Curve;               // 中心线
    CurveInfo* SeparationJCurve;   // 分隔J曲线（可为null）
    double SeparationThickness;    // 分隔厚度
    CurveInfo FirstLine;           // 第一条边线
    CurveInfo SecondLine;          // 第二条边线
    double Height;                 // 高度
    double Thickness;              // 厚度
    int RoomBoundary;             // 房间边界
    double BottomHeight;          // 底部高度
    int ElevationType;            // 标高类型
    std::string Material;         // 材料（可为null）
    int ElementId;                // 元素ID
    bool IsHafFloor;              // 是否为半层
    std::string JKType;           // JK类型
    std::string FloorNum;         // 楼层号
    std::string FloorName;        // 楼层名称
};

// 节点显示类型结构
struct NodeDisplay {
    int NodeDsiplayType;
    std::string NodeInfo;
    int PlaneType;
    std::string FigureClass;
    std::string Name;
    bool IsIndex;
    std::string Atlas;
    std::string Page;
    std::string SerialNumber;
    bool IsInstructions;
};

// 平面显示类型结构
struct PlaneDisplay {
    int PlaneDsiplayType;
    std::string PlaneInfo;
    int PlaneType;
    std::string FigureClass;
    std::string Name;
    bool IsIndex;
    std::string Atlas;
    std::string Page;
    std::string SerialNumber;
    bool IsInstructions;
};

// 门窗索引结构
struct DoorAndWindowIndex {
    bool IsDoor;
    int PlaneType;
    std::string FigureClass;
    std::string Name;
    bool IsIndex;
    std::string Atlas;
    std::string Page;
    std::string SerialNumber;
    bool IsInstructions;
};

// 项目建筑索引结构
struct ProjArIndexs {
    std::vector<NodeDisplay> ProjArIndexStairs;    // 楼梯索引
    std::vector<NodeDisplay> ProjArIndexWalls;     // 墙体索引
    std::vector<PlaneDisplay> ProjArIndexPlanes;   // 平面索引
    std::vector<DoorAndWindowIndex> ProjArIndexDoorAndWindows; // 门窗索引
    std::vector<NodeDisplay> ProjArIndexWCs;       // 卫生间索引
};

// 门窗参数结构
struct DWParam {
    double GeneralDoorHeight;      // 普通门高度
    double EntranceDoorHeight;     // 入口门高度
    double KitchenDoorHeight;      // 厨房门高度
    double BalconyDoorHeight;      // 阳台门高度
    double TubeWellDoorHeight;     // 管井门高度
    double UnderFoyerDoorHeight;   // 地下门厅高度
    double TubeWellThresholdHeight; // 管井门槛高度
    std::string FireDoorMaterial;   // 防火门材料
    std::string FireWindowMaterial; // 防火窗材料
    std::string FireShutterMaterial; // 防火卷帘材料
    std::string WindowMaterial;     // 窗户材料
    double WindowThickness;         // 窗户厚度
};

// 剖面信息结构
struct SectionInfo {
    std::string SectionSymbolName;      // 剖面符号名称
    std::string SectionSymbolType;      // 剖面符号类型
    CurveInfo SectionSymbolLine;        // 剖面符号线
    std::vector<std::string> LevelRange;// 标高范围
    std::vector<Room> SectionRooms;     // 剖面房间
    double SectionLength;               // 剖面长度
    double CircleLength;                // 圆周长度
    Point SectionDirection;             // 剖面方向
    Point Direction;                    // 方向
    std::string SectionFamilySymbol;    // 剖面族符号
    std::string Number;                 // 编号
    std::string DrawingNumber;          // 图纸编号
    CurveInfo SectionLine;              // 剖面线
    std::string SectionView;            // 剖面视图
    int SectionType;                    // 剖面类型
    std::string WindowDrawingModel;     // 窗户绘图模型
    std::string DoorDrawingModel;       // 门绘图模型
    int VisibleElevationPlan;          // 可见立面图
    int SectionSymbol;                 // 剖面符号
};

// 图框信息结构
struct FrameInfo {
    double Scale;                       // 比例
    double Length;                      // 长度
    double Width;                       // 宽度
};

// 楼板结构
struct STSlab {
    std::string FloorNum;              // 楼层号
    std::string RoomName;              // 房间名称
    std::vector<CurveInfo> Boundary;   // 边界线列表
    std::string Type;                  // 类型
    double Thickness;                  // 厚度
    Point Location;                    // 位置
    std::string RoomGuid;             // 房间GUID
    std::string ID;                    // ID
    double Height;                     // 高度
    std::string ElementId;             // 元素ID
    int IsLowerPlate;                  // 是否为下板
    std::vector<Wall> Walls;           // 墙体列表
    int RoomIsHole;                    // 房间是否为洞
    std::string Guid;                  // GUID
    std::string MaterialType;          // 材料类型
    std::string SlabName;              // 楼板名称
};

// 楼层类型结构
struct FloorType {
    std::string FloorNum;              // 楼层号
    std::string Type;                  // 楼层类型（如"平面图"、"屋面层平面图"等）
};

// 防火信息结构
struct FireInfo {
};

// 建筑通用信息结构
struct ARGeneralInfo {
    std::string Company;           // 公司名称
    std::string BuildingName;      // 建筑名称
    std::string Address;           // 地址
    double Area;                   // 面积
    std::string Floors;           // 楼层数
    double Height;                // 高度
    std::string Intensity;        // 强度
    int SeismicDegree;           // 抗震等级
    std::string STType;          // 结构类型
    std::string PlotTime;        // 绘图时间
    FrameInfo FrameInfo;         // 图框信息
    std::vector<FireInfo> FireInfos; // 防火信息列表
    std::string ArFloorOn;       // 地上层数
    std::string ArFloorUnder;    // 地下层数
    std::string ArFloorRemark;   // 楼层备注
    std::string Climatepart;     // 气候分区
    std::string Province;        // 省份
    std::string City;            // 城市
    bool IsOpenJK;              // 是否开放JK
    std::string FireLevel;      // 防火等级
    std::string RoofingLevel;   // 屋面等级
    std::string Type;           // 类型
    std::string MGTpye;         // MGTpye
};

// 墙体保温结构
struct Insulation {
    // 待补充字段
};

// 元数据墙体结构
struct MetaWall {
    std::vector<Wall> RectWalls;      // 矩形墙体列表
    std::vector<Wall> NonRectWalls;   // 非矩形墙体列表
    std::vector<STSlab> STSlabs;      // 楼板列表
    std::vector<Insulation> Insulations; // 保温层列表
};

// 区域边模型结构
struct AreaBoundaryModel {
    std::vector<CurveInfo> JBoundary;      // 边界线列表
    double Area;                           // 面积
};

// 楼层面积信息结构
struct FloorAreaInfo {
    std::string FloorNum;                  // 楼层号
    double LevelHeight;                    // 层高
    double FloorArea;                      // 楼层面积
    std::vector<CurveInfo> JBoundarys;     // 边界线列表
    std::vector<void*> OneFloorJKDoorAreas;// 单层门窗面积列表
    std::vector<AreaBoundaryModel> AreaBoundaryModels; // 区域边界模型列表
};

// 建筑面积信息结构
struct ARArea {
    double TotalArea;                      // 总面积
    std::vector<FloorAreaInfo> FloorArea;  // 楼层面积信息列表
};

// 集水槽结构
struct WaterDustpanItem {
    Point Location;                // 位置
    CurveInfo Line;               // 线条
};

// 楼层集水槽结构
struct FloorWaterDustpan {
    std::vector<Point> RainPipelocations;  // 雨水管位置
    std::string Num;                       // 楼层号
    std::vector<WaterDustpanItem> WaterDustpans;  // 集水槽列表
};

// 坑结构
struct Pit {
    std::string Number;           // 编号
    std::string Name;            // 名称
    std::string PitCategory;     // 坑类别
    double BaseLevelElevation;   // 基准标高
    Point Location;                 // 点位置
    Point Center;                // 中心点
    double Length;               // 长度
    double Width;                // 宽度
    double Heigh;                // 高度
};

// 窗户样式结构
struct WindowStyle {
    int StyleId;                      // 样式ID
    std::string StyleName;            // 样式名称
    double SSMHeight;                 // SSM高度
    double SSMWidth;                  // SSM宽度
    double SSMBottomHeight;           // SSM底部高度
    std::vector<Point> Vertices;    // 顶点数组
    std::vector<StyleCell> Cells; // 单元格列表
    std::vector<StyleView> Views; // 视图列表
    std::string WindowsMaterial;  // 窗户材料
    std::string GlassMaterial;    // 玻璃材料
    std::map<std::string, bool> GlassConfig;    // 玻璃配置，如 "安全玻璃": true
    std::map<std::string, std::string> OtherConfig;  // 其他配置，如 "otherconfig5": "0"
};

// 门窗结构（统一处理门和窗）
struct DoorAndWindow {
    std::string Guid;          // 全局唯一标识符
    std::string FamilyName;    // 族名称
    std::string Name;          // 名称
    std::string Type;          // 类型（"门"或"窗"）
    int DoorType;             // 门类型
    std::string HostWall;      // 所属墙体
    Point Location;            // 位置
    double BottomHeight;       // 底部高度
    double TopHeight;          // 顶部高度
    Size dimensions;          // 尺寸 (renamed from Size)
    Point FlipFaceNormal;      // 翻转面法线
    Point FlipHandNormal;      // 翻转手法线
    int IsVisible;            // 是否可见
    int IsMirror;             // 是否镜像
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

    // 门特有属性
    std::string Number;        // 编号
    std::string NumberForModeling; // 建模编号
    bool IsHaveShutter;       // 是否有百叶
    Size Size;  // 尺寸数组
    double MortarThickness;   // 砂浆厚度
    double AirArea;           // 通风面积
    std::string OpenDirection; // 开启方向
    std::string FamilyType;    // 族类型
    std::string SectionSymbolName; // 剖面符号名称
    bool IsHaveHole;          // 是否有洞
    bool IsDimension;         // 是否标注
    int WindowBorderType;     // 窗框类型
    bool IsFireWindow;        // 是否防火窗
    std::string FireClass;    // 防火等级
    std::string DoorTypeForNum; // 门类型编号
    std::string WindowType;    // 窗类型
    std::string ShutterType;   // 百叶类型
    bool IsCross;             // 是否交叉
    double Elevation;         // 标高

    std::map<std::string, std::string> Parameters; // 参数列表
    std::string Material; // 材料
    std::vector<CurveInfo> WindowFrame; // 窗框
    std::vector<CurveInfo> WindowGlassModel; // 窗玻璃模型
    std::map<std::string, std::string> OpenInfo; // 开启信息
    CurveInfo WindowWidthLine; // 窗宽度线
};



// 建筑结构
struct Construction {
    std::vector<HouseType> houseTypes;     // 户型列表
    std::vector<Room> rooms;               // 房间列表
    std::vector<JCW> jcws;                 // 家具列表
    std::vector<DoorAndWindow> doorAndWindows;  // 门窗列表
    std::vector<Grid> Grid;                // 轴网列表
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
    std::vector<Wall> walls; // 墙体列表
};

// Web参数结构
struct WebParam {
    std::vector<Construction> Constructions;
    DWParam DoorWindowParam;
    ProjArIndexs ProjArIndexs;
    std::string PlotTime;
    std::string PlotStage;
    std::string NorthAngle;
    std::string Climatepart;
    std::string Province;
    std::string City;
    std::vector<std::string> RoomTypes;
    std::string STOffsetDirction;
    std::string STTpye;
    std::string MGTpye;
    std::string Guadcbwhd;
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

// 平面元素样式配置
struct PlaneElementStyle {
    std::string Type;
    int DimensionDrawing;
    std::string IndexNumbel;
    std::string LegendStyle;
};

// 公司统一标准配置
struct CompanyUnifiedStandarsConfig {
    struct {
        bool IsDimension_inHouse; // 是否在户内标注
    } InternalDimensionConfig;
    
    struct {
        std::string Style;
        std::string Location;
    } Rail_balconyConfig, Rail_protectWinConfig;
    
    struct {
        std::string Style_Elevation;
        std::string WindowFrameWidth;
        struct {
            bool IsHasGlassSymbol_FixedWin;
            bool IsHasGlassSymbol_OpenWin;
        } GlassSymbolStyle;
        std::string WinOpenLineStyle;
    } DoorAndWinStyleConfig;
    
    struct MaterialFillingConfig {
        std::string Type;
        std::string PlaneStyle;
        std::string HouseTypeStyle;
        std::string StairSectionStyle;
        std::string WallSectionStyle;
    } MaterialFillingConfig;

    struct FlueHoleConfig {
        std::string FlueHoleSymbol;
    } FlueHoleConfig;

    struct InsulationConfig {
        std::string InsulationStyle;
    } InsulationConfig;

    struct SteelLadderConfig {
        std::string SteelLadderStyle;
    } SteelLadderConfig;

    struct SplashBlockConfig {
        std::string SplashBlockStyle;
    } SplashBlockConfig;

    struct PlaneElementsStyleConfig {
        PlaneElementStyle RainPip;               // 雨水管
        PlaneElementStyle CondensatePip;         // 冷凝水管
        PlaneElementStyle SewagePip_balcony;     // 污水管（阳台）
        PlaneElementStyle SewagePip_kitAndtoi;   // 污水管（厨房、卫生间）
        PlaneElementStyle WastePip;              // 废水管
        PlaneElementStyle FireStandPip;          // 消防立管
        PlaneElementStyle FireHydrant;           // 消火栓
        PlaneElementStyle Drain_1;               // 地漏（卫生间、空调板、阳台雨水）
        PlaneElementStyle Drain_2;               // 地漏（洗衣机）
        PlaneElementStyle ElectricalBoxHigh_V;   // 强电箱
        PlaneElementStyle ElectricalBoxLow_V;    // 弱电箱
        PlaneElementStyle VideoPhone;            // 视频对讲机
        PlaneElementStyle RainStrainer_Roof;     // 屋面雨水斗
        PlaneElementStyle RainStrainer_Side;     // 侧排雨水斗
        PlaneElementStyle OverflowPipe;          // 溢流管
        PlaneElementStyle AirConditionHole_Low;  // 低位空调预留洞
        PlaneElementStyle AirConditionHole_High; // 高位空调预留洞
        PlaneElementStyle AirConditionHole_Symbol; // 空调预留洞内外高差符号
        PlaneElementStyle ToiletHole;            // 卫生间排气洞
        PlaneElementStyle KitchenHole;           // 厨房排气洞
    } PlaneElementsStyleConfig;

    struct AirConditionerBracketConfig {
        std::string FloorHeightSyle;
        std::string HafFloorHeightSyle;
    } AirConditionerBracketConfig;

    struct LevelStyleConfig {
        std::string PlaneLevelStyle;
        std::string EleLevelStyle;
    } LevelStyleConfig;
};

// 项目统一标准配置
struct ProjectUnifiedStandarsConfig {
    struct DoubleAirConditionPanelConfig {
        std::string TwoHouse;
        std::string OneHouse;
    } DoubleAirConditionPanelConfig;

    struct WaterDispersalConfig {
        double Width;
        std::string Style;
    } WaterDispersalConfig;

    struct ConcreteWallConfig {
        bool IsAll;
    } ConcreteWallConfig;

    struct WaterProofConfig {
        std::string Position;
        double OnHeight;
        int LineWidth;
    };

    struct WaterProofStyleConfig {
        WaterProofConfig UndergroundFirstFloor;
        WaterProofConfig WaterRoom;
    } WaterProofStyleConfig;

    struct SlopeConfig {
        std::string Position;
        std::string SlopeValue;
    };

    struct SlopeStyleConfig {
        SlopeConfig Toilet;
        SlopeConfig Shower;
        SlopeConfig Lanai;
        SlopeConfig Balcony;
        SlopeConfig Aircondition;
        SlopeConfig OutsideWindowSill;
        SlopeConfig Parapet;
        SlopeConfig Roof;
        SlopeConfig Canopy;
        SlopeConfig GarageRoof;
        SlopeConfig WaterWell;
        SlopeConfig WaistLine;
    } SlopeStyleConfig;

    struct WallSectionGridConfig {
        bool IsTrueNum;
        std::string ReplaceSymbol;
    } WallSectionGridConfig;

    struct RoofFlueConfig {
        bool IsDrawing;
        double Height;
    } RoofFlueConfig;

    struct DoorSillConfig {
        double UnderGroundWaterRoom;
        double UnderGroundEleRoom;
        double EquipmentWell;
        double OutRoof;
        double HeatRoom;
        double WaterPumpRoom;
        double ElevatorRoom;
    } DoorSillConfig;

    bool RefugeIsFireWinConfig;

    struct ARInsulationConfig {
        std::string Style;
        std::string Materials;
        double Thickness;
    } ARInsulationConfig;
};

// 建筑统一标准
struct ARUniformStanDards {
    CompanyUnifiedStandarsConfig CompanyUnifiedStandarsConfig;
    ProjectUnifiedStandarsConfig ProjectUnifiedStandarsConfig;
};

// 建筑设计数据结构
struct ARDesign {
    std::vector<Floor> Floor;                  // 楼层列表
    std::vector<Level> Level;                  // 标高列表
    std::vector<Grid> Grid;                    // 顶层轴网列表
    std::vector<StandardInfo> StandardInfo;    // 标准信息列表
    WebParam WebParam;                         // Web参数
    std::vector<SectionInfo> SectionInfos;     // 剖面信息列表
    FrameInfo FrameInfo;                       // 图框信息
    std::vector<STSlab> STSlabs;              // 楼板列表
    double ARHeight;                           // 建筑高度
    std::vector<FloorType> FloorType;         // 楼层类型列表
    int FloorNumber;                           // 楼层数量
    std::vector<void*> FireCompartment;        // 防火分区列表
    ARGeneralInfo ARGeneralInfo;               // 建筑通用信息
    MetaWall MetaWall;                         // 元数据墙体
    std::vector<void*> AccessHole;             // 检修口列表
    std::vector<void*> Shaft;                  // 竖井列表
    ARArea ARArea;                             // 建筑面积信息
    std::vector<FloorWaterDustpan> WaterDustpan;  // 集水槽列表
    std::vector<void*> Ditchs;                    // 沟槽列表
    std::vector<Pit> Pit;                         // 坑列表
    ARUniformStanDards ARUniformStanDards;        // 建筑统一标准
};

#endif // AR_DESIGN_STRUCTURES_H
