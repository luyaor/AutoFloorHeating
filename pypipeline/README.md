# PyPipeline

一个用于管道处理和分析的Python项目。

## 项目结构

```text
pypipeline/
├── data/           # 数据和测试数据文件
├── src/            # 源代码
│   ├── cactus.py   # 仙人掌算法实现
│   ├── partition.py # 分区处理
│   └── __init__.py
└── tests/          # 测试文件
```

## 安装说明

1. 创建虚拟环境（推荐）：

```bash
python -m venv venv
source venv/bin/activate  # Windows系统使用: venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 依赖项

主要依赖项列在 `requirements.txt` 中：

- numpy: 科学计算
- scipy: 科学算法
- matplotlib: 绘图和可视化
- networkx: 图操作
- shapely: 几何操作
- opencv-python: 计算机视觉
- loguru: 日志记录

## 工作流程

项目由三个主要组件协同工作：

### 1. 数据提供者 (test_data.py)

- 提供测试多边形顶点坐标 (`SEG_PTS`)
- 定义墙体路径 (`WALL_PT_PATH`)
- 指定区域信息 (`CAC_REGIONS_FAKE`)，包含顶点索引和颜色标识

### 2. 分区处理 (partition.py)

分区模块通过以下步骤执行区域划分：

```python
def polygon_grid_partition_and_merge(polygon_coords, num_x=3, num_y=4):
    # 步骤1: 网格划分
    # - 根据 num_x 和 num_y 创建网格线
    # - 获取自然分割线
    # - 使用网格线和自然分割线切分多边形
    
    # 步骤2: 构建邻接图
    # - 为每个子多边形分配ID
    # - 建立相邻多边形的连接关系
    
    # 步骤3: 合并小区域
    # - 循环合并面积较小的分区
    # - 根据形状评分选择最佳合并方案
    
    # 步骤4: 生成全局点列表和区域信息
    # - 收集所有多边形的边界点
    # - 去除重复和共线点
    # - 生成区域的边界点索引列表
```

### 3. 管道布线 (cactus.py)

仙人掌模块通过以下步骤处理管道布线：

1. 初始化数据结构
   - 创建 CacRegion 类存储区域信息
   - 创建 EdgePipes 类管理边上的管道
   - 设置管道宽度参数

2. Dijkstra 寻路
   - 对每个区域进行反向 Dijkstra 搜索
   - 从目标点（分水器位置）开始搜索到各个区域
   - 确定管道的大致路径

3. 管道布局优化
   - 使用 Tarjan 算法处理环路
   - 计算管道的具体位置和宽度
   - 处理管道交叉和避让

4. 生成最终结果
   - 生成管道的精确路径点
   - 输出可视化结果

### 完整工作流程

1. 从 `test_data.py` 读取输入数据（多边形顶点、墙体路径）
2. `partition.py` 进行网格分区，生成合适大小的子区域
3. `cactus.py` 接收分区结果，进行管道布线：
   - 计算各区域到目标点的路径
   - 处理管道的具体布局和宽度
   - 生成最终的管道布线方案
4. 通过可视化函数展示结果

该系统主要解决在给定多边形区域内布置不同颜色（代表不同功能）管道的问题，确保所有管道都能连接到目标点（分水器位置），同时保证管道布局合理、不交叉。
