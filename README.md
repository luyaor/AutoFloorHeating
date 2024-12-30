# AutoFloorHeating

这个项目是一个地暖管线自动规划插件，可以根据输入的房间平面图和约束条件，自动生成地暖管线布局。

## 安装

### 依赖项

需要安装以下依赖：

```bash
# 核心依赖
brew install opencv@4.10.0
brew install eigen@3.4.0
brew install jsoncpp@1.9.6
brew install matplotplusplus

# 开发依赖
brew install cmake@3.28
brew install googletest
```

### libdxfrw 安装
libdxfrw 需要手动编译安装：
```bash
git clone https://github.com/LibreCAD/libdxfrw.git
cd libdxfrw
mkdir build && cd build
cmake ..
make
sudo make install
```

### 系统要求
- CMake 3.28 或更高版本
- C++20 编译器
- pkg-config

### 依赖项版本要求
- OpenCV 4.10.0
- Eigen 3.4.0
- JsonCpp 1.9.6
- Matplotplusplus (最新版本)
- libdxfrw (最新版本)
- GoogleTest (最新版本)

### 构建

确保系统已安装所有依赖后，执行以下命令：

```sh
chmod +x scripts/build.sh
./scripts/build.sh
```

## 使用

准备一个包含房间平面图数据和规划参数的 JSON 文件，然后运行：
   ```sh
./build/iad <ARDesign.json路径> <inputData.json路径>
   ```

## 算法

- [x] Json parse 
    - 平面图和数学约束条件
- 平面图遗传算法
    - 划分为面积接近的区域，每个区域由 `M1` 结点组成树形图，`M1` 结点为单点或矩形，带约束条件
- [x] 路径生成
    - 将每个 `M1` 树转换为管道路径
    - [ ] 调整
- 交付路径
    - Web

## 导出格式

### JSON 到 DWG/DXF 的转换

本项目支持将生成的地暖管线布局导出为 DXF 格式。虽然最终目标格式是 DWG，但我们选择使用 DXF 格式作为中间格式，原因如下：

1. **开放性**: DXF 是 AutoCAD 的开放交换格式，有完整的格式规范文档，而 DWG 是私有格式，缺乏官方文档支持。
2. **兼容性**: DXF 文件可以被所有主流 CAD 软件读取，包括 AutoCAD、LibreCAD 等。
3. **实现简单**: 使用开源库 libdxfrw 可以方便地生成 DXF 文件，无需处理复杂的 DWG 二进制格式。
4. **版本稳定**: DXF 格式相对稳定，不同版本间的兼容性好。

导出过程：
1. 准备包含地暖设计的 JSON 文件（例如 `HeatingDesign.json`）
2. 运行程序生成 DXF 文件：
   ```sh
   ./build/iad <HeatingDesign.json路径> --export-dxf <输出.dxf路径>
   ```
3. 生成的 DXF 文件可以直接用 AutoCAD 打开，或者转换为 DWG 格式

注意：如果需要 DWG 格式，可以使用 AutoCAD 或其他 CAD 软件将生成的 DXF 文件另存为 DWG 格式。

## Todo

- 多边形

