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

## Todo

- 多边形

