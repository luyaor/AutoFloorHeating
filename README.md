# AutoFloorHeating

这个项目是一个地暖管线自动规划插件，可以根据输入的房间平面图和约束条件，自动生成地暖管线布局。

## 安装

依赖:

```
brew install opencv
brew install eigen
brew install jsoncpp@1.9.6
brew install matplotplusplus
```

确保您的系统已安装 CMake、OpenCV 和 JsonCpp。然后执行以下命令：
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

## 贡献
欢迎贡献代码，请遵循以下指南...
