# 地暖管线自动规划插件

这个项目是一个地暖管线自动规划插件，可以根据输入的房间平面图和约束条件，自动生成地暖管线布局。

## 安装

确保您的系统已安装 CMake、OpenCV 和 JsonCpp。然后执行以下命令：
   ```sh
chmod +x scripts/build.sh
./scripts/build.sh
   ```

## 使用

准备一个包含房间平面图数据和规划参数的 JSON 文件，然后运行：
   ```sh
./build/iad_planner < input.json > output.json
   ```

## 贡献
欢迎贡献代码，请遵循以下指南...