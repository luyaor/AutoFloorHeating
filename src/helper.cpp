//
// Created by Wang Tai on 2024/10/20.
//

#include "helper.h"



std::string planToJson(const std::vector<cv::Point>& plan) {
    // 实现将规划结果转换为JSON的逻辑
    Json::Value root;
    // ... 填充 root
    Json::FastWriter writer;
    return writer.write(root);
}


CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson){
    CombinedData combinedData;
    Json::Value arDesignRoot, inputDataRoot;
    Json::Reader reader;

    // Parse ARDesign.json
    if (reader.parse(arDesignJson, arDesignRoot)) {
        const Json::Value& floors = arDesignRoot["Floor"];
        for (const auto& floor : floors) {
            Floor f;
            f.Name = floor["Name"].asString();
            f.Num = floor["Num"].asString();
            f.AllFloor = floor["AllFloor"].asString();
            f.LevelHeight = floor["LevelHeight"].asDouble();
            f.LevelElevation = floor["LevelElevation"].asDouble();

            const Json::Value& construction = floor["Construction"];
            const Json::Value& grids = construction["Grid"];
            for (const auto& grid : grids) {
                Grid g;
                g.GridTextNote = grid["GridTextNote"].asString();
                g.StartPoint = {grid["StartPoint"]["X"].asDouble(), grid["StartPoint"]["Y"].asDouble(), grid["StartPoint"]["Z"].asDouble()};
                g.EndPoint = {grid["EndPoint"]["X"].asDouble(), grid["EndPoint"]["Y"].asDouble(), grid["EndPoint"]["Z"].asDouble()};
                g.EP1Visible = grid["EP1Visible"].asInt();
                g.EP2Visible = grid["EP2Visible"].asInt();
                g.IsSub = grid["IsSub"].asInt();
                f.Construction.Grid.push_back(g);
            }
            combinedData.arDesign.Floor.push_back(f);
        }
    }

    // Parse inputData.json
    if (reader.parse(inputDataJson, inputDataRoot)) {
        const Json::Value& webData = inputDataRoot["WebData"];
        combinedData.inputData.WebData.ImbalanceRatio = webData["ImbalanceRatio"].asInt();
        combinedData.inputData.WebData.JointPipeSpan = webData["JointPipeSpan"].asInt();
        combinedData.inputData.WebData.DenseAreaWallSpan = webData["DenseAreaWallSpan"].asInt();
        combinedData.inputData.WebData.DenseAreaSpanLess = webData["DenseAreaSpanLess"].asInt();

        // Parse LoopSpanSet
        const Json::Value& loopSpanSet = webData["LoopSpanSet"];
        for (const auto& loopSpan : loopSpanSet) {
            LoopSpan ls;
            ls.TypeName = loopSpan["TypeName"].asString();
            ls.MinSpan = loopSpan["MinSpan"].asInt();
            ls.MaxSpan = loopSpan["MaxSpan"].asInt();
            ls.Curvity = loopSpan["Curvity"].asInt();
            combinedData.inputData.WebData.LoopSpanSet.push_back(ls);
        }

        // Parse ObsSpanSet
        const Json::Value& obsSpanSet = webData["ObsSpanSet"];
        for (const auto& obsSpan : obsSpanSet) {
            ObsSpan os;
            os.ObsName = obsSpan["ObsName"].asString();
            os.MinSpan = obsSpan["MinSpan"].asInt();
            os.MaxSpan = obsSpan["MaxSpan"].asInt();
            combinedData.inputData.WebData.ObsSpanSet.push_back(os);
        }

        // Parse DeliverySpanSet
        const Json::Value& deliverySpanSet = webData["DeliverySpanSet"];
        for (const auto& deliverySpan : deliverySpanSet) {
            ObsSpan ds;
            ds.ObsName = deliverySpan["ObsName"].asString();
            ds.MinSpan = deliverySpan["MinSpan"].asInt();
            ds.MaxSpan = deliverySpan["MaxSpan"].asInt();
            combinedData.inputData.WebData.DeliverySpanSet.push_back(ds);
        }

        // Parse PipeSpanSet
        const Json::Value& pipeSpanSet = webData["PipeSpanSet"];
        for (const auto& pipeSpan : pipeSpanSet) {
            PipeSpan ps;
            ps.LevelDesc = pipeSpan["LevelDesc"].asString();
            ps.FuncName = pipeSpan["FuncName"].asString();
            for (const auto& direction : pipeSpan["Directions"])
                ps.Directions.push_back(direction.asString());
            ps.ExterWalls = pipeSpan["ExterWalls"].asInt();
            ps.PipeSpan = pipeSpan["PipeSpan"].asDouble();
            combinedData.inputData.WebData.PipeSpanSet.push_back(ps);
        }

        // Parse ElasticSpanSet
        const Json::Value& elasticSpanSet = webData["ElasticSpanSet"];
        for (const auto& elasticSpan : elasticSpanSet) {
            ElasticSpan es;
            es.FuncName = elasticSpan["FuncName"].asString();
            es.PriorSpan = elasticSpan["PriorSpan"].asDouble();
            es.MinSpan = elasticSpan["MinSpan"].asDouble();
            es.MaxSpan = elasticSpan["MaxSpan"].asDouble();
            combinedData.inputData.WebData.ElasticSpanSet.push_back(es);
        }

        // Parse FuncRooms
        const Json::Value& funcRooms = webData["FuncRooms"];
        for (const auto& funcRoom : funcRooms) {
            FuncRoom fr;
            fr.FuncName = funcRoom["FuncName"].asString();
            for (const auto& roomName : funcRoom["RoomNames"])
                fr.RoomNames.push_back(roomName.asString());
            combinedData.inputData.WebData.FuncRooms.push_back(fr);
        }
    }

    return combinedData;
}
std::vector<cv::Point> generatePipePlan(const CombinedData& combinedData){
    std::vector<cv::Point> pipePlan;

    // 获取楼层信息
    const auto& floors = combinedData.arDesign.Floor;
    if (floors.empty()) {
        return pipePlan;  // 如果没有楼层信息，返回空的管线规划
    }

    // 假设我们只处理第一个楼层
    const auto& floor = floors[0];
    const auto& grids = floor.Construction.Grid;

    // 创建一个网格来表示房间布局
    int gridSize = 50;  // 假设网格大小为50x50
    cv::Mat layout = cv::Mat::zeros(1000, 1000, CV_8UC1);  // 假设最大尺寸为1000x1000

    // 根据网格信息绘制房间布局
    for (const auto& grid : grids) {
        cv::Point start(static_cast<int>(grid.StartPoint.x * gridSize), 
                        static_cast<int>(grid.StartPoint.y * gridSize));
        cv::Point end(static_cast<int>(grid.EndPoint.x * gridSize), 
                      static_cast<int>(grid.EndPoint.y * gridSize));
        cv::line(layout, start, end, cv::Scalar(255), 2);
    }

    // 找到房间的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(layout, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 对每个房间进行管线规划
    for (const auto& contour : contours) {
        cv::Rect roomRect = cv::boundingRect(contour);
        
        // 获取房间的功能类型（这里需要根据实际情况来判断）
        std::string roomFunction = "公区类";  // 默认为公区类
        
        // 根据房间功能获取管线间距
        double pipeSpan = 250.0;  // 默认间距
        for (const auto& elasticSpan : combinedData.inputData.WebData.ElasticSpanSet) {
            if (elasticSpan.FuncName == roomFunction) {
                pipeSpan = elasticSpan.PriorSpan;
                break;
            }
        }

        // 在房间内生成管线
        for (int y = roomRect.y; y < roomRect.y + roomRect.height; y += static_cast<int>(pipeSpan)) {
            pipePlan.emplace_back(roomRect.x, y);
            pipePlan.emplace_back(roomRect.x + roomRect.width, y);
            
            // 添加返回的管线（如果不是最后一行）
            if (y + pipeSpan < roomRect.y + roomRect.height) {
                pipePlan.emplace_back(roomRect.x + roomRect.width, y + static_cast<int>(pipeSpan/2));
                pipePlan.emplace_back(roomRect.x, y + static_cast<int>(pipeSpan/2));
            }
        }
    }

    return pipePlan;
}