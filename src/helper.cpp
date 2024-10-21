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

    // Parse ARDesign.json
    Json::Value arDesignRoot;
    Json::Reader arDesignReader;
    if (!arDesignReader.parse(arDesignJson, arDesignRoot)) {
        throw std::runtime_error("Failed to parse ARDesign.json");
    }

    // Parse Floors
    const Json::Value& floorsJson = arDesignRoot["Floors"];
    for (const auto& floorJson : floorsJson) {
        Floor floor;
        floor.Name = floorJson["Name"].asString();
        floor.Num = floorJson["Num"].asString();
        floor.LevelHeight = floorJson["LevelHeight"].asDouble();

        // Parse Construction
        const Json::Value& constructionJson = floorJson["Construction"];
        
        // Parse HouseTypes
        const Json::Value& houseTypesJson = constructionJson["HouseType"];
        for (const auto& houseTypeJson : houseTypesJson) {
            HouseType houseType;
            houseType.houseName = houseTypeJson["houseName"].asString();
            for (const auto& roomName : houseTypeJson["RoomNames"])
                houseType.RoomNames.push_back(roomName.asString());
            for (const auto& pointJson : houseTypeJson["Boundary"])
                houseType.Boundary.push_back(Point{pointJson["x"].asDouble(), pointJson["y"].asDouble(), pointJson["z"].asDouble()});
            floor.construction.houseTypes.push_back(houseType);
        }

        // Parse Rooms
        const Json::Value& roomsJson = constructionJson["Room"];
        for (const auto& roomJson : roomsJson) {
            Room room;
            room.Guid = roomJson["Guid"].asString();
            room.Name = roomJson["Name"].asString();
            room.NameType = roomJson["NameType"].asString();
            for (const auto& doorId : roomJson["DoorIds"])
                room.DoorIds.push_back(doorId.asString());
            for (const auto& jcwGuidName : roomJson["JCWGuidNames"])
                room.JCWGuidNames.push_back(jcwGuidName.asString());
            for (const auto& wallName : roomJson["WallNames"])
                room.WallNames.push_back(wallName.asString());
            for (const auto& pointJson : roomJson["Boundary"])
                room.Boundary.push_back(Point{pointJson["x"].asDouble(), pointJson["y"].asDouble(), pointJson["z"].asDouble()});
            room.IsRecreationalRoom = roomJson["IsRecreationalRoom"].asBool();
            floor.construction.rooms.push_back(room);
        }

        // Parse JCWs and Doors (similar to Rooms)
        // ...

        combinedData.arDesign.Floors.push_back(floor);
    }

    // Parse inputData.json
    Json::Value inputDataRoot;
    Json::Reader inputDataReader;
    if (!inputDataReader.parse(inputDataJson, inputDataRoot)) {
        throw std::runtime_error("Failed to parse inputData.json");
    }

    // Parse AssistData
    const Json::Value& assistDataJson = inputDataRoot["AssistData"];
    for (const auto& collectorJson : assistDataJson["AssistCollectors"]) {
        AssistCollector collector;
        collector.Id = collectorJson["Id"].asString();
        collector.Loc = Point{collectorJson["Loc"]["x"].asDouble(), collectorJson["Loc"]["y"].asDouble(), collectorJson["Loc"]["z"].asDouble()};
        collector.LevelName = collectorJson["LevelName"].asString();
        for (const auto& boundaryJson : collectorJson["Boundaries"]) {
            AssistCollector::Boundary boundary;
            boundary.Offset = boundaryJson["Offset"].asDouble();
            for (const auto& borderJson : boundaryJson["Borders"]) {
                Border border;
                border.StartPoint = Point{borderJson["StartPoint"]["x"].asDouble(), borderJson["StartPoint"]["y"].asDouble(), borderJson["StartPoint"]["z"].asDouble()};
                border.EndPoint = Point{borderJson["EndPoint"]["x"].asDouble(), borderJson["EndPoint"]["y"].asDouble(), borderJson["EndPoint"]["z"].asDouble()};
                border.ColorIndex = borderJson["ColorIndex"].asInt();
                border.CurveType = borderJson["CurveType"].asInt();
                boundary.Borders.push_back(border);
            }
            collector.Boundaries.push_back(boundary);
        }
        combinedData.inputData.assistData.AssistCollectors.push_back(collector);
    }

    // Parse WebData
    const Json::Value& webDataJson = inputDataRoot["WebData"];
    combinedData.inputData.webData.ImbalanceRatio = webDataJson["ImbalanceRatio"].asInt();
    combinedData.inputData.webData.JointPipeSpan = webDataJson["JointPipeSpan"].asDouble();
    combinedData.inputData.webData.DenseAreaWallSpan = webDataJson["DenseAreaWallSpan"].asDouble();
    combinedData.inputData.webData.DenseAreaSpanLess = webDataJson["DenseAreaSpanLess"].asDouble();

    // Parse LoopSpanSet, ObsSpanSet, DeliverySpanSet, PipeSpanSet, ElasticSpanSet, and FuncRooms
    // Parse LoopSpanSet
    const Json::Value& loopSpanSetJson = webDataJson["LoopSpanSet"];
    for (const auto& loopSpanJson : loopSpanSetJson) {
        LoopSpan loopSpan;
        loopSpan.TypeName = loopSpanJson["TypeName"].asString();
        loopSpan.MinSpan = loopSpanJson["MinSpan"].asDouble();
        loopSpan.MaxSpan = loopSpanJson["MaxSpan"].asDouble();
        loopSpan.Curvity = loopSpanJson["Curvity"].asDouble();
        combinedData.inputData.webData.LoopSpanSet.push_back(loopSpan);
    }

    // Parse ObsSpanSet
    const Json::Value& obsSpanSetJson = webDataJson["ObsSpanSet"];
    for (const auto& obsSpanJson : obsSpanSetJson) {
        ObstacleSpan obsSpan;
        obsSpan.ObsName = obsSpanJson["ObsName"].asString();
        obsSpan.MinSpan = obsSpanJson["MinSpan"].asDouble();
        obsSpan.MaxSpan = obsSpanJson["MaxSpan"].asDouble();
        combinedData.inputData.webData.ObsSpanSet.push_back(obsSpan);
    }

    // Parse DeliverySpanSet
    const Json::Value& deliverySpanSetJson = webDataJson["DeliverySpanSet"];
    for (const auto& deliverySpanJson : deliverySpanSetJson) {
        ObstacleSpan deliverySpan;
        deliverySpan.ObsName = deliverySpanJson["ObsName"].asString();
        deliverySpan.MinSpan = deliverySpanJson["MinSpan"].asDouble();
        deliverySpan.MaxSpan = deliverySpanJson["MaxSpan"].asDouble();
        combinedData.inputData.webData.DeliverySpanSet.push_back(deliverySpan);
    }

    // Parse PipeSpanSet
    const Json::Value& pipeSpanSetJson = webDataJson["PipeSpanSet"];
    for (const auto& pipeSpanJson : pipeSpanSetJson) {
        PipeSpanSet pipeSpan;
        pipeSpan.LevelDesc = pipeSpanJson["LevelDesc"].asString();
        pipeSpan.FuncName = pipeSpanJson["FuncName"].asString();
        for (const auto& direction : pipeSpanJson["Directions"])
            pipeSpan.Directions.push_back(direction.asString());
        pipeSpan.ExterWalls = pipeSpanJson["ExterWalls"].asInt();
        pipeSpan.PipeSpan = pipeSpanJson["PipeSpan"].asDouble();
        combinedData.inputData.webData.PipeSpanSet.push_back(pipeSpan);
    }

    // Parse ElasticSpanSet
    const Json::Value& elasticSpanSetJson = webDataJson["ElasticSpanSet"];
    for (const auto& elasticSpanJson : elasticSpanSetJson) {
        ElasticSpan elasticSpan;
        elasticSpan.FuncName = elasticSpanJson["FuncName"].asString();
        elasticSpan.PriorSpan = elasticSpanJson["PriorSpan"].asDouble();
        elasticSpan.MinSpan = elasticSpanJson["MinSpan"].asDouble();
        elasticSpan.MaxSpan = elasticSpanJson["MaxSpan"].asDouble();
        combinedData.inputData.webData.ElasticSpanSet.push_back(elasticSpan);
    }

    // Parse FuncRooms
    const Json::Value& funcRoomsJson = webDataJson["FuncRooms"];
    for (const auto& funcRoomJson : funcRoomsJson) {
        FuncRoom funcRoom;
        funcRoom.FuncName = funcRoomJson["FuncName"].asString();
        for (const auto& roomName : funcRoomJson["RoomNames"])
            funcRoom.RoomNames.push_back(roomName.asString());
        combinedData.inputData.webData.FuncRooms.push_back(funcRoom);
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