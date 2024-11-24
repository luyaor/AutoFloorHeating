#include "parsers/input_data_parser.hpp"

namespace iad {
namespace parsers {

void InputDataParser::parseAssistCollector(const Json::Value& json, AssistCollector& collector) {
    collector.Id = json["Id"].asString();
    collector.LevelName = json["LevelName"].asString();
    
    const auto& locJson = json["Loc"];
    collector.Loc = Point{
        locJson["x"].asDouble(),
        locJson["y"].asDouble(),
        locJson["z"].asDouble()
    };

    for (const auto& boundaryJson : json["Boundaries"]) {
        AssistCollector::Boundary boundary;
        boundary.Offset = boundaryJson["Offset"].asDouble();
        
        for (const auto& borderJson : boundaryJson["Borders"]) {
            Border border;
            const auto& startJson = borderJson["StartPoint"];
            const auto& endJson = borderJson["EndPoint"];
            
            border.StartPoint = Point{
                startJson["x"].asDouble(),
                startJson["y"].asDouble(),
                startJson["z"].asDouble()
            };
            border.EndPoint = Point{
                endJson["x"].asDouble(),
                endJson["y"].asDouble(),
                endJson["z"].asDouble()
            };
            border.ColorIndex = borderJson["ColorIndex"].asInt();
            border.CurveType = borderJson["CurveType"].asInt();
            
            boundary.Borders.push_back(border);
        }
        collector.Boundaries.push_back(boundary);
    }
}

void InputDataParser::parseAssistData(const Json::Value& json, AssistData& data) {
    for (const auto& collectorJson : json["AssistCollectors"]) {
        AssistCollector collector;
        parseAssistCollector(collectorJson, collector);
        data.AssistCollectors.push_back(collector);
    }
}

void InputDataParser::parseLoopSpanSet(const Json::Value& json, std::vector<LoopSpan>& spans) {
    for (const auto& spanJson : json) {
        LoopSpan span;
        span.TypeName = spanJson["TypeName"].asString();
        span.MinSpan = spanJson["MinSpan"].asDouble();
        span.MaxSpan = spanJson["MaxSpan"].asDouble();
        span.Curvity = spanJson["Curvity"].asDouble();
        spans.push_back(span);
    }
}

void InputDataParser::parseObstacleSpans(const Json::Value& json, std::vector<ObstacleSpan>& spans) {
    for (const auto& spanJson : json) {
        ObstacleSpan span;
        span.ObsName = spanJson["ObsName"].asString();
        span.MinSpan = spanJson["MinSpan"].asDouble();
        span.MaxSpan = spanJson["MaxSpan"].asDouble();
        spans.push_back(span);
    }
}

void InputDataParser::parsePipeSpanSet(const Json::Value& json, std::vector<PipeSpanSet>& spans) {
    for (const auto& spanJson : json) {
        PipeSpanSet span;
        span.LevelDesc = spanJson["LevelDesc"].asString();
        span.FuncName = spanJson["FuncName"].asString();
        span.ExterWalls = spanJson["ExterWalls"].asInt();
        span.PipeSpan = spanJson["PipeSpan"].asDouble();
        
        for (const auto& direction : spanJson["Directions"]) {
            span.Directions.push_back(direction.asString());
        }
        spans.push_back(span);
    }
}

void InputDataParser::parseElasticSpanSet(const Json::Value& json, std::vector<ElasticSpan>& spans) {
    for (const auto& spanJson : json) {
        ElasticSpan span;
        span.FuncName = spanJson["FuncName"].asString();
        span.PriorSpan = spanJson["PriorSpan"].asDouble();
        span.MinSpan = spanJson["MinSpan"].asDouble();
        span.MaxSpan = spanJson["MaxSpan"].asDouble();
        spans.push_back(span);
    }
}

void InputDataParser::parseFuncRooms(const Json::Value& json, std::vector<FuncRoom>& rooms) {
    for (const auto& roomJson : json) {
        FuncRoom room;
        room.FuncName = roomJson["FuncName"].asString();
        for (const auto& name : roomJson["RoomNames"]) {
            room.RoomNames.push_back(name.asString());
        }
        rooms.push_back(room);
    }
}

void InputDataParser::parseWebData(const Json::Value& json, WebData& data) {
    data.ImbalanceRatio = json["ImbalanceRatio"].asInt();
    data.JointPipeSpan = json["JointPipeSpan"].asDouble();
    data.DenseAreaWallSpan = json["DenseAreaWallSpan"].asDouble();
    data.DenseAreaSpanLess = json["DenseAreaSpanLess"].asDouble();

    parseLoopSpanSet(json["LoopSpanSet"], data.LoopSpanSet);
    parseObstacleSpans(json["ObsSpanSet"], data.ObsSpanSet);
    parseObstacleSpans(json["DeliverySpanSet"], data.DeliverySpanSet);
    parsePipeSpanSet(json["PipeSpanSet"], data.PipeSpanSet);
    parseElasticSpanSet(json["ElasticSpanSet"], data.ElasticSpanSet);
    parseFuncRooms(json["FuncRooms"], data.FuncRooms);
}

InputData InputDataParser::parse(const std::string& jsonStr) {
    InputData data;
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(jsonStr, root)) {
        throw std::runtime_error("Failed to parse Input Data JSON");
    }
    
    parseAssistData(root["AssistData"], data.assistData);
    parseWebData(root["WebData"], data.webData);
    
    return data;
}

} // namespace parsers
} // namespace iad 