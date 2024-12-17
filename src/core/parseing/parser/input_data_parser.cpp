#include "core/parsing/parser/input_data_parser.hpp"

namespace iad {
namespace parsers {

void InputDataParser::parseAssistCollector(const Json::Value& json, AssistCollector& collector) {
    // Parse Location
    const auto& locJson = json["Location"];
    collector.Location = Point{
        locJson["x"].asDouble(),
        locJson["y"].asDouble(),
        locJson["z"].asDouble()
    };
    
    // Parse Borders
    if (json.isMember("Borders") && json["Borders"].isArray()) {
        for (const auto& borderJson : json["Borders"]) {
            CurveInfo curve;
            const auto& startJson = borderJson["StartPoint"];
            const auto& endJson = borderJson["EndPoint"];
            
            curve.StartPoint = Point{
                startJson["x"].asDouble(),
                startJson["y"].asDouble(),
                startJson["z"].asDouble()
            };
            curve.EndPoint = Point{
                endJson["x"].asDouble(),
                endJson["y"].asDouble(),
                endJson["z"].asDouble()
            };
            
            if (borderJson.isMember("MidPoint")) {
                const auto& midJson = borderJson["MidPoint"];
                curve.MidPoint = Point{
                    midJson["x"].asDouble(),
                    midJson["y"].asDouble(),
                    midJson["z"].asDouble()
                };
            }
            
            curve.ColorIndex = borderJson["ColorIndex"].asInt();
            curve.CurveType = borderJson["CurveType"].asInt();
            
            collector.Borders.push_back(curve);
        }
    }
}

void InputDataParser::parseAssistData(const Json::Value& json, AssistData& data) {
    if (json.isMember("Floor") && json["Floor"].isArray()) {
        for (const auto& floorJson : json["Floor"]) {
            AssistFloor floor;
            floor.Name = floorJson["Name"].asString();
            floor.Num = floorJson["Num"].asString();
            
            const auto& basePointJson = floorJson["BasePoint"];
            floor.BasePoint = Point{
                basePointJson["x"].asDouble(),
                basePointJson["y"].asDouble(),
                basePointJson["z"].asDouble()
            };
            
            floor.LevelHeight = floorJson["LevelHeight"].asDouble();
            floor.LevelElevation = floorJson["LevelElevation"].asDouble();
            
            // Parse Construction
            if (floorJson.isMember("Construction")) {
                const auto& constructionJson = floorJson["Construction"];
                if (constructionJson.isMember("AssistCollector") && 
                    constructionJson["AssistCollector"].isArray()) {
                    for (const auto& collectorJson : constructionJson["AssistCollector"]) {
                        AssistCollector collector;
                        parseAssistCollector(collectorJson, collector);
                        floor.Construction.AssistCollector.push_back(collector);
                    }
                }
            }
            
            data.Floor.push_back(floor);
        }
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
        
        if (spanJson.isMember("Directions") && spanJson["Directions"].isArray()) {
            for (const auto& direction : spanJson["Directions"]) {
                span.Directions.push_back(direction.asString());
            }
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
        if (roomJson.isMember("RoomNames") && roomJson["RoomNames"].isArray()) {
            for (const auto& name : roomJson["RoomNames"]) {
                room.RoomNames.push_back(name.asString());
            }
        }
        rooms.push_back(room);
    }
}

void InputDataParser::parseWebData(const Json::Value& json, WebData& data) {
    data.ImbalanceRatio = json["ImbalanceRatio"].asInt();
    data.JointPipeSpan = json["JointPipeSpan"].asDouble();
    data.DenseAreaWallSpan = json["DenseAreaWallSpan"].asDouble();
    data.DenseAreaSpanLess = json["DenseAreaSpanLess"].asDouble();

    if (json.isMember("LoopSpanSet")) parseLoopSpanSet(json["LoopSpanSet"], data.LoopSpanSet);
    if (json.isMember("ObsSpanSet")) parseObstacleSpans(json["ObsSpanSet"], data.ObsSpanSet);
    if (json.isMember("DeliverySpanSet")) parseObstacleSpans(json["DeliverySpanSet"], data.DeliverySpanSet);
    if (json.isMember("PipeSpanSet")) parsePipeSpanSet(json["PipeSpanSet"], data.PipeSpanSet);
    if (json.isMember("ElasticSpanSet")) parseElasticSpanSet(json["ElasticSpanSet"], data.ElasticSpanSet);
    if (json.isMember("FuncRooms")) parseFuncRooms(json["FuncRooms"], data.FuncRooms);
}

InputData InputDataParser::parse(const std::string& jsonStr) {
    InputData data;
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(jsonStr, root)) {
        throw std::runtime_error("Failed to parse Input Data JSON");
    }
    
    if (root.isMember("AssistData")) parseAssistData(root["AssistData"], data.AssistData);
    if (root.isMember("WebData")) parseWebData(root["WebData"], data.WebData);
    
    return data;
}

} // namespace parsers
} // namespace iad