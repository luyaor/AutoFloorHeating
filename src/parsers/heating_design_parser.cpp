#include "parsers/heating_design_parser.hpp"

namespace iad {
namespace parsers {

Json::Value HeatingDesignParser::pointToJson(const Point& point) {
    Json::Value json;
    json["x"] = point.x;
    json["y"] = point.y;
    json["z"] = point.z;
    return json;
}

Json::Value HeatingDesignParser::jLineToJson(const JLine& line) {
    Json::Value json;
    json["StartPoint"] = pointToJson(line.StartPoint);
    json["EndPoint"] = pointToJson(line.EndPoint);
    json["ColorIndex"] = line.ColorIndex;
    json["CurveType"] = line.CurveType;
    return json;
}

Json::Value HeatingDesignParser::coilAreaToJson(const CoilArea& area) {
    Json::Value json;
    json["AreaName"] = area.AreaName;
    return json;
}

Json::Value HeatingDesignParser::coilLoopToJson(const CoilLoop& loop) {
    Json::Value json;
    json["Length"] = loop.Length;
    json["Curvity"] = loop.Curvity;

    Json::Value areasJson(Json::arrayValue);
    for (const auto& area : loop.Areas) {
        areasJson.append(coilAreaToJson(area));
    }
    json["Areas"] = areasJson;

    Json::Value pathJson(Json::arrayValue);
    for (const auto& line : loop.Path) {
        pathJson.append(jLineToJson(line));
    }
    json["Path"] = pathJson;

    return json;
}

Json::Value HeatingDesignParser::collectorCoilToJson(const CollectorCoil& collector) {
    Json::Value json;
    json["CollectorName"] = collector.CollectorName;
    json["Loops"] = collector.Loops;

    Json::Value loopsJson(Json::arrayValue);
    for (const auto& loop : collector.CoilLoops) {
        loopsJson.append(coilLoopToJson(loop));
    }
    json["CoilLoops"] = loopsJson;

    Json::Value deliverysJson(Json::arrayValue);
    for (const auto& delivery : collector.Deliverys) {
        Json::Value deliveryJson(Json::arrayValue);
        for (const auto& line : delivery) {
            deliveryJson.append(jLineToJson(line));
        }
        deliverysJson.append(deliveryJson);
    }
    json["Deliverys"] = deliverysJson;

    return json;
}

Json::Value HeatingDesignParser::heatingCoilToJson(const HeatingCoil& coil) {
    Json::Value json;
    json["LevelName"] = coil.LevelName;
    json["LevelNo"] = coil.LevelNo;
    json["LevelDesc"] = coil.LevelDesc;
    json["HouseName"] = coil.HouseName;

    Json::Value expansionsJson(Json::arrayValue);
    for (const auto& line : coil.Expansions) {
        expansionsJson.append(jLineToJson(line));
    }
    json["Expansions"] = expansionsJson;

    Json::Value collectorsJson(Json::arrayValue);
    for (const auto& collector : coil.CollectorCoils) {
        collectorsJson.append(collectorCoilToJson(collector));
    }
    json["CollectorCoils"] = collectorsJson;

    return json;
}

std::string HeatingDesignParser::toJson(const HeatingDesign& design) {
    Json::Value root;
    Json::Value heatingCoilsJson(Json::arrayValue);
    
    for (const auto& coil : design.HeatingCoils) {
        heatingCoilsJson.append(heatingCoilToJson(coil));
    }
    root["HeatingCoils"] = heatingCoilsJson;
    
    Json::FastWriter writer;
    return writer.write(root);
}

} // namespace parsers
} // namespace iad 