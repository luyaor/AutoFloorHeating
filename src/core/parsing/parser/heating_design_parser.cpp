#include "core/parsing/parser/heating_design_parser.hpp"

namespace iad {
namespace parsers {

Json::Value HeatingDesignParser::pointToJson(const Point& point) {
    Json::Value json;
    json["x"] = point.x;
    json["y"] = point.y;
    json["z"] = point.z;
    return json;
}

Json::Value HeatingDesignParser::curveInfoToJson(const CurveInfo& curve) {
    Json::Value json;
    json["StartPoint"] = pointToJson(curve.StartPoint);
    json["EndPoint"] = pointToJson(curve.EndPoint);
    json["ColorIndex"] = curve.ColorIndex;
    json["CurveType"] = curve.CurveType;
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
    for (const auto& curve : loop.Path) {
        pathJson.append(curveInfoToJson(curve));
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
        for (const auto& curve : delivery) {
            deliveryJson.append(curveInfoToJson(curve));
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
    for (const auto& curve : coil.Expansions) {
        expansionsJson.append(curveInfoToJson(curve));
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

Point HeatingDesignParser::pointFromJson(const Json::Value& json) {
    Point point;
    point.x = json["x"].asDouble();
    point.y = json["y"].asDouble();
    point.z = json["z"].asDouble();
    return point;
}

CurveInfo HeatingDesignParser::curveInfoFromJson(const Json::Value& json) {
    CurveInfo curve;
    curve.StartPoint = pointFromJson(json["StartPoint"]);
    curve.EndPoint = pointFromJson(json["EndPoint"]);
    curve.CurveType = json["CurveType"].asInt();
    
    if (curve.CurveType == 1) { // 圆弧
        curve.Center = pointFromJson(json["Center"]);
        curve.Radius = json["Radius"].asDouble();
        curve.StartAngle = json["StartAngle"].asDouble();
        curve.EndAngle = json["EndAngle"].asDouble();
    }
    
    return curve;
}

CoilArea HeatingDesignParser::coilAreaFromJson(const Json::Value& json) {
    CoilArea area;
    area.AreaName = json["AreaName"].asString();
    return area;
}

CoilLoop HeatingDesignParser::coilLoopFromJson(const Json::Value& json) {
    CoilLoop loop;
    loop.Length = json["Length"].asDouble();
    loop.Curvity = json["Curvity"].asInt();
    
    // 解析区域
    const Json::Value& areas = json["Areas"];
    for (const auto& area : areas) {
        loop.Areas.push_back(coilAreaFromJson(area));
    }
    
    // 解析路径
    const Json::Value& path = json["Path"];
    for (const auto& curve : path) {
        loop.Path.push_back(curveInfoFromJson(curve));
    }
    
    return loop;
}

CollectorCoil HeatingDesignParser::collectorCoilFromJson(const Json::Value& json) {
    CollectorCoil collector;
    collector.CollectorName = json["CollectorName"].asString();
    collector.Loops = json["Loops"].asInt();
    
    // 解析盘管回路
    const Json::Value& coilLoops = json["CoilLoops"];
    for (const auto& loop : coilLoops) {
        collector.CoilLoops.push_back(coilLoopFromJson(loop));
    }
    
    return collector;
}

HeatingCoil HeatingDesignParser::heatingCoilFromJson(const Json::Value& json) {
    HeatingCoil coil;
    coil.LevelName = json["LevelName"].asString();
    coil.LevelNo = json["LevelNo"].asInt();
    coil.LevelDesc = json["LevelDesc"].asString();
    coil.HouseName = json["HouseName"].asString();
    
    // 解析伸缩缝
    const Json::Value& expansions = json["Expansions"];
    for (const auto& expansion : expansions) {
        coil.Expansions.push_back(curveInfoFromJson(expansion));
    }
    
    // 解析分集水器回路
    const Json::Value& collectors = json["CollectorCoils"];
    for (const auto& collector : collectors) {
        coil.CollectorCoils.push_back(collectorCoilFromJson(collector));
    }
    
    return coil;
}

HeatingDesign HeatingDesignParser::fromJson(const Json::Value& json) {
    HeatingDesign design;
    const Json::Value& coils = json["HeatingCoils"];
    for (const auto& coil : coils) {
        design.HeatingCoils.push_back(heatingCoilFromJson(coil));
    }
    return design;
}

} // namespace parsers
} // namespace iad