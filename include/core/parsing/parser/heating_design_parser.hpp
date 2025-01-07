#ifndef HEATING_DESIGN_PARSER_HPP
#define HEATING_DESIGN_PARSER_HPP

#include <json/json.h>
#include "types/heating_design_structures.hpp"
#include "types/data_structures.hpp"

namespace iad {
namespace parsers {

class HeatingDesignParser {
private:
    static Json::Value pointToJson(const Point& point);
    static Json::Value curveInfoToJson(const CurveInfo& curve);
    static Json::Value coilAreaToJson(const CoilArea& area);
    static Json::Value coilLoopToJson(const CoilLoop& loop);
    static Json::Value collectorCoilToJson(const CollectorCoil& collector);
    static Json::Value heatingCoilToJson(const HeatingCoil& coil);

    static Point pointFromJson(const Json::Value& json);
    static CurveInfo curveInfoFromJson(const Json::Value& json);
    static CoilArea coilAreaFromJson(const Json::Value& json);
    static CoilLoop coilLoopFromJson(const Json::Value& json);
    static CollectorCoil collectorCoilFromJson(const Json::Value& json);
    static HeatingCoil heatingCoilFromJson(const Json::Value& json);

public:
    static std::string toJson(const HeatingDesign& design);
    static HeatingDesign fromJson(const Json::Value& json);
};

} // namespace parsers
} // namespace iad

#endif // HEATING_DESIGN_PARSER_HPP 