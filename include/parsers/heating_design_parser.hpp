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
    static Json::Value jLineToJson(const JLine& line);
    static Json::Value coilAreaToJson(const CoilArea& area);
    static Json::Value coilLoopToJson(const CoilLoop& loop);
    static Json::Value collectorCoilToJson(const CollectorCoil& collector);
    static Json::Value heatingCoilToJson(const HeatingCoil& coil);

public:
    static std::string toJson(const HeatingDesign& design);
};

} // namespace parsers
} // namespace iad

#endif // HEATING_DESIGN_PARSER_HPP 