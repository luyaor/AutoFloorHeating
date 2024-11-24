#ifndef AR_DESIGN_PARSER_HPP
#define AR_DESIGN_PARSER_HPP

#include <json/json.h>
#include "types/ar_design_structures.hpp"

namespace iad {
namespace parsers {

class ARDesignParser {
private:
    static void parseCurveInfo(const Json::Value& curveJson, CurveInfo& curve);
    static void parseRoom(const Json::Value& roomJson, Room& room);
    static void parseJCW(const Json::Value& jcwJson, JCW& jcw);
    static void parseDoor(const Json::Value& doorJson, Door& door);
    static void parseHouseType(const Json::Value& houseTypeJson, HouseType& houseType);

public:
    static ARDesign parse(const std::string& jsonStr);
};

} // namespace parsers
} // namespace iad

#endif // AR_DESIGN_PARSER_HPP 