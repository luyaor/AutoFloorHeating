#ifndef INPUT_DATA_PARSER_HPP
#define INPUT_DATA_PARSER_HPP

#include <json/json.h>
#include "types/input_data_structures.hpp"

namespace iad {
namespace parsers {

class InputDataParser {
private:
    static void parseAssistData(const Json::Value& json, AssistData& data);
    static void parseWebData(const Json::Value& json, WebData& data);
    static void parseAssistCollector(const Json::Value& json, AssistCollector& collector);
    static void parseLoopSpanSet(const Json::Value& json, std::vector<LoopSpan>& spans);
    static void parseObstacleSpans(const Json::Value& json, std::vector<ObstacleSpan>& spans);
    static void parsePipeSpanSet(const Json::Value& json, std::vector<PipeSpanSet>& spans);
    static void parseElasticSpanSet(const Json::Value& json, std::vector<ElasticSpan>& spans);
    static void parseFuncRooms(const Json::Value& json, std::vector<FuncRoom>& rooms);

public:
    static InputData parse(const std::string& jsonStr);
};

} // namespace parsers
} // namespace iad

#endif // INPUT_DATA_PARSER_HPP 