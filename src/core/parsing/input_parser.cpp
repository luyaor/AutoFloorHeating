#include <json/json.h>
#include "core/parsing/input_parser.hpp"
#include "visualization/visualization.hpp"
#include "core/parsing/parser/ar_design_parser.hpp"
#include "core/parsing/parser/input_data_parser.hpp"

namespace iad {
    namespace input_parser {

        CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson) {
            CombinedData combinedData;
    
            try {
                combinedData.arDesign = parsers::ARDesignParser::parse(arDesignJson);
                printARDesign(combinedData.arDesign, std::cout);
                combinedData.inputData = parsers::InputDataParser::parse(inputDataJson);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Error parsing JSON: ") + e.what());
            }

            return combinedData;
        }

    }
} 