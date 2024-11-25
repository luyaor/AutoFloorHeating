#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <string>
#include "types/ar_design_structures.hpp"
#include "types/input_data_structures.hpp"
#include "types/heating_design_structures.hpp"

namespace iad {

struct CombinedData {
    ARDesign arDesign;
    InputData inputData;
};

// JSON parsing related functions
CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson);
HeatingDesign generatePipePlan(const CombinedData& combinedData);

}

#endif // JSON_PARSER_H 