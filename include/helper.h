#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <json/json.h>
#include "ar_design_structures.h"
#include "input_data_structures.h"
#include "heating_design_structures.h"

// Combined structure for both JSON files
struct CombinedData {
    ARDesign arDesign;
    InputData inputData;
};

// Function declarations
CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson);
HeatingDesign generatePipePlan(const CombinedData& combinedData);
std::string planToJson(const HeatingDesign& plan);

#endif // HELPER_H
