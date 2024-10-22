#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <string>
#include "data/ar_design_structures.hpp"
#include "data/input_data_structures.hpp"
#include "data/heating_design_structures.hpp"

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
