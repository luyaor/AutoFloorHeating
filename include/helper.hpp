#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <string>
#include "types/ar_design_structures.hpp"
#include "types/input_data_structures.hpp"
#include "types/heating_design_structures.hpp"
#include <ostream>

namespace iad {


// Combined structure for both JSON files
struct CombinedData {
    ARDesign arDesign;
    InputData inputData;
};

// Function declarations
CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson);
HeatingDesign generatePipePlan(const CombinedData& combinedData);
void printARDesign(const ARDesign& design, std::ostream& out = std::cout);
void drawARDesign(const ARDesign& design, const std::string& outputPath);

}

#endif // HELPER_H
