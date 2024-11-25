#include "json_parser.hpp"
#include "visualization.hpp"
#include <json/json.h>
#include "pipe_layout_generator.hpp"
#include "parsers/ar_design_parser.hpp"
#include "parsers/input_data_parser.hpp"

namespace iad {

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

HeatingDesign generatePipePlan(const CombinedData& combinedData) {
    HeatingDesign heatingDesign;

    // Iterate through each floor
    for (const auto& floor : combinedData.arDesign.Floor) {
        HeatingCoil heatingCoil;
        heatingCoil.LevelName = floor.Name;
        heatingCoil.LevelNo = std::stoi(floor.Num);
        heatingCoil.LevelDesc = "Floor " + floor.Num;

        // Iterate through each house type
        for (const auto& houseType : floor.construction.houseTypes) {
            heatingCoil.HouseName = houseType.houseName;

            // Call the pipe layout generation function
            CollectorCoil collectorCoil = generatePipeLayout(houseType, combinedData.inputData.webData);

            // Add the generated CollectorCoil to HeatingCoil
            heatingCoil.CollectorCoils.push_back(collectorCoil);
        }

        // Add the generated HeatingCoil to HeatingDesign
        heatingDesign.HeatingCoils.push_back(heatingCoil);
    }

    return heatingDesign;
}

} 