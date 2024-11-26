#include <json/json.h>
#include "core/parsing/output_parser.hpp"
#include "visualization/visualization.hpp"
#include "core/parsing/parser/ar_design_parser.hpp"
#include "core/parsing/parser/input_data_parser.hpp"
#include "core/pipeline/pipe_layout_generator.hpp"

namespace iad {
    namespace output_parser {

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
} 