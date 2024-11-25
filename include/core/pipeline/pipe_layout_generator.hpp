#ifndef PIPE_LAYOUT_GENERATOR_H
#define PIPE_LAYOUT_GENERATOR_H

#include "types/heating_design_structures.hpp"
#include "types/ar_design_structures.hpp"
#include "types/input_data_structures.hpp"

CollectorCoil generatePipeLayout(const HouseType& houseType, const WebData& webData);

#endif // PIPE_LAYOUT_GENERATOR_H
