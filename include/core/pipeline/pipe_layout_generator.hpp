#ifndef PIPE_LAYOUT_GENERATOR_H
#define PIPE_LAYOUT_GENERATOR_H

#include "core/parsing/json_parser.hpp"

CollectorCoil generatePipeLayout(const HouseType& houseType, const WebData& webData);

#endif // PIPE_LAYOUT_GENERATOR_H
