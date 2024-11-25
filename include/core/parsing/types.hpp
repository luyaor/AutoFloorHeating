#ifndef TYPES_H
#define TYPES_H

#include "types/ar_design_structures.hpp"   
#include "types/input_data_structures.hpp"
#include "types/heating_design_structures.hpp"


namespace iad {

struct CombinedData {
    ARDesign arDesign;
    InputData inputData;
};

}   

#endif // TYPES_H