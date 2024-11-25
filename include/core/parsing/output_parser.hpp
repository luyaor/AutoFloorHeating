#ifndef OUTPUT_PARSER_H
#define OUTPUT_PARSER_H

#include <string>
#include "types.hpp"

namespace iad {
    namespace output_parser {
        HeatingDesign generatePipePlan(const CombinedData& combinedData);
    }
}

#endif // OUTPUT_PARSER_H 