#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <string>
#include "types.hpp"

namespace iad {
    namespace input_parser {
        CombinedData parseJsonData(const std::string& arDesignJson, const std::string& inputDataJson);
    }
}

#endif // INPUT_PARSER_H 