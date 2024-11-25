#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <ostream>
#include "types/ar_design_structures.hpp"

namespace iad {

// Visualization related functions
void printARDesign(const ARDesign& design, std::ostream& out = std::cout);
void drawARDesign(const ARDesign& design, const std::string& outputPath);

}

#endif // VISUALIZATION_H 