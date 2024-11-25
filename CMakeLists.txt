cmake_minimum_required(VERSION 3.28)
project(iad)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

# Find OpenCV
set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.10.0_12/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find JsonCpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

# Explicitly find jsoncpp library
find_library(JSONCPP_LIBRARY
    NAMES jsoncpp
    PATHS /opt/homebrew/Cellar/jsoncpp/1.9.6/lib/  # Adjust this path if necessary
    NO_DEFAULT_PATH
)

if(NOT JSONCPP_LIBRARY)
    message(FATAL_ERROR "jsoncpp library not found. Path searched: /opt/homebrew/lib")
endif()


# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find Matplotplusplus
find_package(Matplot++ REQUIRED)
include_directories(${Matplot++_INCLUDE_DIRS})

# Find OpenCV freetype
find_package(OpenCV REQUIRED freetype)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES}
        include/json_parser.hpp
        src/json_parser.cpp
        include/visualization.hpp
        src/visualization.cpp
        include/pipe_layout_generator.hpp
        src/pipe_layout_generator.cpp
        include/parsers/ar_design_parser.hpp
        src/parsers/ar_design_parser.cpp
        include/parsers/input_data_parser.hpp
        src/parsers/input_data_parser.cpp
        include/parsers/heating_design_parser.hpp
        src/parsers/heating_design_parser.cpp
        include/types/ar_design_structures.hpp
        include/types/input_data_structures.hpp
        include/types/heating_design_structures.hpp)

# Create a library target for the main project code
add_library(${PROJECT_NAME}_lib STATIC ${SOURCES})

# Link OpenCV and JsonCpp to the library
target_link_libraries(${PROJECT_NAME}_lib ${OpenCV_LIBS} ${JSONCPP_LIBRARY} Matplot++::matplot)

# Link the library to the main executable
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

# Test target
enable_testing()
add_subdirectory(tests)
include(GoogleTest)

# Add timeout setting for test discovery
set(GTEST_DISCOVERY_TIMEOUT 60) # Increase timeout to 60 seconds (default is 5)

# Install rules
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Print JsonCpp information for debugging
message(STATUS "JSONCPP_LIBRARY: ${JSONCPP_LIBRARY}")
message(STATUS "JSONCPP_INCLUDE_DIRS: ${JSONCPP_INCLUDE_DIRS}")
