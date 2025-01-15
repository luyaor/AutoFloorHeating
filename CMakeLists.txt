# CMakeLists.txt

cmake_minimum_required(VERSION 3.28)
project(iad)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.10.0_12/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find JsonCpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

find_library(JSONCPP_LIBRARY
    NAMES jsoncpp
    PATHS /opt/homebrew/Cellar/jsoncpp/1.9.6/lib/
    NO_DEFAULT_PATH
)
if(NOT JSONCPP_LIBRARY)
    message(FATAL_ERROR "jsoncpp library not found.")
endif()

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Find Eigen3
set(EIGEN3_ROOT "/opt/homebrew/Cellar/eigen/3.4.0_1")
set(EIGEN3_INCLUDE_DIR "${EIGEN3_ROOT}/include/eigen3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EIGEN3_ROOT}/share/cmake/Modules")

find_package(Eigen3 3.4 REQUIRED)
include_directories(BEFORE ${EIGEN3_INCLUDE_DIR})

# Find Matplotplusplus
find_package(Matplot++ REQUIRED)
include_directories(${Matplot++_INCLUDE_DIRS})

# Find OpenCV freetype
find_package(OpenCV REQUIRED freetype)

# Find libdxfrw
find_path(DXFRW_INCLUDE_DIR
    NAMES libdxfrw/libdxfrw.h
    PATHS
        /opt/homebrew/include
        /usr/local/include
        /usr/include
)

find_library(DXFRW_LIBRARY
    NAMES dxfrw
    PATHS
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib
)

if(NOT DXFRW_INCLUDE_DIR OR NOT DXFRW_LIBRARY)
    message(FATAL_ERROR "libdxfrw not found. Please install libdxfrw.")
endif()

include_directories(${DXFRW_INCLUDE_DIR})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE SOURCES 
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Create a library target first
add_library(${PROJECT_NAME}_lib STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME}_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME}_lib 
    ${OpenCV_LIBS} 
    ${JSONCPP_LIBRARY} 
    ${DXFRW_LIBRARY}
    Matplot++::matplot
    Eigen3::Eigen
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

enable_testing()
add_subdirectory(tests)
include(GoogleTest)
set(GTEST_DISCOVERY_TIMEOUT 60)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")