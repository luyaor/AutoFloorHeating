cmake_minimum_required(VERSION 3.29)
project(iad)

set(CMAKE_CXX_STANDARD 20)

# Find OpenCV
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

add_executable(iad main.cpp)

# Link OpenCV and JsonCpp
target_link_libraries(iad ${OpenCV_LIBS} ${JSONCPP_LIBRARY})

# Print JsonCpp information for debugging
message(STATUS "JSONCPP_LIBRARY: ${JSONCPP_LIBRARY}")
message(STATUS "JSONCPP_INCLUDE_DIRS: ${JSONCPP_INCLUDE_DIRS}")
