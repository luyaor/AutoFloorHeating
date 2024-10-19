cmake_minimum_required(VERSION 3.29)
project(iad)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



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


# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES}
        src/helper.cpp
        include/helper.h)

# Test target
enable_testing()
add_subdirectory(tests)

# Install rules
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Link OpenCV and JsonCpp
target_link_libraries(iad ${OpenCV_LIBS} ${JSONCPP_LIBRARY})

# Print JsonCpp information for debugging
message(STATUS "JSONCPP_LIBRARY: ${JSONCPP_LIBRARY}")
message(STATUS "JSONCPP_INCLUDE_DIRS: ${JSONCPP_INCLUDE_DIRS}")
