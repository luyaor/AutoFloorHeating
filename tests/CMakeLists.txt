file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/*.cpp)
add_executable(tests ${TEST_SOURCES})

# Find GTest package
find_package(GTest REQUIRED)

# Include GTest headers
include_directories(${GTEST_INCLUDE_DIRS})

# Find JsonCpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

# Set ENABLE_EXPORTS property on the iad target
set_target_properties(iad PROPERTIES ENABLE_EXPORTS ON)

find_library(JSONCPP_LIBRARY
    NAMES jsoncpp
    PATHS /opt/homebrew/Cellar/jsoncpp/1.9.6/lib
    NO_DEFAULT_PATH
)

if(NOT JSONCPP_LIBRARY)
    message(FATAL_ERROR "JsonCpp library not found")
endif()

# Link against GTest libraries, JsonCpp, and the main project target
target_link_libraries(tests 
    ${GTEST_BOTH_LIBRARIES}
    ${JSONCPP_LIBRARY}
    iad_lib
)

# Explicitly add JsonCpp library
target_link_libraries(tests /opt/homebrew/Cellar/jsoncpp/1.9.6/lib/libjsoncpp.dylib)

# Add JsonCpp library directories to the linker search path
link_directories(/opt/homebrew/Cellar/jsoncpp/1.9.6/lib)

# Link against pthread (required by GTest)
find_package(Threads REQUIRED)
target_link_libraries(tests Threads::Threads)

# Include the project's include directory
target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include)

# 链接必要的库
target_link_libraries(tests
    GTest::GTest
    GTest::Main
    ${PROJECT_NAME}_lib
)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# 添加测试
gtest_discover_tests(tests)