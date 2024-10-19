file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/tests/*.cpp)
add_executable(tests ${TEST_SOURCES})

# Find GTest package
find_package(GTest REQUIRED)

# Include GTest headers
include_directories(${GTEST_INCLUDE_DIRS})

# Link against GTest libraries
target_link_libraries(tests ${GTEST_BOTH_LIBRARIES})

# Link against pthread (required by GTest)
find_package(Threads REQUIRED)
target_link_libraries(tests Threads::Threads)
