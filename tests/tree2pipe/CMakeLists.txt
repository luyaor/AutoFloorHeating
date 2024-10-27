cmake_minimum_required(VERSION 3.15)
project(tree2pipe)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")


add_executable(${PROJECT_NAME} tree2pipe.cpp)

find_package(Eigen3 REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})

find_package(Matplot++ REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Matplot++_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Matplot++::matplot)
