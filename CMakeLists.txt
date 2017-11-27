project(2dFluid)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -ffast-math -fopenmp -march=native -std=c++14  " )

find_package(OpenGL)
find_package(SDL2)

link_directories(/usr/local/lib/)



aux_source_directory(. SRC_LIST)
#add_executable(${PROJECT_NAME} ${SRC_LIST})
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} SDL2 GLEW ${OPENGL_LIBRARIES})
