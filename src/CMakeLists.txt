cmake_minimum_required(VERSION 2.8)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
add_compile_options(-std=c++11 -O3)

add_library(tools utils.cpp optimization.cpp)

add_executable(tsne tsne.cpp)
target_link_libraries(tsne tools ${Boost_LIBRARIES} armadillo)
