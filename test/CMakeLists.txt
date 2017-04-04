cmake_minimum_required(VERSION 2.8)

find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)

add_compile_options(-std=c++11)
add_definitions (-DBOOST_TEST_DYN_LINK)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils
    tools armadillo)

add_executable(test_optimization test_optimization.cpp)
target_link_libraries(test_optimization
    tools armadillo)

add_test(NAME _test_utils COMMAND test_utils)
add_test(NAME _test_optimization COMMAND test_optimization)
