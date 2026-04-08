cmake_minimum_required(VERSION 3.18)
project(docscanner_cpp LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(docscanner_cpp cpp/docscanner.cpp)
target_link_libraries(docscanner_cpp PRIVATE ${OpenCV_LIBS})
target_include_directories(docscanner_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_features(docscanner_cpp PRIVATE cxx_std_17)
