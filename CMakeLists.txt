cmake_minimum_required(VERSION 3.8)
project(context_manager LANGUAGES CXX CUDA)

add_library(context_manager SHARED context_manager.cu)
target_link_libraries(context_manager -lcublas)
