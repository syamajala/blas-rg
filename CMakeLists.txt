cmake_minimum_required(VERSION 3.8)
project(blas-rg LANGUAGES CXX)

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(BLAS REQUIRED)

find_path(BLAS_INCLUDE_DIRS cblas.h
  /usr/include
  /usr/local/include
  $ENV{BLAS_HOME}/include)

set(RegentFlags --blas-header ${BLAS_INCLUDE_DIRS} --blas-library ${BLAS_LIBRARIES} --output-dir ${PROJECT_BINARY_DIR} -fseparate 1)

add_custom_target(blas_tasks ALL
  DEPENDS blas_tasks.so)
set(OUTPUTS blas_tasks.so blas_tasks.h cblas_tasks.so cblas_tasks.h)
set(SRCS blas.rg blas_tasks.rg cblas.rg utils.rg)

option(USE_CUDA "Enable support for CUDA" OFF)
if(USE_CUDA)
  message("Building CUBLAS")
  enable_language(CUDA)
  add_library(context_manager SHARED context_manager.cu)
  target_link_libraries(context_manager -lcublas)
  LIST(APPEND RegentFlags --use-gpu)
  LIST(APPEND OUTPUTS cublas_tasks.so cublas_tasks.h)
  LIST(APPEND SRCS context_manager.cu context_manager.h cublas.rg)
  add_dependencies(blas_tasks context_manager)
endif()

add_custom_command(OUTPUT blas_tasks.so
  COMMAND regent.py blas_tasks.rg ${RegentFlags}
  DEPENDS ${SRCS}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  BYPRODUCTS ${OUTPUTS}
  VERBATIM)

add_custom_command(TARGET blas_tasks POST_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/blas.rg ${PROJECT_BINARY_DIR})
