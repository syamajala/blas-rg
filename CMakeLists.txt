cmake_minimum_required(VERSION 3.8)
project(lapack-rg LANGUAGES CXX)

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(LAPACK REQUIRED)

find_path(LAPACK_INCLUDE_DIRS lapacke.h
  /usr/include
  /usr/local/include
  $ENV{LAPACK_HOME}/include)

set(RegentFlags --lapack-header ${LAPACK_INCLUDE_DIRS} --lapack-library ${LAPACK_LIBRARIES} -fseparate 1)

add_custom_target(lapack_tasks ALL
  DEPENDS lapack_tasks.so)
set(OUTPUTS lapack_tasks.so lapack_tasks.h lapacke_tasks.so lapacke_tasks.h)

option(USE_CUDA "Enable support for CUDA" OFF)
if(USE_CUDA)
  message("Building CUSOLVER")
  enable_language(CUDA)
  add_library(context_manager SHARED context_manager.cu)
  target_link_libraries(context_manager -lcusolver)
  LIST(APPEND RegentFlags --use-gpu)
  LIST(APPEND OUTPUTS cusolver_tasks.so cusolver_tasks.h)
  add_dependencies(lapack_tasks context_manager)
endif()

add_custom_command(OUTPUT lapack_tasks.so
  COMMAND regent.py ${CMAKE_CURRENT_LIST_DIR}/lapack_tasks.rg ${RegentFlags}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/lapack_tasks.rg
  BYPRODUCTS ${OUTPUTS}
  VERBATIM)
