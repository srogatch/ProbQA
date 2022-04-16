cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ProbQA LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)

string(COMPARE NOTEQUAL "${CMAKE_CUDA_ARCHITECTURES}" "" PROBQA_CUDA)
if(PROBQA_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options --suppress-stack-size-warning")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount 32")
endif()

add_subdirectory(ProbQA)
