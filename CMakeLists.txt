cmake_minimum_required(VERSION 3.21)

project(rope LANGUAGES CXX CUDA)

option(ROPE_TARGET_DEVICE "Target device backend" "cuda")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${ROPE_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

set(TORCH_SUPPORTED_VERSION_CUDA "2.2.1")

find_python_from_executable("usr/bin/python" "${PYTHON_SUPPORTED_VERSIONS}")

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

find_package(Torch REQUIRED)

find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")

if (NOT HIP_FOUND AND CUDA_FOUND)
  set(ROPE_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()

override_gpu_arches(ROPE_GPU_ARCHES
  ${ROPE_GPU_LANG}
  "${${ROPE_GPU_LANG}_SUPPORTED_ARCHS}")

get_torch_gpu_compiler_flags(ROPE_GPU_FLAGS ${ROPE_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND ROPE_GPU_LANG STREQUAL "CUDA")
  list(APPEND ROPE_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

set(ROPE_EXT_SRC
  "csrc/rope.cu"
  "csrc/pybind.cpp")

define_gpu_extension_target(
  _C
  DESTINATION rope
  LANGUAGE ${ROPE_GPU_LANG}
  SOURCES ${ROPE_EXT_SRC}
  COMPILE_FLAGS ${ROPE_GPU_FLAGS}
  ARCHITECTURES ${ROPE_GPU_ARCHES}
  WITH_SOABI)

add_custom_target(default)

if(ROPE_GPU_LANG STREQUAL "CUDA")
  message(STATUS "Enabling C extension.")
  add_dependencies(default _C)
endif()
