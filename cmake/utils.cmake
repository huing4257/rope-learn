macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()


function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    run_python(GPU_FLAGS
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND GPU_FLAGS "-DENABLE_FP8_E5M2")
    endif()
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      list(REMOVE_ITEM GPU_FLAGS
        "-D__CUDA_NO_HALF_OPERATORS__"
        "-D__CUDA_NO_HALF_CONVERSIONS__"
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-D__CUDA_NO_HALF2_OPERATORS__")
    endif()
  endif()
  set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()

# Macro for converting a `gencode` version number to a cmake version number.
macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

macro(override_gpu_arches GPU_ARCHES GPU_LANG GPU_SUPPORTED_ARCHES)
  set(_GPU_SUPPORTED_ARCHES_LIST ${GPU_SUPPORTED_ARCHES} ${ARGN})
  message(STATUS "${GPU_LANG} supported arches: ${_GPU_SUPPORTED_ARCHES_LIST}")
  if(${GPU_LANG} STREQUAL "CUDA")
    message(DEBUG "initial CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

    # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
    string(REGEX MATCHALL "-gencode arch=[^ ]+" _CUDA_ARCH_FLAGS
      ${CMAKE_CUDA_FLAGS})

    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
      ${CMAKE_CUDA_FLAGS})
    if (NOT _CUDA_ARCH_FLAGS)
      message(FATAL_ERROR
        "Could not find any architecture related code generation flags in "
        "CMAKE_CUDA_FLAGS. (${CMAKE_CUDA_FLAGS})")
    endif()

    message(DEBUG "final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    message(DEBUG "arch flags: ${_CUDA_ARCH_FLAGS}")

    # Initialize the architecture lists to empty.
    set(${GPU_ARCHES})

    # Process each `gencode` flag.
    foreach(_ARCH ${_CUDA_ARCH_FLAGS})
      # For each flag, extract the version number and whether it refers to PTX
      # or native code.
      # Note: if a regex matches then `CMAKE_MATCH_1` holds the binding
      # for that match.

      string(REGEX MATCH "arch=compute_\([0-9]+a?\)" _COMPUTE ${_ARCH})
      if (_COMPUTE)
        set(_COMPUTE ${CMAKE_MATCH_1})
      endif()

      string(REGEX MATCH "code=sm_\([0-9]+a?\)" _SM ${_ARCH})
      if (_SM)
        set(_SM ${CMAKE_MATCH_1})
      endif()

      string(REGEX MATCH "code=compute_\([0-9]+a?\)" _CODE ${_ARCH})
      if (_CODE)
        set(_CODE ${CMAKE_MATCH_1})
      endif()

      # Make sure the virtual architecture can be matched.
      if (NOT _COMPUTE)
        message(FATAL_ERROR
          "Could not determine virtual architecture from: ${_ARCH}.")
      endif()

      # One of sm_ or compute_ must exist.
      if ((NOT _SM) AND (NOT _CODE))
        message(FATAL_ERROR
          "Could not determine a codegen architecture from: ${_ARCH}.")
      endif()

      if (_SM)
        # -real suffix let CMake to only generate elf code for the kernels.
        # we want this, otherwise the added ptx (default) will increase binary size.
        set(_VIRT "-real")
        set(_CODE_ARCH ${_SM})
      else()
        # -virtual suffix let CMake to generate ptx code for the kernels.
        set(_VIRT "-virtual")
        set(_CODE_ARCH ${_CODE})
      endif()

      # Check if the current version is in the supported arch list.
      string_to_ver(_CODE_VER ${_CODE_ARCH})
      if (NOT _CODE_VER IN_LIST _GPU_SUPPORTED_ARCHES_LIST)
        message(STATUS "discarding unsupported CUDA arch ${_VER}.")
        continue()
      endif()

      # Add it to the arch list.
      list(APPEND ${GPU_ARCHES} "${_CODE_ARCH}${_VIRT}")
    endforeach()
  endif()
  message(STATUS "${GPU_LANG} target arches: ${${GPU_ARCHES}}")
endmacro()

#
# Define a target named `GPU_MOD_NAME` for a single extension. The
# arguments are:
#
# DESTINATION <dest>         - Module destination directory.
# LANGUAGE <lang>            - The GPU language for this module, e.g CUDA, HIP,
#                              etc.
# SOURCES <sources>          - List of source files relative to CMakeLists.txt
#                              directory.
#
# Optional arguments:
#
# ARCHITECTURES <arches>     - A list of target GPU architectures in cmake
#                              format.
#                              Refer `CMAKE_CUDA_ARCHITECTURES` documentation
#                              and `CMAKE_HIP_ARCHITECTURES` for more info.
#                              ARCHITECTURES will use cmake's defaults if
#                              not provided.
# COMPILE_FLAGS <flags>      - Extra compiler flags passed to NVCC/hip.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# LIBRARIES <libraries>      - Extra link libraries.
# WITH_SOABI                 - Generate library with python SOABI suffix name.
#
# Note: optimization level/debug info is set via cmake build type.
#
function (define_gpu_extension_target GPU_MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    GPU
    "WITH_SOABI"
    "DESTINATION;LANGUAGE"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  # Add hipify preprocessing step when building with HIP/ROCm.
  if (GPU_LANGUAGE STREQUAL "HIP")
    hipify_sources_target(GPU_SOURCES ${GPU_MOD_NAME} "${GPU_SOURCES}")
  endif()

  if (GPU_WITH_SOABI)
    set(GPU_WITH_SOABI WITH_SOABI)
  else()
    set(GPU_WITH_SOABI)
  endif()

  Python_add_library(${GPU_MOD_NAME} MODULE "${GPU_SOURCES}" ${GPU_WITH_SOABI})

  if (GPU_ARCHITECTURES)
    set_target_properties(${GPU_MOD_NAME} PROPERTIES
      ${GPU_LANGUAGE}_ARCHITECTURES "${GPU_ARCHITECTURES}")
  endif()

  set_property(TARGET ${GPU_MOD_NAME} PROPERTY CXX_STANDARD 17)

  target_compile_options(${GPU_MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${GPU_LANGUAGE}>:${GPU_COMPILE_FLAGS}>)

  target_compile_definitions(${GPU_MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${GPU_MOD_NAME}")

  target_include_directories(${GPU_MOD_NAME} PRIVATE csrc
    ${GPU_INCLUDE_DIRECTORIES})

  target_link_libraries(${GPU_MOD_NAME} PRIVATE torch ${torch_python_LIBRARY}
    ${GPU_LIBRARIES})

  # Don't use `TORCH_LIBRARIES` for CUDA since it pulls in a bunch of
  # dependencies that are not necessary and may not be installed.
  if (GPU_LANGUAGE STREQUAL "CUDA")
    target_link_libraries(${GPU_MOD_NAME} PRIVATE ${CUDA_CUDA_LIB}
      ${CUDA_LIBRARIES})
  else()
    target_link_libraries(${GPU_MOD_NAME} PRIVATE ${TORCH_LIBRARIES})
  endif()

  install(TARGETS ${GPU_MOD_NAME} LIBRARY DESTINATION ${GPU_DESTINATION})
endfunction()