cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MedVisNetInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_DIR environment variable not set")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found in ${ONNXRUNTIME_DIR}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_DIR}/lib")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(${ONNXRUNTIME_DIR}/lib)

set(SOURCES
    src/inferencer.cpp
    src/main.cpp
)

add_executable(inference ${SOURCES})

target_link_libraries(inference
    ${OpenCV_LIBS}
    onnxruntime
    Threads::Threads
)

target_compile_options(inference PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0>
    -Wall -Wextra -Wpedantic
)

set_target_properties(inference PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

install(TARGETS inference RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")
message(STATUS "=============================")
message(STATUS "")
