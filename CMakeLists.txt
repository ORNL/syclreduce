cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(syclreduce VERSION 0.0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(install)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

## Setup SYCL
### Due to CMake limitations, hipSYCL requires C++ standard to be set manually
#set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")
#find_package(hipSYCL CONFIG REQUIRED)

###################################################################################################
# - define target   -------------------------------------------------------------------------------
add_library(syclreduce INTERFACE)
FILE(GLOB REDUCE_HEADERS include/syclreduce/*.hpp)
target_compile_features(syclreduce INTERFACE cxx_std_17)
target_include_directories(syclreduce INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

install_libs(TARGETS syclreduce HEADERS ${REDUCE_HEADERS})
