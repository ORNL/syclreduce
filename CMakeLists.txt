# Note - this is header-only, so we don't care what
# SYCL is being used.
# Downstreams are responsible for requiring sycl independently.
#
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(syclreduce VERSION 0.0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(install)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

###################################################################################################
# - define target   -------------------------------------------------------------------------------
add_library(syclreduce INTERFACE)
add_library(syclreduce::syclreduce ALIAS syclreduce) # for CPM imports
FILE(GLOB REDUCE_HEADERS include/syclreduce/*.hpp)
target_compile_features(syclreduce INTERFACE cxx_std_17)
target_include_directories(syclreduce INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

install_libs(TARGETS syclreduce HEADERS ${REDUCE_HEADERS})
