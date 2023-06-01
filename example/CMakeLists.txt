cmake_minimum_required(VERSION 3.8)

project(battleship VERSION 1.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(install)

#implicit option(BUILD_TESTING "Build tests accompanying this project" ON)
option(ENABLE_OPENSYCL "Build using opensycl backend." ON)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(BUILD_TESTS "Build the tests accompanying this library." ON)
option(BUILD_DOCS "Build the documentation accompanying this library." ON)

############ Package dependencies ###############
if(ENABLE_OPENSYCL)
	# Due to CMake limitations, hipSYCL requires C++ standard to be set manually
    set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")
    find_package(hipSYCL CONFIG REQUIRED)
endif()
find_package(syclreduce CONFIG REQUIRED)

##########  TARGETS  ################################################
add_library(batleth INTERFACE IMPORTED)
#add_library(batleth src/io.cpp src/util.cpp)
#FILE(GLOB BTL_HEADERS include/battleship/*.hpp)
target_link_libraries(batleth INTERFACE syclreduce::syclreduce)

#####################################################################

# configured header file stashing compile-time config options
#file(MAKE_DIRECTORY include/battleship)
#configure_file(include/battleship/config.hpp.in include/battleship/config.hpp)
#list(APPEND BTL_HEADERS ${PROJECT_BINARY_DIR}/include/battleship/config.hpp)

target_compile_features(batleth INTERFACE cxx_std_17)
#target_include_directories(batleth INTERFACE
#                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#                            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
#                            $<INSTALL_INTERFACE:include>
#                           )

# lib needs PIC when BUILD_SHARED_LIBS=ON
#set_target_properties(batleth PROPERTIES
#                      POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
#                      )

### Testing Instructions ###
#if(BUILD_TESTS)
#    add_subdirectory(tests)
#endif()
#if(BUILD_DOCS)
#    add_subdirectory(docs)
#endif()

### Installation Instructions ###
add_executable(run example.cpp)
target_link_libraries(run PUBLIC batleth)
install_bins(run)
#install_libs(TARGETS batleth HEADERS ${BTL_HEADERS})
