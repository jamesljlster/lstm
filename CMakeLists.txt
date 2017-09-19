cmake_minimum_required(VERSION 2.8)

# Environment variables
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory")
set(BUILD_TEST OFF CACHE BOOL "Enable building test")
set(WITH_CUDA OFF CACHE BOOL "Build CUDA library")

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)

# Set project
project(lstm C)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/doc)

# CUDA
if(${WITH_CUDA})
	# Find cuda package
	find_package(CUDA QUIET REQUIRED)

	# Add cuda subdirectory
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda_src)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda_src)

	if(${BUILD_TEST})
		add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda_test)
	endif()
endif()

if(${BUILD_TEST})
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
