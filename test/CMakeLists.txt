cmake_minimum_required(VERSION 2.8)

# Set project
project(lstm_test C)

# Set file list
file(GLOB PROJECT_SRCS 
	${CMAKE_CURRENT_SOURCE_DIR}/*.c
	)

file(GLOB TEST_TXT
	${CMAKE_CURRENT_SOURCE_DIR}/*.lstm
	${CMAKE_CURRENT_SOURCE_DIR}/*.csv
	)

# Set dependences
set(PROJECT_DEPS lstm m)
if(UNIX)
	find_library(CONIO_LIB conio
		"/usr/lib"
		"/usr/local/lib"
		)
	find_path(CONIO_INCLUDE_DIR conio.h
		"/usr/include"
		"/usr/local/include"
		)

	set(PROJECT_DEPS ${PROJECT_DEPS} ${CONIO_LIB})
	include_directories(${CONIO_INCLUDE_DIR})
endif()

install(FILES ${TEST_TXT}
	DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
	)

# Build test
foreach(TEST_FILE_PATH ${PROJECT_SRCS})
	# Get each file name
	get_filename_component(TEST_FILE_NAME ${TEST_FILE_PATH} NAME_WE)
	
	# Build executable
	add_executable(${TEST_FILE_NAME} ${TEST_FILE_PATH})
	set_target_properties(${TEST_FILE_NAME}
		PROPERTIES OUTPUT_NAME ${TEST_FILE_NAME}
		)
	target_link_libraries(${TEST_FILE_NAME} ${PROJECT_DEPS})

	# Install
	install(TARGETS ${TEST_FILE_NAME}
		RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
		)
endforeach()

