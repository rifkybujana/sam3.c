cmake_minimum_required(VERSION 3.20)
project(sam3 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SAM3_METAL "Enable Metal backend" OFF)
option(SAM3_CPU   "Enable CPU backend"   ON)
option(SAM3_TESTS "Build tests"          ON)
option(SAM3_PROFILE "Enable profiler"     OFF)
option(SAM3_TRACE   "Enable dtype tracing" OFF)

# Auto-detect Metal on macOS
if(APPLE AND NOT DEFINED SAM3_METAL)
	set(SAM3_METAL ON)
elseif(APPLE)
	# User explicitly set it, keep their choice
endif()

# Compiler warnings
add_compile_options(
	-Wall -Wextra -Wpedantic
	-Wno-unused-parameter
	-Wstrict-prototypes
	-Wmissing-prototypes
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_compile_options(-fsanitize=address,undefined)
	add_link_options(-fsanitize=address,undefined)
endif()

if(SAM3_PROFILE)
	add_definitions(-DSAM3_HAS_PROFILE)
endif()

if(SAM3_TRACE)
	add_definitions(-DSAM3_HAS_TRACE)
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Library sources
file(GLOB_RECURSE SAM3_CORE_SOURCES   "src/core/*.c")
file(GLOB_RECURSE SAM3_MODEL_SOURCES  "src/model/*.c")
file(GLOB_RECURSE SAM3_UTIL_SOURCES   "src/util/*.c")

# Suppress warnings for vendored code
file(GLOB_RECURSE SAM3_VENDORED_SOURCES "src/core/json/*.c")
set_source_files_properties(${SAM3_VENDORED_SOURCES} PROPERTIES
    COMPILE_FLAGS "-Wno-strict-prototypes -Wno-missing-prototypes -Wno-unused-function -Wno-deprecated-declarations")

set(SAM3_SOURCES
	src/sam3.c
	${SAM3_CORE_SOURCES}
	${SAM3_MODEL_SOURCES}
	${SAM3_UTIL_SOURCES}
)

# Backend factory (always needed)
list(APPEND SAM3_SOURCES src/backend/backend.c)

# Backend sources
if(SAM3_CPU)
	file(GLOB_RECURSE SAM3_CPU_SOURCES "src/backend/cpu/*.c")
	list(APPEND SAM3_SOURCES ${SAM3_CPU_SOURCES})
	add_definitions(-DSAM3_HAS_CPU)
endif()

if(SAM3_METAL)
	include(FetchContent)
	FetchContent_Declare(mlx-c
		GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
		GIT_TAG        v0.6.0
	)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
	set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(mlx-c)

	file(GLOB_RECURSE SAM3_METAL_SOURCES "src/backend/metal/*.c")
	list(APPEND SAM3_SOURCES ${SAM3_METAL_SOURCES})
	add_definitions(-DSAM3_HAS_METAL)
endif()

# Static library
add_library(sam3 STATIC ${SAM3_SOURCES})

# Link pthreads
find_package(Threads REQUIRED)
target_link_libraries(sam3 Threads::Threads)

# Link zlib (for gzip BPE vocab loading)
find_package(ZLIB REQUIRED)
target_link_libraries(sam3 ZLIB::ZLIB)

if(SAM3_METAL)
	target_link_libraries(sam3 mlxc)
endif()

# CLI tools
add_executable(sam3_main tools/sam3_main.c)
target_link_libraries(sam3_main sam3)

add_executable(sam3_convert tools/sam3_convert.c)
target_link_libraries(sam3_convert sam3)

# Tests
if(SAM3_TESTS)
	enable_testing()
	file(GLOB TEST_SOURCES "tests/test_*.c")
	foreach(test_src ${TEST_SOURCES})
		get_filename_component(test_name ${test_src} NAME_WE)
		add_executable(${test_name} ${test_src})
		target_link_libraries(${test_name} sam3)
		add_test(NAME ${test_name} COMMAND ${test_name})
	endforeach()

	# test_trace needs trace enabled regardless of global setting
	if(TARGET test_trace)
		target_compile_definitions(test_trace PRIVATE SAM3_HAS_TRACE)
	endif()
endif()

# Benchmarks (not included in CTest)
add_executable(bench_dtype tests/bench_dtype.c)
target_link_libraries(bench_dtype sam3)

add_executable(bench_metal tests/bench_metal.c)
target_link_libraries(bench_metal sam3)
