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
option(SAM3_DEBUG_DUMP "Enable debug tensor dumps to /tmp" OFF)
option(SAM3_BLAS    "Use BLAS for CPU matmul (Accelerate/OpenBLAS/MKL)" ON)
option(SAM3_BENCH   "Build benchmarking harness" OFF)
option(SAM3_SHARED  "Build shared library" OFF)

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
	set(SAM3_SANITIZE_FLAGS "-fsanitize=address,undefined")
endif()

if(SAM3_PROFILE)
	add_definitions(-DSAM3_HAS_PROFILE)
endif()

if(SAM3_TRACE)
	add_definitions(-DSAM3_HAS_TRACE)
endif()

if(SAM3_DEBUG_DUMP)
	add_definitions(-DSAM3_DEBUG_DUMP)
endif()

if(SAM3_BLAS)
	if(APPLE)
		find_library(ACCELERATE_LIB Accelerate)
		if(ACCELERATE_LIB)
			add_definitions(-DSAM3_HAS_BLAS)
			message(STATUS "BLAS: using Accelerate framework")
		else()
			message(WARNING "BLAS: Accelerate not found, disabling")
			set(SAM3_BLAS OFF)
		endif()
	else()
		find_package(BLAS QUIET)
		if(BLAS_FOUND)
			add_definitions(-DSAM3_HAS_BLAS)
			message(STATUS "BLAS: found ${BLAS_LIBRARIES}")
		else()
			message(WARNING "BLAS: not found, falling back to SIMD kernels")
			set(SAM3_BLAS OFF)
		endif()
	endif()
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

if(SAM3_SHARED)
	add_library(sam3 SHARED ${SAM3_SOURCES})
	set_target_properties(sam3 PROPERTIES
		POSITION_INDEPENDENT_CODE ON
	)
	if(APPLE)
		set_target_properties(sam3 PROPERTIES
			LINK_FLAGS "-exported_symbols_list ${CMAKE_SOURCE_DIR}/python/exports.txt"
		)
	endif()
else()
	add_library(sam3 STATIC ${SAM3_SOURCES})
endif()

# Apply sanitizers only to our own code (not fetched dependencies like MLX)
if(SAM3_SANITIZE_FLAGS)
	target_compile_options(sam3 PRIVATE ${SAM3_SANITIZE_FLAGS})
	target_link_options(sam3 PUBLIC ${SAM3_SANITIZE_FLAGS})
endif()

# Link pthreads
find_package(Threads REQUIRED)
target_link_libraries(sam3 Threads::Threads)

# Link zlib (for gzip BPE vocab loading)
find_package(ZLIB REQUIRED)
target_link_libraries(sam3 ZLIB::ZLIB)

if(SAM3_BLAS)
	if(APPLE)
		target_link_libraries(sam3 ${ACCELERATE_LIB})
	else()
		target_link_libraries(sam3 ${BLAS_LIBRARIES})
	endif()
endif()

if(SAM3_METAL)
	target_link_libraries(sam3 mlxc)
endif()

# Benchmark harness library
if(SAM3_BENCH)
	add_definitions(-DSAM3_HAS_BENCH)
	file(GLOB_RECURSE SAM3_BENCH_SOURCES "src/bench/*.c")
	add_library(sam3_bench STATIC ${SAM3_BENCH_SOURCES})
	target_link_libraries(sam3_bench sam3)
endif()

# CLI tool (unified binary with subcommands)
add_executable(sam3_cli
	tools/sam3_cli.c
	tools/cli_common.c
	tools/cli_segment.c
	tools/cli_convert.c
	tools/cli_info.c
	tools/weight_rename.c
	tools/weight_conv_perm.c)
target_include_directories(sam3_cli PRIVATE ${CMAKE_SOURCE_DIR}/tools)
target_link_libraries(sam3_cli sam3)
if(SAM3_BENCH)
	target_sources(sam3_cli PRIVATE tools/cli_bench.c)
	target_link_libraries(sam3_cli sam3_bench)
endif()

add_executable(gen_nhwc_fixtures tools/gen_nhwc_fixtures.c)
target_link_libraries(gen_nhwc_fixtures sam3)

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

	# test_weight_rename needs weight_rename.c and tools/ include path
	if(TARGET test_weight_rename)
		target_sources(test_weight_rename PRIVATE tools/weight_rename.c)
		target_include_directories(test_weight_rename PRIVATE
			${CMAKE_SOURCE_DIR}/tools)
	endif()

	# test_fixture_compare needs to find fixtures and model at build time
	if(TARGET test_fixture_compare)
		target_compile_definitions(test_fixture_compare PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_neck_nhwc reads pre-migration reference .bin fixtures
	if(TARGET test_neck_nhwc)
		target_compile_definitions(test_neck_nhwc PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_seg_head_nhwc reads pre-migration reference .bin fixtures
	if(TARGET test_seg_head_nhwc)
		target_compile_definitions(test_seg_head_nhwc PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_mask_decoder_nhwc reads pre-migration reference .bin fixtures
	if(TARGET test_mask_decoder_nhwc)
		target_compile_definitions(test_mask_decoder_nhwc PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_processor_async needs the model path at build time
	if(TARGET test_processor_async)
		target_compile_definitions(test_processor_async PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_cli_common needs tools/ include path and cli_common.c
	if(TARGET test_cli_common)
		target_sources(test_cli_common PRIVATE tools/cli_common.c)
		target_include_directories(test_cli_common PRIVATE
			${CMAKE_SOURCE_DIR}/tools)
	endif()

	# test_bench needs the bench harness library
	if(SAM3_BENCH AND TARGET test_bench)
		target_link_libraries(test_bench sam3_bench)
	endif()
endif()

# Benchmarks (not included in CTest; use Release build for accurate numbers)
set(BENCH_SOURCES
	tests/bench_dtype.c
	tests/bench_metal.c
	tests/bench_tokenizer.c
)

foreach(BENCH_SRC ${BENCH_SOURCES})
	get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)
	add_executable(${BENCH_NAME} ${BENCH_SRC})
	target_link_libraries(${BENCH_NAME} sam3)
endforeach()
