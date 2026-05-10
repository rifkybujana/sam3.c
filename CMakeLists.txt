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
option(SAM3_VIDEO "Enable FFmpeg-backed video I/O" ON)
option(SAM3_REQUIRE_VIDEO "Fail configure if video dependencies are unavailable" OFF)
option(SAM3_REQUIRE_BLAS "Fail configure if BLAS is unavailable" OFF)
option(SAM3_BENCH   "Build benchmarking harness" OFF)
option(SAM3_SHARED  "Build shared library" OFF)
option(SAM3_FIXTURE_TESTS
	"Register tests that require Python-generated fixtures" OFF)
option(SAM3_E2E_TESTS
	"Build end-to-end video integration tests" OFF)
option(SAM3_BUILD_PARITY_TESTS
	"Build end-to-end Python-parity tests (requires fixtures)" OFF)
set(SAM3_PARITY_VARIANT "sam3_1" CACHE STRING
	"Variant for test_video_parity_kids {sam3, sam3_1}")
set_property(CACHE SAM3_PARITY_VARIANT PROPERTY STRINGS sam3 sam3_1)
set(SAM3_TEST_MODEL "" CACHE STRING
	"Path to a .sam3 model file used by E2E tests")

if(SAM3_E2E_TESTS AND NOT SAM3_TEST_MODEL)
	message(FATAL_ERROR
		"SAM3_E2E_TESTS=ON requires "
		"-DSAM3_TEST_MODEL=/abs/path/to/model.sam3")
endif()

# Auto-detect Metal on macOS
if(APPLE AND NOT DEFINED SAM3_METAL)
	set(SAM3_METAL ON)
elseif(APPLE)
	# User explicitly set it, keep their choice
endif()

# Compiler warnings
if(MSVC)
	add_compile_options(/W4 /wd4100 /wd4244 /wd4267)
else()
	add_compile_options(
		-Wall -Wextra -Wpedantic
		-Wno-unused-parameter
		-Wstrict-prototypes
		-Wmissing-prototypes
	)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(SAM3_SANITIZE_FLAGS "-fsanitize=address,undefined")
endif()

function(sam3_configure_executable target)
	if(MSVC)
		target_link_options(${target} PRIVATE /STACK:8388608)
	endif()
endfunction()

if(SAM3_PROFILE)
	add_definitions(-DSAM3_HAS_PROFILE)
endif()

if(SAM3_TRACE)
	add_definitions(-DSAM3_HAS_TRACE)
endif()

if(SAM3_DEBUG_DUMP)
	add_definitions(-DSAM3_DEBUG_DUMP)
endif()

set(SAM3_HAS_VIDEO OFF)
if(SAM3_VIDEO)
	include(${CMAKE_SOURCE_DIR}/cmake/Sam3Ffmpeg.cmake)
elseif(SAM3_REQUIRE_VIDEO)
	message(FATAL_ERROR "SAM3_REQUIRE_VIDEO=ON requires SAM3_VIDEO=ON")
endif()

set(SAM3_BLAS_TARGETS)
set(SAM3_BLAS_LIBRARIES)
set(SAM3_BLAS_INCLUDE_DIRS)
set(SAM3_BLAS_FOUND OFF)

if(SAM3_BLAS)
	if(APPLE)
		find_library(ACCELERATE_LIB Accelerate)
		if(ACCELERATE_LIB)
			set(SAM3_BLAS_FOUND ON)
			set(SAM3_BLAS_LIBRARIES ${ACCELERATE_LIB})
			message(STATUS "BLAS: using Accelerate framework")
		endif()
	elseif(WIN32)
		find_package(OpenBLAS CONFIG QUIET)
		if(TARGET OpenBLAS::OpenBLAS)
			set(SAM3_BLAS_FOUND ON)
			set(SAM3_BLAS_TARGETS OpenBLAS::OpenBLAS)
			message(STATUS "BLAS: found OpenBLAS")
		elseif(OpenBLAS_FOUND)
			set(SAM3_BLAS_FOUND ON)
			set(SAM3_BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
			set(SAM3_BLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
			message(STATUS "BLAS: found OpenBLAS")
		endif()

		if(NOT SAM3_BLAS_FOUND)
			find_package(BLAS QUIET)
			if(BLAS_FOUND)
				find_path(SAM3_CBLAS_INCLUDE_DIR cblas.h)
				if(BLAS_INCLUDE_DIRS OR SAM3_CBLAS_INCLUDE_DIR)
					set(SAM3_BLAS_FOUND ON)
					set(SAM3_BLAS_LIBRARIES ${BLAS_LIBRARIES})
					set(SAM3_BLAS_INCLUDE_DIRS
						${BLAS_INCLUDE_DIRS} ${SAM3_CBLAS_INCLUDE_DIR})
					message(STATUS "BLAS: found ${BLAS_LIBRARIES}")
				else()
					message(WARNING
						"BLAS: library found but cblas.h was not found")
				endif()
			endif()
		endif()
	else()
		find_package(BLAS QUIET)
		if(BLAS_FOUND)
			set(SAM3_BLAS_FOUND ON)
			set(SAM3_BLAS_LIBRARIES ${BLAS_LIBRARIES})
			message(STATUS "BLAS: found ${BLAS_LIBRARIES}")
		endif()
	endif()

	if(NOT SAM3_BLAS_FOUND)
		if(SAM3_REQUIRE_BLAS)
			message(FATAL_ERROR
				"BLAS: not found; install OpenBLAS via vcpkg or disable "
				"SAM3_REQUIRE_BLAS")
		endif()
		message(WARNING "BLAS: not found, falling back to SIMD kernels")
		set(SAM3_BLAS OFF)
	endif()
elseif(SAM3_REQUIRE_BLAS)
	message(FATAL_ERROR "SAM3_REQUIRE_BLAS=ON requires SAM3_BLAS=ON")
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
if(MSVC)
	include_directories(BEFORE ${CMAKE_SOURCE_DIR}/compat/msvc)
endif()

# Library sources
file(GLOB_RECURSE SAM3_CORE_SOURCES   "src/core/*.c")
file(GLOB_RECURSE SAM3_MODEL_SOURCES  "src/model/*.c")
file(GLOB_RECURSE SAM3_UTIL_SOURCES   "src/util/*.c")

if(NOT SAM3_HAS_VIDEO)
	list(REMOVE_ITEM SAM3_MODEL_SOURCES
		${CMAKE_SOURCE_DIR}/src/model/sam3_video.c
		${CMAKE_SOURCE_DIR}/src/model/video_session.c)
	list(REMOVE_ITEM SAM3_UTIL_SOURCES
		${CMAKE_SOURCE_DIR}/src/util/video.c
		${CMAKE_SOURCE_DIR}/src/util/video_encode.c)
endif()

# Suppress warnings for vendored code
file(GLOB_RECURSE SAM3_VENDORED_SOURCES "src/core/json/*.c")
if(MSVC)
	set_source_files_properties(${SAM3_VENDORED_SOURCES} PROPERTIES
		COMPILE_FLAGS "/wd4100 /wd4244 /wd4267")
else()
	set_source_files_properties(${SAM3_VENDORED_SOURCES} PROPERTIES
		COMPILE_FLAGS "-Wno-strict-prototypes -Wno-missing-prototypes -Wno-unused-function -Wno-deprecated-declarations")
endif()

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
	if(WIN32)
		set_target_properties(sam3 PROPERTIES
			WINDOWS_EXPORT_ALL_SYMBOLS ON
		)
	endif()
	if(APPLE)
		set_target_properties(sam3 PROPERTIES
			LINK_FLAGS "-exported_symbols_list ${CMAKE_SOURCE_DIR}/bindings/python/exports.txt"
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

if(SAM3_HAS_VIDEO)
	target_compile_definitions(sam3 PUBLIC SAM3_HAS_VIDEO)
	target_link_libraries(sam3 sam3_ffmpeg)
endif()

if(SAM3_BLAS)
	target_compile_definitions(sam3 PUBLIC SAM3_HAS_BLAS)
	if(SAM3_BLAS_INCLUDE_DIRS)
		target_include_directories(sam3 PRIVATE ${SAM3_BLAS_INCLUDE_DIRS})
	endif()
	if(SAM3_BLAS_TARGETS OR SAM3_BLAS_LIBRARIES)
		target_link_libraries(sam3 ${SAM3_BLAS_TARGETS} ${SAM3_BLAS_LIBRARIES})
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
	tools/cli_track.c
	tools/cli_convert.c
	tools/cli_info.c
	tools/weight_rename.c
	tools/weight_conv_perm.c)
sam3_configure_executable(sam3_cli)
target_include_directories(sam3_cli PRIVATE ${CMAKE_SOURCE_DIR}/tools)
target_link_libraries(sam3_cli sam3)
if(SAM3_BENCH)
	target_sources(sam3_cli PRIVATE tools/cli_bench.c)
	target_link_libraries(sam3_cli sam3_bench)
endif()

add_executable(gen_nhwc_fixtures tools/gen_nhwc_fixtures.c)
sam3_configure_executable(gen_nhwc_fixtures)
target_link_libraries(gen_nhwc_fixtures sam3)

# SAM 3.1 parity fixture seed dumper (manual regen tool, not CI).
add_executable(sam3_1_dump_seed tools/sam3_1_dump_seed.c)
sam3_configure_executable(sam3_1_dump_seed)
target_link_libraries(sam3_1_dump_seed sam3)
target_include_directories(sam3_1_dump_seed PRIVATE
	${CMAKE_SOURCE_DIR}/src)

# Tests
if(SAM3_TESTS)
	enable_testing()
	file(GLOB TEST_SOURCES "tests/test_*.c")
	# test_helpers_png.c is a helper TU, not a standalone test.
	list(REMOVE_ITEM TEST_SOURCES
		${CMAKE_SOURCE_DIR}/tests/test_helpers_png.c)
	if(NOT SAM3_HAS_VIDEO)
		file(GLOB VIDEO_TEST_SOURCES "tests/test_video*.c")
		list(REMOVE_ITEM TEST_SOURCES ${VIDEO_TEST_SOURCES})
	endif()
	if(NOT SAM3_FIXTURE_TESTS)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_tracker_fixtures.c)
	endif()
	if(NOT SAM3_E2E_TESTS)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_video_e2e.c)
	endif()
	if(NOT SAM3_BUILD_PARITY_TESTS)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_video_parity_kids.c)
	endif()
	if(NOT SAM3_BENCH)
		list(REMOVE_ITEM TEST_SOURCES
			${CMAKE_SOURCE_DIR}/tests/test_bench.c
			${CMAKE_SOURCE_DIR}/tests/test_bench_video.c)
	endif()
	foreach(test_src ${TEST_SOURCES})
		get_filename_component(test_name ${test_src} NAME_WE)
		add_executable(${test_name} ${test_src})
		sam3_configure_executable(${test_name})
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

	# test_processor_cache opens models/sam3.sam3
	if(TARGET test_processor_cache)
		target_compile_definitions(test_processor_cache PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_segment_batch opens models/sam3.sam3
	if(TARGET test_segment_batch)
		target_compile_definitions(test_segment_batch PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_persist_bus_smoke opens models/sam3.sam3 and assets/bus.jpg
	if(TARGET test_persist_bus_smoke)
		target_compile_definitions(test_persist_bus_smoke PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_tracker_multiplex_load opens models/sam3.1.sam3
	if(TARGET test_tracker_multiplex_load)
		target_compile_definitions(test_tracker_multiplex_load PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_sam3_1_track opens models/sam3.1.sam3 + assets/kids.mp4
	if(TARGET test_sam3_1_track)
		target_compile_definitions(test_sam3_1_track PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_maskmem_multiplex_forward runs the forward graph; same dependency.
	if(TARGET test_maskmem_multiplex_forward)
		target_compile_definitions(test_maskmem_multiplex_forward PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_memory_attn_multiplex_forward also loads sam3.1.sam3.
	if(TARGET test_memory_attn_multiplex_forward)
		target_compile_definitions(test_memory_attn_multiplex_forward PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_mask_decoder_multiplex_forward also loads sam3.1.sam3.
	if(TARGET test_mask_decoder_multiplex_forward)
		target_compile_definitions(test_mask_decoder_multiplex_forward PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_cli_common needs tools/ include path and cli_common.c
	if(TARGET test_cli_common)
		target_sources(test_cli_common PRIVATE tools/cli_common.c)
		target_include_directories(test_cli_common PRIVATE
			${CMAKE_SOURCE_DIR}/tools)
	endif()

	# test_cli_track needs tools/ include path and cli_track.c
	if(TARGET test_cli_track)
		target_sources(test_cli_track PRIVATE
			tools/cli_track.c)
		target_include_directories(test_cli_track PRIVATE
			${CMAKE_SOURCE_DIR}/tools)
	endif()

	# test_sam3_main shells out to the built CLI executable.
	if(TARGET test_sam3_main)
		target_compile_definitions(test_sam3_main PRIVATE
			"SAM3_CLI_PATH=\"$<TARGET_FILE_NAME:sam3_cli>\"")
		set_tests_properties(test_sam3_main PROPERTIES
			WORKING_DIRECTORY "$<TARGET_FILE_DIR:sam3_cli>")
	endif()

	# test_video_encode needs libav headers directly for the roundtrip test
	if(SAM3_HAS_VIDEO AND TARGET test_video_encode)
		target_link_libraries(test_video_encode sam3_ffmpeg)
	endif()

	# test_video_io needs source dir for test data
	if(TARGET test_video_io)
		target_compile_definitions(test_video_io PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_video_api needs source dir for model path and fixture data
	if(TARGET test_video_api)
		target_compile_definitions(test_video_api PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_tracker_fixtures needs source dir for fixture paths
	if(TARGET test_tracker_fixtures)
		target_compile_definitions(test_tracker_fixtures PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_mobileclip_text needs source dir for fixture paths
	if(TARGET test_mobileclip_text)
		target_compile_definitions(test_mobileclip_text PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()

	# test_video_e2e needs the model path at build time
	if(TARGET test_video_e2e)
		target_compile_definitions(test_video_e2e PRIVATE
			SAM3_TEST_MODEL="${SAM3_TEST_MODEL}")
	endif()

	# test_video_parity_kids needs source dir, model path, and variant
	if(TARGET test_video_parity_kids)
		target_compile_definitions(test_video_parity_kids PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
			SAM3_TEST_MODEL="${SAM3_TEST_MODEL}"
			SAM3_PARITY_VARIANT_${SAM3_PARITY_VARIANT}=1)
		target_sources(test_video_parity_kids PRIVATE
			${CMAKE_SOURCE_DIR}/tests/test_helpers_png.c)
		target_include_directories(test_video_parity_kids PRIVATE
			${CMAKE_SOURCE_DIR}/src
			${CMAKE_SOURCE_DIR}/tests)
	endif()

	# test_bench needs the bench harness library
	if(SAM3_BENCH AND TARGET test_bench)
		target_link_libraries(test_bench sam3_bench)
	endif()

	# test_bench_video needs the bench harness library
	if(SAM3_BENCH AND TARGET test_bench_video)
		target_link_libraries(test_bench_video sam3_bench)
	endif()
endif()

# Benchmarks (not included in CTest; use Release build for accurate numbers)
set(BENCH_SOURCES
	tests/bench_dtype.c
	tests/bench_metal.c
	tests/bench_tokenizer.c
	tests/bench_segment.c
)

foreach(BENCH_SRC ${BENCH_SOURCES})
	get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)
	add_executable(${BENCH_NAME} ${BENCH_SRC})
	sam3_configure_executable(${BENCH_NAME})
	target_link_libraries(${BENCH_NAME} sam3)
endforeach()
