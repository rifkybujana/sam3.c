# cmake/Sam3Ffmpeg.cmake - FFmpeg dependency detection
#
# Exposes sam3_ffmpeg when FFmpeg is available and sets SAM3_HAS_VIDEO.
# Windows prefers vcpkg/config packages, then pkg-config. Non-Windows keeps
# the existing static source-built FFmpeg path.

set(SAM3_HAS_VIDEO OFF)

function(sam3_ffmpeg_define_target)
	if(NOT TARGET sam3_ffmpeg)
		add_library(sam3_ffmpeg INTERFACE)
	endif()
	set(SAM3_HAS_VIDEO ON PARENT_SCOPE)
endfunction()

function(sam3_ffmpeg_try_targets out_var)
	set(found_targets)
	foreach(component IN ITEMS avformat avcodec swscale swresample avutil)
		foreach(prefix IN ITEMS unofficial::ffmpeg FFmpeg FFMPEG)
			if(TARGET ${prefix}::${component})
				list(APPEND found_targets ${prefix}::${component})
			endif()
		endforeach()
	endforeach()
	set(${out_var} ${found_targets} PARENT_SCOPE)
endfunction()

if(WIN32)
	find_package(unofficial-ffmpeg CONFIG QUIET)
	sam3_ffmpeg_try_targets(SAM3_FFMPEG_TARGETS)
	if(SAM3_FFMPEG_TARGETS)
		sam3_ffmpeg_define_target()
		target_link_libraries(sam3_ffmpeg INTERFACE ${SAM3_FFMPEG_TARGETS})
		message(STATUS "FFmpeg: found (unofficial-ffmpeg config)")
	endif()

	if(NOT SAM3_HAS_VIDEO)
		find_package(FFmpeg CONFIG QUIET COMPONENTS
			avformat avcodec swscale swresample avutil)
		sam3_ffmpeg_try_targets(SAM3_FFMPEG_TARGETS)
		if(SAM3_FFMPEG_TARGETS)
			sam3_ffmpeg_define_target()
			target_link_libraries(sam3_ffmpeg INTERFACE ${SAM3_FFMPEG_TARGETS})
			message(STATUS "FFmpeg: found (FFmpeg config)")
		elseif(TARGET FFmpeg::FFmpeg)
			sam3_ffmpeg_define_target()
			target_link_libraries(sam3_ffmpeg INTERFACE FFmpeg::FFmpeg)
			message(STATUS "FFmpeg: found (FFmpeg::FFmpeg)")
		endif()
	endif()

	if(NOT SAM3_HAS_VIDEO)
		find_package(FFMPEG CONFIG QUIET COMPONENTS
			avformat avcodec swscale swresample avutil)
		sam3_ffmpeg_try_targets(SAM3_FFMPEG_TARGETS)
		if(SAM3_FFMPEG_TARGETS)
			sam3_ffmpeg_define_target()
			target_link_libraries(sam3_ffmpeg INTERFACE ${SAM3_FFMPEG_TARGETS})
			message(STATUS "FFmpeg: found (FFMPEG config)")
		elseif(TARGET FFMPEG::FFMPEG)
			sam3_ffmpeg_define_target()
			target_link_libraries(sam3_ffmpeg INTERFACE FFMPEG::FFMPEG)
			message(STATUS "FFmpeg: found (FFMPEG::FFMPEG)")
		endif()
	endif()

	if(NOT SAM3_HAS_VIDEO)
		find_package(FFMPEG QUIET)
		if(FFMPEG_FOUND AND FFMPEG_LIBRARIES)
			sam3_ffmpeg_define_target()
			if(FFMPEG_INCLUDE_DIRS)
				target_include_directories(sam3_ffmpeg INTERFACE
					${FFMPEG_INCLUDE_DIRS})
			endif()
			if(FFMPEG_LIBRARY_DIRS)
				target_link_directories(sam3_ffmpeg INTERFACE
					${FFMPEG_LIBRARY_DIRS})
			endif()
			target_link_libraries(sam3_ffmpeg INTERFACE
				${FFMPEG_LIBRARIES})
			message(STATUS "FFmpeg: found (FFMPEG module)")
		endif()
	endif()

	if(NOT SAM3_HAS_VIDEO)
		if(FFMPEG_LIBRARIES OR FFmpeg_LIBRARIES)
			if(FFMPEG_LIBRARIES)
				set(SAM3_FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
				set(SAM3_FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIRS})
				set(SAM3_FFMPEG_LIBRARY_DIRS ${FFMPEG_LIBRARY_DIRS})
			else()
				set(SAM3_FFMPEG_LIBRARIES ${FFmpeg_LIBRARIES})
				set(SAM3_FFMPEG_INCLUDE_DIRS ${FFmpeg_INCLUDE_DIRS})
				set(SAM3_FFMPEG_LIBRARY_DIRS ${FFmpeg_LIBRARY_DIRS})
			endif()
			sam3_ffmpeg_define_target()
			if(SAM3_FFMPEG_INCLUDE_DIRS)
				target_include_directories(sam3_ffmpeg INTERFACE
					${SAM3_FFMPEG_INCLUDE_DIRS})
			endif()
			if(SAM3_FFMPEG_LIBRARY_DIRS)
				target_link_directories(sam3_ffmpeg INTERFACE
					${SAM3_FFMPEG_LIBRARY_DIRS})
			endif()
			target_link_libraries(sam3_ffmpeg INTERFACE
				${SAM3_FFMPEG_LIBRARIES})
			message(STATUS "FFmpeg: found (config variables)")
		endif()
	endif()

	if(NOT SAM3_HAS_VIDEO)
		find_package(PkgConfig QUIET)
		if(PkgConfig_FOUND)
			pkg_check_modules(SAM3_PC_FFMPEG QUIET IMPORTED_TARGET
				libavformat libavcodec libswscale libswresample libavutil)
			if(TARGET PkgConfig::SAM3_PC_FFMPEG)
				sam3_ffmpeg_define_target()
				target_link_libraries(sam3_ffmpeg INTERFACE
					PkgConfig::SAM3_PC_FFMPEG)
				message(STATUS "FFmpeg: found (pkg-config)")
			endif()
		endif()
	endif()
else()
	include(${CMAKE_SOURCE_DIR}/cmake/StaticFfmpeg.cmake)
	sam3_ffmpeg_define_target()
	target_link_libraries(sam3_ffmpeg INTERFACE ffmpeg_static)
	message(STATUS "FFmpeg: using static source build")
endif()

if(NOT SAM3_HAS_VIDEO)
	if(SAM3_REQUIRE_VIDEO)
		message(FATAL_ERROR
			"FFmpeg: not found; install FFmpeg via vcpkg or disable "
			"SAM3_REQUIRE_VIDEO")
	endif()
	message(WARNING "FFmpeg: not found, disabling video support")
endif()