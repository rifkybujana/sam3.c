# cmake/StaticFfmpeg.cmake - Static FFmpeg + openh264 + libvpx build
#
# Builds FFmpeg and its required encoders from source as static
# archives so the final sam3 binary has no runtime dependency on
# Homebrew/system ffmpeg. Exposes a single INTERFACE library,
# `ffmpeg_static`, that aggregates every static archive plus the
# transitive system libraries/frameworks that FFmpeg needs.
#
# License notes:
#   - openh264  (Cisco, BSD-2-Clause): encodes H.264 Baseline profile.
#     Selected over libx264 to keep the resulting binary MIT-compatible.
#   - libvpx    (Google, BSD-3-Clause): encodes VP9 (WebM).
#   - ffmpeg    (LGPL-2.1+): libav* libraries; with only openh264 and
#     libvpx enabled, no GPL components are pulled in.
#
# First build cost: ~10-15 minutes on an Apple-silicon laptop. Fetches
# tarballs to <build>/external/ and caches in-source builds across
# incremental rebuilds; `rm -rf build` forces a refetch+rebuild.

include(ExternalProject)

set(FFMPEG_PREFIX "${CMAKE_BINARY_DIR}/external/install")
file(MAKE_DIRECTORY "${FFMPEG_PREFIX}/include")

# Pin versions. Update together and retest.
set(OPENH264_VERSION 2.5.0)
set(LIBVPX_VERSION   1.14.1)
set(FFMPEG_VERSION   7.1)

# Parallel build jobs (defaults to all cores).
include(ProcessorCount)
ProcessorCount(EP_JOBS)
if(EP_JOBS EQUAL 0)
	set(EP_JOBS 4)
endif()

# --- openh264 -----------------------------------------------------------
# openh264's Makefile writes .a and .h into PREFIX when the
# `install-static` target is used; no shared lib is installed.
ExternalProject_Add(openh264_ep
	URL https://github.com/cisco/openh264/archive/refs/tags/v${OPENH264_VERSION}.tar.gz
	DOWNLOAD_NO_PROGRESS ON
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	PREFIX "${CMAKE_BINARY_DIR}/external/openh264"
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ""
	BUILD_COMMAND     make PREFIX=${FFMPEG_PREFIX} -j${EP_JOBS}
	INSTALL_COMMAND   make PREFIX=${FFMPEG_PREFIX} install-static
	BUILD_BYPRODUCTS  "${FFMPEG_PREFIX}/lib/libopenh264.a"
	LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
)

# --- libvpx -------------------------------------------------------------
ExternalProject_Add(libvpx_ep
	URL https://github.com/webmproject/libvpx/archive/refs/tags/v${LIBVPX_VERSION}.tar.gz
	DOWNLOAD_NO_PROGRESS ON
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	PREFIX "${CMAKE_BINARY_DIR}/external/libvpx"
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ./configure
		--prefix=${FFMPEG_PREFIX}
		--enable-static
		--disable-shared
		--disable-examples
		--disable-tools
		--disable-docs
		--disable-unit-tests
		--disable-install-bins
		--disable-install-srcs
	BUILD_COMMAND     make -j${EP_JOBS}
	INSTALL_COMMAND   make install
	BUILD_BYPRODUCTS  "${FFMPEG_PREFIX}/lib/libvpx.a"
	LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
)

# --- ffmpeg -------------------------------------------------------------
# Autodetected Apple frameworks (VideoToolbox, CoreMedia, etc.) are
# acceptable — they ship with macOS and add no runtime dep.
# --disable-network avoids pulling in OpenSSL/Security for TLS.
ExternalProject_Add(ffmpeg_ep
	URL https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${FFMPEG_VERSION}.tar.gz
	DOWNLOAD_NO_PROGRESS ON
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	PREFIX "${CMAKE_BINARY_DIR}/external/ffmpeg"
	BUILD_IN_SOURCE 1
	DEPENDS openh264_ep libvpx_ep
	CONFIGURE_COMMAND
		${CMAKE_COMMAND} -E env
			PKG_CONFIG_PATH=${FFMPEG_PREFIX}/lib/pkgconfig
		./configure
		--prefix=${FFMPEG_PREFIX}
		--pkg-config-flags=--static
		--extra-cflags=-I${FFMPEG_PREFIX}/include
		--extra-ldflags=-L${FFMPEG_PREFIX}/lib
		--enable-static
		--disable-shared
		--disable-programs
		--disable-doc
		--disable-network
		--enable-libopenh264
		--enable-libvpx
		--enable-encoder=libopenh264
		--enable-encoder=libvpx_vp9
	BUILD_COMMAND     make -j${EP_JOBS}
	INSTALL_COMMAND   make install
	BUILD_BYPRODUCTS
		"${FFMPEG_PREFIX}/lib/libavformat.a"
		"${FFMPEG_PREFIX}/lib/libavcodec.a"
		"${FFMPEG_PREFIX}/lib/libswscale.a"
		"${FFMPEG_PREFIX}/lib/libavutil.a"
	LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
)

# --- aggregate INTERFACE library ---------------------------------------
add_library(ffmpeg_static INTERFACE)
add_dependencies(ffmpeg_static ffmpeg_ep)

target_include_directories(ffmpeg_static INTERFACE
	"${FFMPEG_PREFIX}/include")

# Order matters for static linking: avformat references avcodec, which
# references swscale and avutil; the encoder libs are leaf-most.
target_link_libraries(ffmpeg_static INTERFACE
	"${FFMPEG_PREFIX}/lib/libavformat.a"
	"${FFMPEG_PREFIX}/lib/libavcodec.a"
	"${FFMPEG_PREFIX}/lib/libswscale.a"
	"${FFMPEG_PREFIX}/lib/libswresample.a"
	"${FFMPEG_PREFIX}/lib/libavutil.a"
	"${FFMPEG_PREFIX}/lib/libvpx.a"
	"${FFMPEG_PREFIX}/lib/libopenh264.a"
)

# Transitive system deps pulled in by ffmpeg with our configure flags.
if(APPLE)
	target_link_libraries(ffmpeg_static INTERFACE
		"-framework CoreFoundation"
		"-framework CoreMedia"
		"-framework CoreVideo"
		"-framework VideoToolbox"
		"-framework CoreAudio"
		"-framework AudioToolbox"
		"-framework CoreGraphics"
		"-framework CoreServices"
		"-lz"
		"-lbz2"
		"-liconv"
		"-lc++"
	)
else()
	target_link_libraries(ffmpeg_static INTERFACE
		"-lz"
		"-lbz2"
		"-llzma"
		"-lm"
		"-lpthread"
	)
endif()
