# FindOpenBLAS.cmake — locate OpenBLAS via env vars, vcpkg config, or
# pkg-config. Sets the imported target OpenBLAS::OpenBLAS.
#
# Search order:
#   1. OpenBLAS_DIR / $ENV{OpenBLAS_HOME} / $ENV{OPENBLAS_HOME}
#   2. CMake's default search paths (vcpkg integrates here)
#   3. pkg-config "openblas"
include(FindPackageHandleStandardArgs)

set(_obl_hints
    ${OpenBLAS_DIR}
    $ENV{OpenBLAS_HOME}
    $ENV{OPENBLAS_HOME})

find_path(OpenBLAS_INCLUDE_DIR cblas.h
    HINTS ${_obl_hints}
    PATH_SUFFIXES include include/openblas openblas)

find_library(OpenBLAS_LIBRARY
    NAMES openblas libopenblas
    HINTS ${_obl_hints}
    PATH_SUFFIXES lib lib64)

# pkg-config fallback (Linux/macOS-Homebrew)
if(NOT OpenBLAS_LIBRARY OR NOT OpenBLAS_INCLUDE_DIR)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(_PC_OPENBLAS QUIET openblas)
        if(_PC_OPENBLAS_FOUND)
            if(NOT OpenBLAS_INCLUDE_DIR)
                find_path(OpenBLAS_INCLUDE_DIR cblas.h
                    HINTS ${_PC_OPENBLAS_INCLUDE_DIRS}
                    PATH_SUFFIXES openblas)
            endif()
            if(NOT OpenBLAS_LIBRARY)
                find_library(OpenBLAS_LIBRARY
                    NAMES openblas libopenblas
                    HINTS ${_PC_OPENBLAS_LIBRARY_DIRS})
            endif()
        endif()
    endif()
endif()

find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR)

if(OpenBLAS_FOUND AND NOT TARGET OpenBLAS::OpenBLAS)
    add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
    set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
        IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}")
endif()

mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)
