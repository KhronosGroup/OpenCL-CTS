cmake_minimum_required(VERSION 3.5.1)

set( CONFORMANCE_SUFFIX "" )
set(CLConform_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

project(CLConform${CONFORMANCE_SUFFIX})

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_2_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_1_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_0_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_2_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_0_APIS=1)
add_definitions(-DCL_NO_EXTENSION_PROTOTYPES)

option(USE_CL_EXPERIMENTAL "Use Experimental definitions" OFF)
if(USE_CL_EXPERIMENTAL)
  add_definitions(-DCL_EXPERIMENTAL)
endif(USE_CL_EXPERIMENTAL)

#-----------------------------------------------------------
# Default Configurable Test Set
#-----------------------------------------------------------
option(D3D10_IS_SUPPORTED "Run DirectX 10 interop tests" OFF)
option(D3D11_IS_SUPPORTED "Run DirectX 11 interop tests" OFF)
option(GL_IS_SUPPORTED "Run OpenGL interop tests" OFF)
option(GLES_IS_SUPPORTED "Run OpenGL ES interop tests" OFF)
option(VULKAN_IS_SUPPORTED "Run Vulkan interop tests" OFF)


#-----------------------------------------------------------
# Tests prefix and suffix
#-----------------------------------------------------------
# Set the prefix and suffix for the generated executables
# For example, if you want the api executable to be test_conformance_api_12
# Set prefix to "test_conformance_" and suffix to "_12"
set(CONFORMANCE_PREFIX "test_" )
set(CONFORMANCE_SUFFIX "" )

#-----------------------------------------------------------
# Vendor Customization
#-----------------------------------------------------------
#Vendor Customization File can be included here to provide a way to automatically
#build driver as a dependency of the conformance tests, or other such CMake customization
include(CMakeVendor.txt OPTIONAL)

# CL_INCLUDE_DIR - path to dir with OpenCL headers
if(CL_INCLUDE_DIR AND CL_LIB_DIR)
    link_directories(${CL_LIB_DIR})
else(CL_INCLUDE_DIR AND CL_LIB_DIR)
    message(STATUS "OpenCL hasn't been found!")
    message(FATAL_ERROR "Either install OpenCL or pass -DCL_INCLUDE_DIR and -DCL_LIB_DIR")
endif(CL_INCLUDE_DIR AND CL_LIB_DIR)

# CLConform_GL_LIBRARIES_DIR - path to OpenGL libraries
if(GL_IS_SUPPORTED AND CLConform_GL_LIBRARIES_DIR)
    link_directories(${CLConform_GL_LIBRARIES_DIR})
endif (GL_IS_SUPPORTED AND CLConform_GL_LIBRARIES_DIR)

include(CheckFunctionExists)
include(CheckIncludeFiles)
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(CLConform_TARGET_ARCH ARM)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set(CLConform_TARGET_ARCH ARM64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(CLConform_TARGET_ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*")
    set(CLConform_TARGET_ARCH x86)
endif()

if(NOT DEFINED CLConform_TARGET_ARCH)
    message (FATAL_ERROR "Target architecture not recognised. Exiting.")
endif()

macro(add_cxx_flag_if_supported flag)
    string(REGEX REPLACE "[-=+]" "" FLAG_NO_SIGNS ${flag})
    set(FLAG_TO_TEST ${flag})
    if((${flag} MATCHES "^-Wno-") AND NOT (${flag} MATCHES "^-Wno-error="))
        # -Wno-... only causes a diagnostic if another diagnostic is emitted.
        # Change such flags into a -W... flag to test if the warning is known.
        string(REGEX REPLACE "^-Wno-" "-W" FLAG_TO_TEST ${flag})
    endif()
    check_cxx_compiler_flag(${FLAG_TO_TEST} COMPILER_SUPPORTS_${FLAG_NO_SIGNS})
    if(COMPILER_SUPPORTS_${FLAG_NO_SIGNS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
    endif()
endmacro(add_cxx_flag_if_supported)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    add_cxx_flag_if_supported(-Wmisleading-indentation)
    add_cxx_flag_if_supported(-Wunused-function)
    add_cxx_flag_if_supported(-Wunused-variable)
    add_cxx_flag_if_supported(-Werror)
    if(NOT CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo|MinSizeRel")
        # Enable more warnings if not doing a release build.
        add_cxx_flag_if_supported(-Wall)
    endif()
    add_cxx_flag_if_supported(-Wno-narrowing)
    add_cxx_flag_if_supported(-Wno-format)
    add_cxx_flag_if_supported(-Wno-error=cpp) # Allow #warning directive
    add_cxx_flag_if_supported(-Wno-unknown-pragmas) # Issue #785
    add_cxx_flag_if_supported(-Wno-error=asm-operand-widths) # Issue #784

    # -msse -mfpmath=sse to force gcc to use sse for float math,
    # avoiding excess precision problems that cause tests like int2float
    # to falsely fail. -ffloat-store also works, but WG suggested
    # that sse would be better.
    if(${CLConform_TARGET_ARCH} STREQUAL "x86_64" OR ${CLConform_TARGET_ARCH}
            STREQUAL "x86")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse -msse2 -mfpmath=sse")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -mfpmath=sse")

        add_cxx_flag_if_supported(-frounding-math)
    endif()
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D__SSE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE__")
endif()

# Set a module's COMPILE_FLAGS if using gcc or clang.
macro(set_gnulike_module_compile_flags flags)
    if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
        SET_SOURCE_FILES_PROPERTIES(
            ${${MODULE_NAME}_SOURCES}
            PROPERTIES
            COMPILE_FLAGS ${flags}
        )
    endif()
endmacro(set_gnulike_module_compile_flags)

# Xcode 14.1 deprecated functions such as sprintf.
# Suppress such warnings for now, see Issue #1626
if(APPLE)
    add_cxx_flag_if_supported(-Wno-deprecated-declarations)
endif(APPLE)

if(MSVC)
    # Don't warn when using standard non-secure functions.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Don't warn about using the portable "strdup" function.
    add_compile_definitions(_CRT_NONSTDC_NO_DEPRECATE)
    # Fix std::min and std::max handling with windows.harness.
    add_compile_definitions(NOMINMAX)
endif()

if( WIN32 AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qlong-double -Qpc80 /DWIN32 /D_WINDOWS /W3 /GR /EHsc -nologo -Od -D_CRT_NONSTDC_NO_WARNINGS -EHsc -Wall -Qdiag-disable:68,111,177,186,161,869,1028,2259,2553,181,239,265,1188 -fp:strict -fp:source")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Qlong-double -Qpc80 /DWIN32 /D_WINDOWS /W3 /GR /EHsc -nologo -Od -D_CRT_NONSTDC_NO_WARNINGS -EHsc -Wall -Qdiag-disable:68,111,177,186,161,869,1028,2259,2553,181,239,265,1188 -fp:strict -fp:source")
endif()

list(APPEND CLConform_LIBRARIES ${OPENCL_LIBRARIES})
if(ANDROID)
    list(APPEND CLConform_LIBRARIES m)
endif()
if(NOT DEFINED LINK_PTHREAD)
    if(ANDROID OR WIN32)
        set(LINK_PTHREAD OFF)
    else()
        set(LINK_PTHREAD ON)
    endif()
endif()
if(LINK_PTHREAD)
    list(APPEND CLConform_LIBRARIES pthread)
endif()

if(APPLE)
    find_library(corefoundation CoreFoundation)
    find_library(iokit IOKit)
    list(APPEND CLConform_LIBRARIES ${corefoundation})
    list(APPEND CLConform_LIBRARIES ${iokit})
endif(APPLE)

include_directories(SYSTEM ${CL_INCLUDE_DIR})
include_directories(${CLConform_SOURCE_DIR}/test_common/harness
                    ${CLConform_SOURCE_DIR}/test_common/gles
                    ${CLConform_SOURCE_DIR}/test_common/gl
                    ${CLConform_SOURCE_DIR}/test_common)

add_subdirectory(test_common)
add_subdirectory(test_conformance)
