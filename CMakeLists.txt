cmake_minimum_required(VERSION 3.1)

set( CONFORMANCE_SUFFIX "" )
set(CLConform_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

project(CLConform${CONFORMANCE_SUFFIX})

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "release")
    set (BUILD_FLAVOR "release")
else(CMAKE_BUILD_TYPE STREQUAL "release")
    set (BUILD_FLAVOR "debug")
endif(CMAKE_BUILD_TYPE STREQUAL "release")

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_1_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_0_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_2_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_0_APIS=1)

option(USE_CL_EXPERIMENTAL "Use Experimental definitions" OFF)
if(USE_CL_EXPERIMENTAL)
  add_definitions(-DCL_EXPERIMENTAL)
endif(USE_CL_EXPERIMENTAL)

# Support both VS2008 and VS2012.
set(BUILD_DIR "$ENV{ADRENO_DRIVER}/build")
if(MSVC90)
  set(VS_BUILD_DIR "${BUILD_DIR}/vs2008")
else(MSVC110)
  set(VS_BUILD_DIR "${BUILD_DIR}/vs2012")
endif(MSVC90)

#-----------------------------------------------------------
# Default Configurable Test Set
#-----------------------------------------------------------
option(D3D10_IS_SUPPORTED "Run DirectX 10 interop tests" OFF)
option(D3D11_IS_SUPPORTED "Run DirectX 11 interop tests" OFF)
option(GL_IS_SUPPORTED "Run OpenGL interop tests" OFF)
option(GLES_IS_SUPPORTED "Run OpenGL ES interop tests" OFF)


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
option(USE_VENDOR_CUSTOM_FILE "Use Vendor Customization File" OFF)
if(USE_VENDOR_CUSTOM_FILE)
    include(CMakeVendor.txt OPTIONAL)
endif(USE_VENDOR_CUSTOM_FILE)

#-----------------------------------------------------------
# Development options for OpenCL C++ tests
#-----------------------------------------------------------
# Use OpenCL C kernels instead of OpenCL C++ kernels
option(CLPP_DEVELOPMENT_USE_OPENCLC_KERNELS "Use OpenCL C kernels in OpenCL C++ tests" OFF)
if(CLPP_DEVELOPMENT_USE_OPENCLC_KERNELS)
  set(CLPP_DEVELOPMENT_OPTIONS ${CLPP_DEVELOPMENT_OPTIONS} -DCLPP_DEVELOPMENT_USE_OPENCLC_KERNELS)
endif(CLPP_DEVELOPMENT_USE_OPENCLC_KERNELS)
# Only check if OpenCL C++ kernels compile to SPIR-V
option(CLPP_DEVELOPMENT_ONLY_SPIRV_COMPILATION "Only check if OpenCL C++ kernels compile to SPIR-V" OFF)
if(CLPP_DEVELOPMENT_ONLY_SPIRV_COMPILATION)
  if(CLPP_DEVELOPMENT_USE_OPENCLC_KERNELS)
    message(FATAL_ERROR "Can't use OpenCL C kernels and compile to SPIR-V.")
  endif(CLPP_DEVELOPMENT_USE_OPENCLC_KERNELS)
  set(CLPP_DEVELOPMENT_OPTIONS ${CLPP_DEVELOPMENT_OPTIONS} -DCLPP_DEVELOPMENT_ONLY_SPIRV_COMPILATION)
endif(CLPP_DEVELOPMENT_ONLY_SPIRV_COMPILATION)
#
if(CLPP_DEVELOPMENT_OPTIONS)
  add_definitions(-DCLPP_DEVELOPMENT_OPTIONS)
  add_definitions(${CLPP_DEVELOPMENT_OPTIONS})
endif(CLPP_DEVELOPMENT_OPTIONS)

# Offline OpenCL C/C++ compiler provided by Khronos is the only supported
# offline compiler.
#
# Path to offline OpenCL C/C++ compiler provided by Khronos.
# See https://github.com/KhronosGroup/SPIR/ (spirv-1.1 branch or newer SPIR-V-ready
# branch should be used).
if(KHRONOS_OFFLINE_COMPILER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DKHRONOS_OFFLINE_COMPILER=${KHRONOS_OFFLINE_COMPILER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKHRONOS_OFFLINE_COMPILER=${KHRONOS_OFFLINE_COMPILER}")
    # Additional OpenCL C/C++ compiler option.
    if(KHRONOS_OFFLINE_COMPILER_OPTIONS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DKHRONOS_OFFLINE_COMPILER_OPTIONS=${KHRONOS_OFFLINE_COMPILER_OPTIONS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKHRONOS_OFFLINE_COMPILER_OPTIONS=${KHRONOS_OFFLINE_COMPILER_OPTIONS}")
    endif(KHRONOS_OFFLINE_COMPILER_OPTIONS)
else(KHRONOS_OFFLINE_COMPILER)
    message(WARNING "KHRONOS_OFFLINE_COMPILER is not defined!")
    message(WARNING "Running CL C++ tests will not be possible.")
endif(KHRONOS_OFFLINE_COMPILER)

# CL_LIBCLCXX_DIR - path to dir with OpenCL C++ STL (libclcxx)
# CL_INCLUDE_DIR - path to dir with OpenCL headers
# CL_LIBCLCXX_DIR - path to dir with OpenCL library
if(CL_INCLUDE_DIR AND CL_LIB_DIR AND CL_LIBCLCXX_DIR)
    link_directories(${CL_LIB_DIR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCL_LIBCLCXX_DIR=${CL_LIBCLCXX_DIR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCL_LIBCLCXX_DIR=${CL_LIBCLCXX_DIR}")
else(CL_INCLUDE_DIR AND CL_LIB_DIR AND CL_LIBCLCXX_DIR)
    message(STATUS "OpenCL hasn't been found!")
    message(FATAL_ERROR "Either install OpenCL or pass -DCL_INCLUDE_DIR, -DCL_LIB_DIR and -DCL_LIBCLCXX_DIR")
endif(CL_INCLUDE_DIR AND CL_LIB_DIR AND CL_LIBCLCXX_DIR)

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
    check_cxx_compiler_flag(${flag} COMPILER_SUPPORTS_${FLAG_NO_SIGNS})
    if(COMPILER_SUPPORTS_${FLAG_NO_SIGNS})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
    endif()
endmacro(add_cxx_flag_if_supported)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    add_cxx_flag_if_supported(-Wno-narrowing)
    add_cxx_flag_if_supported(-Wno-format)
    add_cxx_flag_if_supported(-Werror)
    add_cxx_flag_if_supported(-Wno-error=cpp) # Allow #warning directive
    add_cxx_flag_if_supported(-Wno-error=absolute-value) # Issue 783
    add_cxx_flag_if_supported(-Wno-error=unknown-pragmas) # Issue #785
    add_cxx_flag_if_supported(-Wno-error=asm-operand-widths) # Issue #784
    add_cxx_flag_if_supported(-Wno-error=overflow) # Fixed by #699

    # -msse -mfpmath=sse to force gcc to use sse for float math,
    # avoiding excess precision problems that cause tests like int2float
    # to falsely fail. -ffloat-store also works, but WG suggested
    # that sse would be better.
    if(${CLConform_TARGET_ARCH} STREQUAL "x86_64" OR ${CLConform_TARGET_ARCH}
            STREQUAL "x86")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse -msse2 -mfpmath=sse")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -mfpmath=sse")
    endif()
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D__SSE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE__")
endif()

if( WIN32 AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qlong-double -Qpc80 /DWIN32 /D_WINDOWS /W3 /GR /EHsc -nologo -Od -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS -EHsc -Wall -Qdiag-disable:68,111,177,186,161,869,1028,2259,2553,181,239,265,1188 -fp:strict -fp:source")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Qlong-double -Qpc80 /DWIN32 /D_WINDOWS /W3 /GR /EHsc -nologo -Od -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS -EHsc -Wall -Qdiag-disable:68,111,177,186,161,869,1028,2259,2553,181,239,265,1188 -fp:strict -fp:source")
endif()

list(APPEND CLConform_LIBRARIES ${OPENCL_LIBRARIES})
if(ANDROID)
    list(APPEND CLConform_LIBRARIES m)
elseif(NOT WIN32)
    list(APPEND CLConform_LIBRARIES pthread)
endif(ANDROID)

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

if(CMAKE_BUILD_TYPE STREQUAL "release")
    set (BUILD_FLAVOR "release")
elseif (CMAKE_BUILD_TYPE STREQUAL "debug")
    set (BUILD_FLAVOR "debug")
endif(CMAKE_BUILD_TYPE STREQUAL "release")


add_subdirectory(test_common)
add_subdirectory(test_conformance)
add_subdirectory(test_extensions)

# Support both VS2008 and VS2012.
set (DLL_FILES "${VS_BUILD_DIR}/Debug/*.dll")
set (DST_DIR   "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/")

if (WIN32)
    set (COPY "echo")
    add_custom_target(COPY_DLL${CONFORMANCE_SUFFIX} ALL
                      COMMAND ${COPY} "${DLL_FILES}" "${DST_DIR}"
                      COMMENT "Copying dll files.. ")
else (WIN32)
    set (COPY cp)
    add_custom_target(COPY_DLL${CONFORMANCE_SUFFIX})
endif(WIN32)

set_property(TARGET COPY_DLL${CONFORMANCE_SUFFIX} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")

if(WIN32)
  add_custom_target( COPY_FILES${CONFORMANCE_SUFFIX} ALL
                     COMMAND ${COPY} ${DLL_FILES} ${DST_DIR}
                     COMMENT "Copying other files to output folder..." )
else(WIN32)
  add_custom_target( COPY_FILES${CONFORMANCE_SUFFIX} )
endif(WIN32)

# Copy required CL include directories into the build directory
# as required for the compiler testing.

# ... For running the compiler test on the command line.
file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_conformance/compiler)
file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_conformance/compiler)

# ... For running the compiler test with VisualStudio.
if(MSVC)
  file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory" DESTINATION "${CLConform_SOURCE_DIR}/build/test_conformance/compiler")
  file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory" DESTINATION "${CLConform_SOURCE_DIR}/build/test_conformance/compiler")
endif(MSVC)

set_property(TARGET COPY_FILES${CONFORMANCE_SUFFIX} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")
