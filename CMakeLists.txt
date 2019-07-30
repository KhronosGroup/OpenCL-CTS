cmake_minimum_required(VERSION 3.1)

set( CONFORMANCE_SUFFIX "" )
set(CLConform_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

project(CLConform${CONFORMANCE_SUFFIX})

if(CMAKE_BUILD_TYPE STREQUAL "release")
    set (BUILD_FLAVOR "release")
else(CMAKE_BUILD_TYPE STREQUAL "release")
    set (BUILD_FLAVOR "debug")
endif(CMAKE_BUILD_TYPE STREQUAL "release")

set(CLConform_VERSION_MAJOR "2")
set(CLConform_VERSION_MINOR "0")
set(CLConform_VERSION_MICRO "0")
set(CLConform_VERSION_EXTRA "")
set(CLConform_VERSION "${CLConform_VERSION_MAJOR}.${CLConform_VERSION_MINOR}")
set(CLConform_VERSION_FULL
    "${CLConform_VERSION}.${CLConform_VERSION_MICRO}${CLConform_VERSION_EXTRA}")

add_definitions(-DCL_TARGET_OPENCL_VERSION=210)

add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_1_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_2_0_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_2_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_0_APIS=1)


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
# Vendor Customization
#-----------------------------------------------------------
#Vendor Customization File can be included here to provide a way to automatically
#build driver as a dependency of the conformance tests, or other such CMake customization
include(CMakeVendor.txt OPTIONAL)

 if(CL_INCLUDE_DIR AND CL_LIB_DIR)
     set(OPENCL_INCLUDE_DIR ${CL_INCLUDE_DIR})
     link_directories(${CL_LIB_DIR})
 else(CL_INCLUDE_DIR AND CL_LIB_DIR)
     message(STATUS "OpenCL hasn't been found!")
     message(FATAL_ERROR "Either install OpenCL or pass -DCL_INCLUDE_DIR and -DCL_LIB_DIR")
 endif(CL_INCLUDE_DIR AND CL_LIB_DIR)

include(CheckFunctionExists)
include(CheckIncludeFiles)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    # -msse -mfpmath=sse to force gcc to use sse for float math,
    # avoiding excess precision problems that cause tests like int2float
    # to falsely fail. -ffloat-store also works, but WG suggested
    # that sse would be better.
    if(CMAKE_ARM_COMPILER OR ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=gnu99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    else(CMAKE_ARM_COMPILER OR ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=gnu99 -msse -mfpmath=sse")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -msse -mfpmath=sse")
    endif(CMAKE_ARM_COMPILER OR ANDROID)
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D__SSE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE__")
endif()

# Clang gives C++11 narrowing warnings so suppress these for now
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing")
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

include_directories(SYSTEM ${OPENCL_INCLUDE_DIR})
include_directories(${CLConform_SOURCE_DIR}/test_common/harness
                    ${CLConform_SOURCE_DIR}/test_common/gles
                    ${CLConform_SOURCE_DIR}/test_common/gl
                    ${CMAKE_CURRENT_SOURCE_DIR}/test_common/harness)

if(CMAKE_BUILD_TYPE STREQUAL "release")
    set (BUILD_FLAVOR "release")
elseif (CMAKE_BUILD_TYPE STREQUAL "debug")
    set (BUILD_FLAVOR "debug")
endif(CMAKE_BUILD_TYPE STREQUAL "release")


add_subdirectory(test_conformance)

set (PY_PATH   "${CLConform_SOURCE_DIR}/test_conformance/*.py")
set (CSV_PATH  "${CLConform_SOURCE_DIR}/test_conformance/*.csv")
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
                     COMMAND ${COPY} ${PY_PATH} ${DST_DIR}
                     COMMAND ${COPY} ${CSV_PATH} ${DST_DIR}
                     COMMAND ${COPY} ${DLL_FILES} ${DST_DIR}
                     COMMENT "Copying other files to output folder..." )
else(WIN32)
  add_custom_target( COPY_FILES${CONFORMANCE_SUFFIX} )
endif(WIN32)

# Copy required CL include directories into the build directory
# as required for the compiler testing.

# ... For running the compiler test on the command line.
file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory" DESTINATION ${DST_DIR})
file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory" DESTINATION ${DST_DIR})

# ... For running the compiler test with VisualStudio.
if(MSVC)
  file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory" DESTINATION "${CLConform_SOURCE_DIR}/build/test_conformance/compiler")
  file(COPY "${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory" DESTINATION "${CLConform_SOURCE_DIR}/build/test_conformance/compiler")
endif(MSVC)

set_property(TARGET COPY_FILES${CONFORMANCE_SUFFIX} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")
add_subdirectory( "test_extensions" )
