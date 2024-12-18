set(MODULE_NAME CONVERSIONS)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    add_cxx_flag_if_supported(-Wno-narrowing)
endif()

set (${MODULE_NAME}_SOURCES
      Sleep.cpp test_conversions.cpp basic_test_conversions.cpp
)

if("${CLConform_TARGET_ARCH}" STREQUAL "ARM" OR "${CLConform_TARGET_ARCH}" STREQUAL "ARM64")
    list(APPEND ${MODULE_NAME}_SOURCES fplib.cpp)
endif()

set_gnulike_module_compile_flags("-Wno-sign-compare -Wno-format")

include(../CMakeCommon.txt)
