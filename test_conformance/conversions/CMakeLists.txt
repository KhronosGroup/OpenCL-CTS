set(MODULE_NAME CONVERSIONS)

set (${MODULE_NAME}_SOURCES
      Sleep.cpp test_conversions.cpp basic_test_conversions.cpp
)

if("${CLConform_TARGET_ARCH}" STREQUAL "ARM" OR "${CLConform_TARGET_ARCH}" STREQUAL "ARM64")
    list(APPEND ${MODULE_NAME}_SOURCES fplib.cpp)
endif()

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../CMakeCommon.txt)
