set(MODULE_NAME SPIRV_NEW)

file(GLOB SPIRV_NEW_SOURCES "*.cpp")

set(TEST_HARNESS_SOURCES
  ../../test_conformance/math_brute_force/reference_math.cpp
  ../../test_conformance/math_brute_force/utility.cpp
)

set(${MODULE_NAME}_SOURCES ${SPIRV_NEW_SOURCES} ${TEST_HARNESS_SOURCES})

include(../CMakeCommon.txt)
