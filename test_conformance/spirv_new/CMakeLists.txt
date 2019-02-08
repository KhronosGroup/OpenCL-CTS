######################################################################################################
#Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.
#
#This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
#This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
#third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
#broadcast or otherwise exploited in any manner without the express prior written permission
#of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
#disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
#in whole or in part other than under the terms of the Khronos Adopters Agreement
#or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
######################################################################################################

set(MODULE_NAME SPIRV_NEW)
set(CMAKE_CXX_STANDARD 11)

file(GLOB SPIRV_NEW_SOURCES "*.cpp")

add_executable(conformance_test_spirv_new
  ../../test_common/harness/errorHelpers.c
  ../../test_common/harness/threadTesting.c
  ../../test_common/harness/testHarness.c
  ../../test_common/harness/kernelHelpers.c
  ../../test_common/harness/typeWrappers.cpp
  ../../test_common/harness/mt19937.c
  ../../test_common/harness/conversions.c
  ../../test_common/harness/msvc9.c
  ../../test_common/harness/rounding_mode.c
  ../../test_common/harness/os_helpers.cpp
  ../../test_common/harness/crc32.c
  ../../test_common/harness/parseParameters.cpp
  ../../test_conformance/math_brute_force/reference_math.c
  ../../test_conformance/math_brute_force/Utility.c
  ${SPIRV_NEW_SOURCES})

set_source_files_properties(
  ../../test_common/harness/kernelHelpers.c
  ../../test_common/harness/testHarness.c
  ../../test_common/harness/conversions.c
  ../../test_conformance/math_brute_force/reference_math.c
  ../../test_conformance/math_brute_force/Utility.c
  PROPERTIES LANGUAGE CXX)

target_include_directories(conformance_test_spirv_new PRIVATE)

target_compile_definitions(conformance_test_spirv_new PRIVATE)

target_compile_options(conformance_test_spirv_new PRIVATE
  $<$<BOOL:${MSVC}>:/GR /EHs /EHc>)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
  target_compile_options(conformance_test_spirv_new PRIVATE -msse2)
endif()

target_link_libraries(conformance_test_spirv_new PRIVATE
  ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS conformance_test_spirv_new)

install(TARGETS conformance_test_spirv_new
  DESTINATION "${CLConf_OUT_DIR}" COMPONENT OpenCLCTS)

# TODO: Install SPIR-V files?
