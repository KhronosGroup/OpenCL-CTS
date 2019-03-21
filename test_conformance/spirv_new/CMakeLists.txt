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

set(TEST_HARNESS_SOURCES
  ../../test_common/harness/crc32.c
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
  ../../test_common/harness/parseParameters.cpp
  ../../test_conformance/math_brute_force/reference_math.c
  ../../test_conformance/math_brute_force/Utility.c
  )

set(${MODULE_NAME}_SOURCES ${SPIRV_NEW_SOURCES} ${TEST_HARNESS_SOURCES})

include(../CMakeCommon.txt)
