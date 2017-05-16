set(MODULE_NAME GEOMETRICS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_geometrics_double.cpp
    test_geometrics.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
)

# VS2005 optimization WAR for geom_length)
if(MSVC)
set(CMAKE_C_FLAGS_RELEASE "/Od /Ob0")
set(CMAKE_CXX_FLAGS_RELEASE "/Od /Ob0")
endif(MSVC)

include(../CMakeCommon.txt)

