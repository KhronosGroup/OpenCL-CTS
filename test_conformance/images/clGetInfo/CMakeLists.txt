set(MODULE_NAME CL_GET_INFO)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_1D_2D_array.cpp
    test_1D.cpp
    test_2D.cpp
    test_loops.cpp
    test_3D.cpp
    ../../../test_common/harness/errorHelpers.c
    ../../../test_common/harness/threadTesting.c
    ../../../test_common/harness/kernelHelpers.c
    ../../../test_common/harness/imageHelpers.cpp
    ../../../test_common/harness/mt19937.c
    ../../../test_common/harness/conversions.c
    ../../../test_common/harness/testHarness.c
    ../../../test_common/harness/typeWrappers.cpp
    ../../../test_common/harness/msvc9.c
)

include(../../CMakeCommon.txt)
