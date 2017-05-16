
set(MODULE_NAME CL_FILL_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_fill_1D.cpp
    test_fill_1D_array.cpp
    test_fill_2D.cpp
    test_fill_2D_array.cpp
    test_fill_generic.cpp
    test_loops.cpp
    test_fill_3D.cpp
#    test_fill_2D_3D.cpp
    ../../../test_common/harness/testHarness.c
    ../../../test_common/harness/errorHelpers.c
    ../../../test_common/harness/threadTesting.c
    ../../../test_common/harness/kernelHelpers.c
    ../../../test_common/harness/imageHelpers.cpp
    ../../../test_common/harness/mt19937.c
    ../../../test_common/harness/conversions.c
    ../../../test_common/harness/typeWrappers.cpp
    ../../../test_common/harness/msvc9.c
    ../../../test_common/harness/parseParameters.cpp
)


include(../../CMakeCommon.txt)
