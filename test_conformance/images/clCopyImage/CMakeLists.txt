set(MODULE_NAME CL_COPY_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_copy_1D.cpp
    test_copy_1D_array.cpp
    test_copy_2D.cpp
    test_copy_2D_2D_array.cpp
    test_copy_2D_3D.cpp
    test_copy_2D_array.cpp
    test_copy_3D.cpp
    test_copy_3D_2D_array.cpp
    test_copy_generic.cpp
    test_loops.cpp
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
    ../../../test_common/harness/crc32.c
)

include(../../CMakeCommon.txt)

