set(MODULE_NAME NON_UNIFORM_WORK_GROUP)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_advanced_2d.cpp
    test_advanced_3d.cpp
    test_advanced_other.cpp
    test_basic.cpp
    TestNonUniformWorkGroup.cpp
    tools.cpp
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)

# end of file #
