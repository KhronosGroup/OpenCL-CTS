set(MODULE_NAME VECSTEP)

set(${MODULE_NAME}_SOURCES
    globals.c
        test_step.c
        main.c
        structs.c
        type_replacer.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/crc32.c
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
