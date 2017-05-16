set(MODULE_NAME VECALIGN)

set(${MODULE_NAME}_SOURCES
    globals.c
        main.c
        structs.c
        test_vec_align.c
        type_replacer.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/conversions.c
        ../../test_common/harness/parseParameters.cpp
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)

