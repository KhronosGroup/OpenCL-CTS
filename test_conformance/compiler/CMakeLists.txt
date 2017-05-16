set(MODULE_NAME COMPILER)

set(${MODULE_NAME}_SOURCES
    main.c
    test_build_helpers.c
    test_compile.c
    test_async_build.c
    test_build_options.cpp
    test_preprocessor.c
    test_image_macro.c
    test_compiler_defines_for_extensions.cpp
    test_pragma_unroll.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/os_helpers.cpp
)

include(../CMakeCommon.txt)
