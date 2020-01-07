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
)

include(../CMakeCommon.txt)
