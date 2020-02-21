set(MODULE_NAME COMPILER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_build_helpers.cpp
    test_compile.cpp
    test_async_build.cpp
    test_build_options.cpp
    test_preprocessor.cpp
    test_image_macro.cpp
    test_compiler_defines_for_extensions.cpp
    test_pragma_unroll.cpp
)

include(../CMakeCommon.txt)
