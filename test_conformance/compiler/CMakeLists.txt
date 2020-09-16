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
    test_unload_platform_compiler.cpp
)

include(../CMakeCommon.txt)

# Copy required CL include directories into the binary directory

set(COMPILER_SOURCE_DIR ${CLConform_SOURCE_DIR}/test_conformance/compiler)
set(COMPILER_TARGET ${${MODULE_NAME}_OUT})

add_custom_command(
    TARGET  ${COMPILER_TARGET}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${COMPILER_SOURCE_DIR}/includeTestDirectory"
            $<TARGET_FILE_DIR:${COMPILER_TARGET}>/includeTestDirectory)

add_custom_command(
    TARGET  ${COMPILER_TARGET}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${COMPILER_SOURCE_DIR}/secondIncludeTestDirectory"
            $<TARGET_FILE_DIR:${COMPILER_TARGET}>/secondIncludeTestDirectory)
