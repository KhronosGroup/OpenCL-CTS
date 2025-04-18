set(MODULE_NAME COMPILER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_build_helpers.cpp
    test_compile.cpp
    test_async_build.cpp
    test_build_options.cpp
    test_preprocessor.cpp
    test_opencl_c_versions.cpp
    test_image_macro.cpp
    test_compiler_defines_for_extensions.cpp
    test_compiler_defines_for_extensions_new.cpp
    test_pragma_unroll.cpp
    test_unload_platform_compiler.cpp
    test_feature_macro.cpp
)

include(../CMakeCommon.txt)

# Copy the required test include directories into the build directory.
if(NOT DEFINED COMPILER_TEST_RESOURCES)
    set(COMPILER_TEST_RESOURCES $<TARGET_FILE_DIR:${${MODULE_NAME}_OUT}>)
endif()
add_custom_command(
        COMMENT "Copying compiler test resources..."
        TARGET ${${MODULE_NAME}_OUT}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory
        ${COMPILER_TEST_RESOURCES}/includeTestDirectory
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory
        ${COMPILER_TEST_RESOURCES}/secondIncludeTestDirectory)

include(GNUInstallDirs)

install(DIRECTORY
            ${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory
            ${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory
        DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
