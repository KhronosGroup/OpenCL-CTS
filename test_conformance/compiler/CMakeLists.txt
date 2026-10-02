set(MODULE_NAME COMPILER)

find_package(Python3 COMPONENTS Interpreter QUIET)

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
    test_pragma_unroll.cpp
    test_unload_platform_compiler.cpp
    test_feature_macro.cpp
)

include(../CMakeCommon.txt)

# Determine the corresponding binary outputs to the SPV assembly input files
set(COMPILER_ASM_REL_PATH spirv_asm)
set(COMPILER_ASM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${COMPILER_ASM_REL_PATH}")
set(COMPILER_SPV_PATH "${CMAKE_CURRENT_BINARY_DIR}/spirv_bin")
set(COMPILER_ASM_FILES
    write_kernel.spvasm32
    write_kernel.spvasm64)

set(COMPILER_ASM "")
set(COMPILER_SPV "")
foreach(COMPILER_ASM_FILE ${COMPILER_ASM_FILES})
    list(APPEND COMPILER_ASM "${COMPILER_ASM_PATH}/${COMPILER_ASM_FILE}")
    string(REPLACE ".spvasm" ".spv" COMPILER_SPV_FILE ${COMPILER_ASM_FILE})
    list(APPEND COMPILER_SPV "${COMPILER_SPV_PATH}/${COMPILER_SPV_FILE}")
endforeach()

# Copy the required test include directories into the build directory.
if(NOT DEFINED COMPILER_TEST_RESOURCES)
    set(COMPILER_TEST_RESOURCES $<TARGET_FILE_DIR:${${MODULE_NAME}_OUT}>)
endif()

set(COMPILER_SPV_EXTRA "")
if(SPIRV_TOOLS_DIR AND IS_ABSOLUTE "${SPIRV_TOOLS_DIR}" AND
    IS_DIRECTORY "${SPIRV_TOOLS_DIR}")
    message("Using SPIR-V tools from '${SPIRV_TOOLS_DIR}'")
    set(COMPILER_SPV_EXTRA "--assembler=${SPIRV_TOOLS_DIR}/spirv-as" "--validator=${SPIRV_TOOLS_DIR}/spirv-val")
endif()
set(COMPILER_ASSEMBLY_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../spirv_new/spirv_asm/assemble_spirv.py)

include(CMakePrintHelpers)
cmake_print_variables(COMPILER_ASSEMBLY_SCRIPT)

add_custom_command(
        COMMENT "Copying compiler test resources..."
        TARGET ${${MODULE_NAME}_OUT}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory
        ${COMPILER_TEST_RESOURCES}/includeTestDirectory
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory
        ${COMPILER_TEST_RESOURCES}/secondIncludeTestDirectory
        VERBATIM)

add_custom_command(
        OUTPUT ${COMPILER_SPV}
        COMMENT "Generating compiler SPIR-V binaries..."
        COMMAND ${Python3_EXECUTABLE} ${COMPILER_ASSEMBLY_SCRIPT} --source-dir "${COMPILER_ASM_PATH}" --output-dir "${COMPILER_SPV_PATH}" ${COMPILER_SPV_EXTRA} --verbose
        DEPENDS ${COMPILER_ASSEMBLY_SCRIPT} ${COMPILER_ASM}
        USES_TERMINAL
        VERBATIM)

add_custom_target(compiler_spirv_binaries DEPENDS ${COMPILER_SPV})
add_dependencies(${${MODULE_NAME}_OUT} compiler_spirv_binaries)

include(GNUInstallDirs)

install(DIRECTORY
            ${CLConform_SOURCE_DIR}/test_conformance/compiler/includeTestDirectory
            ${CLConform_SOURCE_DIR}/test_conformance/compiler/secondIncludeTestDirectory
        DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)

install(DIRECTORY
            ${COMPILER_SPV_PATH}
        DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
