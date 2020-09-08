######################################################################################################
#Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.
#
#This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
#This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
#third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
#broadcast or otherwise exploited in any manner without the express prior written permission
#of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
#disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
#in whole or in part other than under the terms of the Khronos Adopters Agreement
#or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
######################################################################################################

set(MODULE_NAME SPIRV_NEW)
set(CMAKE_CXX_STANDARD 11)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_cl_khr_spirv_no_integer_wrap_decoration.cpp
    test_decorate.cpp
    test_get_program_il.cpp
    test_linkage.cpp
    test_op_atomic.cpp
    test_op_branch_conditional.cpp
    test_op_branch.cpp
    test_op_composite_construct.cpp
    test_op_constant.cpp
    test_op_copy_object.cpp
    test_op_fmath.cpp
    test_op_function.cpp
    test_op_lifetime.cpp
    test_op_loop_merge.cpp
    test_op_negate.cpp
    test_op_opaque.cpp
    test_op_phi.cpp
    test_op_selection_merge.cpp
    test_op_spec_constant.cpp
    test_op_undef.cpp
    test_op_vector_extract.cpp
    test_op_vector_insert.cpp
    test_op_vector_times_scalar.cpp
)

set(TEST_HARNESS_SOURCES
  ../../test_conformance/math_brute_force/reference_math.cpp
  ../../test_conformance/math_brute_force/utility.cpp
)

list(APPEND ${MODULE_NAME}_SOURCES ${TEST_HARNESS_SOURCES})

include(../CMakeCommon.txt)

# Include a list of the relative paths to SPV assembly files in SPIRV_NEW_ASM
configure_file(spirv_asm_list.txt ${CMAKE_CURRENT_BINARY_DIR}/spirv_asm_list.txt)
include(${CMAKE_CURRENT_BINARY_DIR}/spirv_asm_list.txt)

# Determine the corresponding binary outputs to the SPV assembly input files
set(SPIRV_NEW_ASM_REL_PATH spirv_asm)
set(SPIRV_NEW_ASM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SPIRV_NEW_ASM_REL_PATH}")
set(SPIRV_NEW_SPV_PATH "${CMAKE_CURRENT_BINARY_DIR}/spirv_bin")

function(GET_SPV_BINS SPV_ASM SPV_BIN)
    set(BIN_LIST "")
    foreach(ASM ${${SPV_ASM}})
        string(REPLACE "${SPIRV_NEW_ASM_REL_PATH}" "${SPIRV_NEW_SPV_PATH}" BIN ${ASM})
        string(REPLACE ".spvasm" ".spv" BIN "${BIN}")
        list(APPEND BIN_LIST ${BIN})
    endforeach(ASM)
    SET(${SPV_BIN} "${BIN_LIST}" PARENT_SCOPE)
endfunction(GET_SPV_BINS)

get_spv_bins(SPIRV_NEW_ASM SPIRV_NEW_SPV)

# Assemble the SPIR-V binaries from the source files using the SPIR-V tools
option(SPIRV_TOOLS_DIR "Directory of SPIR-V tools to use, in preference to any found in PATH" "")

set(SPIRV_NEW_SPV_EXTRA "")
if(SPIRV_TOOLS_DIR AND IS_ABSOLUTE "${SPIRV_TOOLS_DIR}" AND
    IS_DIRECTORY "${SPIRV_TOOLS_DIR}")
    message("Using SPIR-V tools from '${SPIRV_TOOLS_DIR}'")
    set(SPIRV_NEW_SPV_EXTRA "--assembler=${SPIRV_TOOLS_DIR}/spirv-as" "--validator=${SPIRV_TOOLS_DIR}/spirv-val")
endif()

set(SPIRV_NEW_ASSEMBLY_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/assemble_spirv.py)

add_custom_command(
    OUTPUT  ${SPIRV_NEW_SPV}
    COMMAND ${SPIRV_NEW_ASSEMBLY_SCRIPT} --source-dir "${SPIRV_NEW_ASM_PATH}"
            --output-dir "${SPIRV_NEW_SPV_PATH}" ${SPIRV_NEW_SPV_EXTRA} --verbose
    DEPENDS ${SPIRV_NEW_ASSEMBLY_SCRIPT} ${SPIRV_NEW_ASM}
    VERBATIM)

add_custom_target(SPIRV_NEW_SPV_BINS
    DEPENDS ${SPIRV_NEW_SPV})

add_dependencies(${${MODULE_NAME}_OUT} SPIRV_NEW_SPV_BINS)
