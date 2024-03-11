# CMP0007:NEW - Don't ignore empty elements in a list
cmake_policy(SET CMP0007 NEW)
find_program(
    Vulkan_glslang_binary
    NAMES glslang
    HINTS $ENV{VULKAN_SDK})

if(${Vulkan_glslang_binary} STREQUAL "Vulkan_glslang_binary-NOTFOUND")
    message(FATAL_ERROR "glslang tool (part of Vulkan SDK) is a prerequisite to compile shaders to spir-v")
else()
    message(STATUS "Found glslang: ${Vulkan_glslang_binary}")

    set(IMAGE2D_SHADER_IN_FILE "image2D.comp")
    set(IMAGE2D_SHADER_TMP_OUT_FILE "tmp_image2D.comp")
    set(BUFFER_SHADER_IN_FILE "buffer")
    set(IMAGE2D_FORMATS_LIST_IN_FILE "image2D_test_formats.txt")

    file(READ ${IMAGE2D_SHADER_IN_FILE} IMAGE2D_SHADER_UNFORMAT_CONTENT)
    file(STRINGS ${IMAGE2D_FORMATS_LIST_IN_FILE} IMAGE2D_FORMATS_LIST)

    foreach(IMAGE2D_FORMAT ${IMAGE2D_FORMATS_LIST})
        list(GET IMAGE2D_FORMAT 1 GLSL_FORMAT)
        list(GET IMAGE2D_FORMAT 2 GLSL_TYPE_PREFIX)
        string(REPLACE "GLSL_FORMAT" "${GLSL_FORMAT}" IMAGE2D_SHADER_CONTENT "${IMAGE2D_SHADER_UNFORMAT_CONTENT}")
        string(REPLACE "GLSL_TYPE_PREFIX" "${GLSL_TYPE_PREFIX}" IMAGE2D_SHADER_CONTENT "${IMAGE2D_SHADER_CONTENT}")
        file(WRITE ${IMAGE2D_SHADER_TMP_OUT_FILE} "${IMAGE2D_SHADER_CONTENT}")
        execute_process(
            COMMAND ${Vulkan_glslang_binary} --target-env vulkan1.0 -o image2D_${GLSL_FORMAT}.spv ${IMAGE2D_SHADER_TMP_OUT_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE commandStatus
            OUTPUT_QUIET)
        if(commandStatus AND NOT commandStatus EQUAL 0)
            message(FATAL_ERROR "shader -> spir-v compilation failed")
        endif()
    endforeach(IMAGE2D_FORMAT)
    execute_process(
        COMMAND ${Vulkan_glslang_binary} --target-env vulkan1.0 -o ${BUFFER_SHADER_IN_FILE}.spv ${BUFFER_SHADER_IN_FILE}.comp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE commandStatus
        OUTPUT_QUIET)
    if(commandStatus AND NOT commandStatus EQUAL 0)
        message(FATAL_ERROR "shader -> spir-v compilation failed")
    endif()
    file(REMOVE ${IMAGE2D_SHADER_TMP_OUT_FILE})
endif()