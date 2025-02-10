find_program(
    Vulkan_glslang_binary
    NAMES glslang
    HINTS $ENV{VULKAN_SDK})

if(${Vulkan_glslang_binary} STREQUAL "Vulkan_glslang_binary-NOTFOUND")
    message(FATAL_ERROR "glslang tool (part of Vulkan SDK) is a prerequisite to compile shaders to spir-v")
else()
    message(STATUS "Found glslang: ${Vulkan_glslang_binary}")

    if(NOT DEFINED VULKAN_TEST_RESOURCES)
        set(VULKAN_TEST_RESOURCES ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    set(vulkan_spirv_files "")

    add_custom_command(
      OUTPUT ${VULKAN_TEST_RESOURCES}/buffer.spv
      COMMAND ${Vulkan_glslang_binary}
              --target-env vulkan1.0
              -o ${VULKAN_TEST_RESOURCES}/buffer.spv
              ${CMAKE_CURRENT_SOURCE_DIR}/buffer.comp
      DEPENDS buffer.comp
      VERBATIM)
    list(APPEND vulkan_spirv_files ${VULKAN_TEST_RESOURCES}/buffer.spv)

    file(STRINGS "image2D_test_formats.txt" image2D_formats)

    foreach(format ${image2D_formats})
        list(GET format 1 GLSL_FORMAT)
        list(GET format 2 GLSL_TYPE_PREFIX)

        configure_file(image2D.comp.in image2D_${GLSL_FORMAT}.comp)

        add_custom_command(
          OUTPUT ${VULKAN_TEST_RESOURCES}/image2D_${GLSL_FORMAT}.spv
          COMMAND ${Vulkan_glslang_binary}
                  --target-env vulkan1.0
                  -o ${VULKAN_TEST_RESOURCES}/image2D_${GLSL_FORMAT}.spv
                  ${CMAKE_CURRENT_BINARY_DIR}/image2D_${GLSL_FORMAT}.comp
          DEPENDS image2D_${GLSL_FORMAT}.comp
          VERBATIM)
        list(APPEND vulkan_spirv_files ${VULKAN_TEST_RESOURCES}/image2D_${GLSL_FORMAT}.spv)
    endforeach()

    add_custom_target(vulkan_shaders DEPENDS ${vulkan_spirv_files})
endif()
