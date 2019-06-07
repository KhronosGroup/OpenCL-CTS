# Import function list from math_brute_force
add_definitions(-DFUNCTION_LIST_ULPS_ONLY)

set(SPIR_OUT ${CONFORMANCE_PREFIX}spir${CONFORMANCE_SUFFIX})

set (SPIR_SOURCES
    main.cpp
    datagen.cpp
    run_build_test.cpp
    run_services.cpp
    kernelargs.cpp
    ../../test_common/harness/parseParameters.cpp
    ../math_brute_force/FunctionList.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/os_helpers.cpp
    ../../test_common/harness/testHarness.c
    ../../test_common/miniz/miniz.c)

add_executable(${SPIR_OUT}
    ${SPIR_SOURCES})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

if(UNIX)
    set_target_properties(${SPIR_OUT} PROPERTIES
       COMPILE_FLAGS "-fexceptions -frtti")
elseif(MSVC)
    set_target_properties(${SPIR_OUT} PROPERTIES
       COMPILE_FLAGS "/GR /EHs /EHc")
endif()

TARGET_LINK_LIBRARIES(${SPIR_OUT}
    ${CLConform_LIBRARIES})


set_source_files_properties(${SPIR_SOURCES} PROPERTIES LANGUAGE CXX)

# Need to copy the spir zips to sit beside the executable
add_custom_command(TARGET ${SPIR_OUT} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/khr.csv" "$<TARGET_FILE_DIR:${SPIR_OUT}>/khr.csv"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/api.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/api.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/atomics.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/atomics.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/basic.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/basic.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/compile_and_link.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/compile_and_link.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/commonfns.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/commonfns.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/conversions.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/conversions.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/geometrics.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/geometrics.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/enum_values.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/enum_values.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/half.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/half.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/kernel_attributes.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/kernel_attributes.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/kernel_image_methods.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/kernel_image_methods.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/images_kernel_read_write.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/images_kernel_read_write.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/images_samplerlessRead.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/images_samplerlessRead.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/integer_ops.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/integer_ops.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/math_brute_force.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/math_brute_force.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/printf.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/printf.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/profiling.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/profiling.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/relationals.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/relationals.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/select.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/select.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/sampler_enumeration.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/sampler_enumeration.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/vec_align.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/vec_align.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/vec_step.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/vec_step.zip"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/binary_type.zip" "$<TARGET_FILE_DIR:${SPIR_OUT}>/binary_type.zip")
