function (install_spir_artifacts suite_name)
  install(FILES "${suite_name}.zip" DESTINATION "${CLConf_OUT_DIR}"
          COMPONENT OpenCLCTS)
endfunction()

add_executable(
    conformance_test_spir
    main.cpp
    datagen.cpp
    run_build_test.cpp
    run_services.cpp
    kernelargs.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/os_helpers.cpp
    ../../test_common/harness/testHarness.c
    ../../test_common/miniz/miniz.c
    ../../test_common/harness/crc32.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT MSVC)
set_source_files_properties(
        COMPILE_FLAGS -msse2)
endif()

if(UNIX)
    set_target_properties(conformance_test_spir PROPERTIES
	   COMPILE_FLAGS "-fexceptions -frtti")
elseif(MSVC)
    set_target_properties(conformance_test_spir PROPERTIES
       COMPILE_FLAGS "/GR /EHs /EHc")
endif()

TARGET_LINK_LIBRARIES(conformance_test_spir
	${CLConform_LIBRARIES})

# Need to copy the spir zips to sit beside the executable
add_custom_command(TARGET conformance_test_spir POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/khr.csv" "$<TARGET_FILE_DIR:conformance_test_spir>/khr.csv"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/api.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/api.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/atomics.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/atomics.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/basic.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/basic.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/compile_and_link.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/compile_and_link.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/commonfns.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/commonfns.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/conversions.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/conversions.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/geometrics.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/geometrics.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/enum_values.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/enum_values.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/half.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/half.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/kernel_attributes.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/kernel_attributes.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/kernel_image_methods.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/kernel_image_methods.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/images_kernel_read_write.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/images_kernel_read_write.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/images_samplerlessRead.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/images_samplerlessRead.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/integer_ops.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/integer_ops.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/math_brute_force.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/math_brute_force.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/printf.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/printf.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/profiling.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/profiling.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/relationals.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/relationals.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/select.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/select.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/sampler_enumeration.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/sampler_enumeration.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/vec_align.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/vec_align.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/vec_step.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/vec_step.zip"
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/binary_type.zip" "$<TARGET_FILE_DIR:conformance_test_spir>/binary_type.zip")

add_dependencies(OpenCLCTS conformance_test_spir)
install(TARGETS conformance_test_spir
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)

install_spir_artifacts(api)
install_spir_artifacts(atomics)
install_spir_artifacts(basic)
install_spir_artifacts(compile_and_link)
install_spir_artifacts(commonfns)
install_spir_artifacts(conversions)
install_spir_artifacts(geometrics)
install_spir_artifacts(enum_values)
install_spir_artifacts(half)
install_spir_artifacts(kernel_attributes)
install_spir_artifacts(kernel_image_methods)
install_spir_artifacts(images_kernel_read_write)
install_spir_artifacts(images_samplerlessRead)
install_spir_artifacts(integer_ops)
install_spir_artifacts(math_brute_force)
install_spir_artifacts(printf)
install_spir_artifacts(profiling)
install_spir_artifacts(relationals)
install_spir_artifacts(select)
install_spir_artifacts(sampler_enumeration)
install_spir_artifacts(vec_align)
install_spir_artifacts(vec_step)
install_spir_artifacts(binary_type)
install(FILES "khr.csv" DESTINATION "${CLConf_OUT_DIR}" COMPONENT OpenCLCTS)
#Add any other runtime directories you need here.
# end of file #
