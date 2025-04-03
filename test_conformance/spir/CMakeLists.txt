set(MODULE_NAME SPIR)

set (${MODULE_NAME}_SOURCES
    main.cpp
    datagen.cpp
    run_build_test.cpp
    run_services.cpp
    kernelargs.cpp
)

include(../CMakeCommon.txt)

if(UNIX)
    target_compile_options(${${MODULE_NAME}_OUT} PRIVATE -fexceptions -frtti)
elseif(MSVC)
    target_compile_options(${${MODULE_NAME}_OUT} PRIVATE /GR /EHs /EHc)
endif()

# Need to copy the spir zips to sit beside the executable

set(SPIR_FILES
    "khr.csv"
    "api.zip"
    "atomics.zip"
    "basic.zip"
    "compile_and_link.zip"
    "commonfns.zip"
    "conversions.zip"
    "geometrics.zip"
    "enum_values.zip"
    "half.zip"
    "kernel_attributes.zip"
    "kernel_image_methods.zip"
    "images_kernel_read_write.zip"
    "images_samplerlessRead.zip"
    "integer_ops.zip"
    "math_brute_force.zip"
    "printf.zip"
    "profiling.zip"
    "relationals.zip"
    "select.zip"
    "sampler_enumeration.zip"
    "vec_align.zip"
    "vec_step.zip"
    "binary_type.zip")

foreach(input_file ${SPIR_FILES})
    add_custom_command(
        TARGET ${${MODULE_NAME}_OUT}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${input_file}" "$<TARGET_FILE_DIR:${${MODULE_NAME}_OUT}>/"
    )
endforeach(input_file)

install(FILES ${SPIR_FILES} DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
