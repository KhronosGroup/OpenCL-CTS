function (install_spir_artifacts suite_name)
  install(FILES "${suite_name}.zip" DESTINATION ${CLConf_OUT_DIR}/spir)
endfunction()

include_directories(${CLConf_SRC_DIR}/test_common)

# Import function list from math_brute_force
add_definitions(-DFUNCTION_LIST_ULPS_ONLY)

clconf_add_executable(
    test_spir
    main.cpp
    datagen.cpp
    run_build_test.cpp
    run_services.cpp
    kernelargs.cpp
    ../math_brute_force/FunctionList.c)

target_link_libraries(
    test_spir${CLConf_SUFFIX}
    miniz${CLConf_SUFFIX}
)

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
install(FILES "khr.csv" DESTINATION ${CLConf_OUT_DIR}/spir)
#Add any other runtime directories you need here.
# end of file #
