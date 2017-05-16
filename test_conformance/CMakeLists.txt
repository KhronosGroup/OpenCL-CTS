# Remember curren source directory (`test_conformance').
set( CLConf_Install_Base_Dir "${CMAKE_CURRENT_SOURCE_DIR}" )

add_subdirectory( allocations )
add_subdirectory( api )
add_subdirectory( atomics )
add_subdirectory( basic )
add_subdirectory( buffers )
add_subdirectory( commonfns )
add_subdirectory( compatibility )
add_subdirectory( compiler )
add_subdirectory( computeinfo )
add_subdirectory( contractions )
add_subdirectory( conversions )
if(D3D10_IS_SUPPORTED)
   add_subdirectory( d3d10 )
endif(D3D10_IS_SUPPORTED)
if(D3D11_IS_SUPPORTED)
   add_subdirectory( d3d11 )
endif(D3D11_IS_SUPPORTED)
add_subdirectory( device_partition )
add_subdirectory( events )
add_subdirectory( geometrics )
if(GL_IS_SUPPORTED)
   add_subdirectory( gl )
endif(GL_IS_SUPPORTED)
if(GLES_IS_SUPPORTED)
   add_subdirectory(gles)
endif(GLES_IS_SUPPORTED)
add_subdirectory( half )
add_subdirectory( headers )
add_subdirectory( images )
add_subdirectory( integer_ops )
add_subdirectory( math_brute_force )
add_subdirectory( mem_host_flags )
add_subdirectory( multiple_device_context )
add_subdirectory( printf )
add_subdirectory( profiling )
add_subdirectory( relationals )
add_subdirectory( select )
add_subdirectory( thread_dimensions )
add_subdirectory( vec_align )
add_subdirectory( vec_step )
add_subdirectory( c11_atomics )
add_subdirectory( device_execution )
add_subdirectory( non_uniform_work_group )
add_subdirectory( SVM )
add_subdirectory( generic_address_space )
add_subdirectory( subgroups )
add_subdirectory( workgroups )
add_subdirectory( pipes ) 
add_subdirectory( device_timer )

install_files(
    opencl_conformance_tests_conversions.csv
    opencl_conformance_tests_d3d.csv
    opencl_conformance_tests_full.csv
    opencl_conformance_tests_full_no_math_or_conversions.csv
    opencl_conformance_tests_math.csv
    opencl_conformance_tests_quick.csv
    run_conformance.py
    build_script_binary.py
    build_script_source.py
    build_script_spirv.py
)

