# Note: Each extension adds its own test executable to the build. When
# creating tests for a new extension ensure to update the
# `opencl_conformance_tests_*.csv` files with the path to the new binary so
# that the extension tests are included in conformance test runs.
add_subdirectory( cl_ext_cxx_for_opencl )
add_subdirectory( cl_khr_command_buffer )
add_subdirectory( cl_khr_dx9_media_sharing )
add_subdirectory( cl_khr_external_memory_dma_buf )
add_subdirectory( cl_khr_external_memory_ahb )
add_subdirectory( cl_khr_semaphore )
add_subdirectory( cl_khr_kernel_clock )
add_subdirectory( cl_ext_buffer_device_address )
if(VULKAN_IS_SUPPORTED)
    add_subdirectory( cl_khr_external_semaphore )
endif()
