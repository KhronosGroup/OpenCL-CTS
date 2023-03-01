set(MODULE_NAME MEM_HOST_FLAGS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    mem_host_buffer.cpp
    mem_host_image.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
1st part, non gl-sharing objects...
mem_host_read_only_buffer
mem_host_read_only_subbuffer
mem_host_write_only_buffer
mem_host_write_only_subbuffer
mem_host_no_access_buffer
mem_host_no_access_subbuffer
mem_host_read_only_image
mem_host_write_only_image
mem_host_no_access_image
)

include(../CMakeCommon.cmake)
