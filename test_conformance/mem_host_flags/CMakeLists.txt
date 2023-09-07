set(MODULE_NAME MEM_HOST_FLAGS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    mem_host_buffer.cpp
    mem_host_image.cpp
)

set_gnulike_module_compile_flags("-Wno-unused-but-set-variable")

include(../CMakeCommon.txt)
