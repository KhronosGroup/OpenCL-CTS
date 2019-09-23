set(MODULE_NAME BUFFERS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_buffer_copy.c
    test_buffer_read.c
    test_buffer_write.c
    test_buffer_mem.c
    array_info.c
    test_buffer_map.c
    test_sub_buffers.cpp
    test_buffer_fill.c
    test_buffer_migrate.c
    test_image_migrate.c
)

include(../CMakeCommon.txt)
