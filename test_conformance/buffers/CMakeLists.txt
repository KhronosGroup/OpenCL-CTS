set(MODULE_NAME BUFFERS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_buffer_copy.cpp
    test_buffer_read.cpp
    test_buffer_write.cpp
    test_buffer_mem.cpp
    array_info.cpp
    test_buffer_map.cpp
    test_sub_buffers.cpp
    test_buffer_fill.cpp
    test_buffer_migrate.cpp
    test_image_migrate.cpp
)

include(../CMakeCommon.txt)
