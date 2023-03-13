set(MODULE_NAME PROFILING)

set(${MODULE_NAME}_SOURCES
    main.cpp
    readArray.cpp
    writeArray.cpp
    readImage.cpp
    writeImage.cpp
    copy.cpp
    execute.cpp
    execute_multipass.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
read_array_int
read_array_uint
read_array_long
read_array_ulong
read_array_short
read_array_ushort
read_array_float
read_array_char
read_array_uchar
read_array_struct
write_array_int
write_array_uint
write_array_long
write_array_ulong
write_array_short
write_array_ushort
write_array_float
write_array_char
write_array_uchar
write_array_struct
read_image_float
read_image_char
read_image_uchar
write_image_float
write_image_char
write_image_uchar
copy_array
copy_partial_array
copy_image
copy_array_to_image
execute
)

include(../CMakeCommon.cmake)
