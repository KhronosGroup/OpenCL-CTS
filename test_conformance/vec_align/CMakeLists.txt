set(MODULE_NAME VECALIGN)

set(${MODULE_NAME}_SOURCES
    globals.c
        main.c
        structs.c
        test_vec_align.c
        type_replacer.c
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)

