set(MODULE_NAME VECALIGN)

set(${MODULE_NAME}_SOURCES
    globals.cpp
        main.cpp
        structs.cpp
        test_vec_align.cpp
        type_replacer.cpp
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)

