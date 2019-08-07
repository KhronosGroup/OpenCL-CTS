set(MODULE_NAME VECSTEP)

set(${MODULE_NAME}_SOURCES
    globals.c
        test_step.c
        main.c
        structs.c
        type_replacer.c
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
