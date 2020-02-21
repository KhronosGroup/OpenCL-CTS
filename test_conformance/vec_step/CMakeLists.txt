set(MODULE_NAME VECSTEP)

set(${MODULE_NAME}_SOURCES
    globals.cpp
        test_step.cpp
        main.cpp
        structs.cpp
        type_replacer.cpp
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
