set(MODULE_NAME PROFILING)

set(${MODULE_NAME}_SOURCES
    main.c
    readArray.c
    writeArray.c
    readImage.c
    writeImage.c
    copy.c
    execute.c
    execute_multipass.c
)

include(../CMakeCommon.txt)
