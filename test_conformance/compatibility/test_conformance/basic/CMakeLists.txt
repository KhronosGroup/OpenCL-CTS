set(MODULE_NAME COMPATIBILITY_BASIC)

set(${MODULE_NAME}_SOURCES
    main.c
    test_readimage.c
    test_writeimage.c
)

set(${MODULE_NAME}_LIBS harness-compat)

include(../../../CMakeCommon.txt)
