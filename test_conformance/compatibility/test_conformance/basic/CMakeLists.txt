set(MODULE_NAME COMPATIBILITY_BASIC)

set(${MODULE_NAME}_SOURCES
    main.c
    test_readimage.c
    test_writeimage.c
    test_imagearraycopy3d.c
    test_async_copy.cpp
    test_sizeof.c
    test_bufferreadwriterect.c
    test_async_strided_copy.cpp
)

set(${MODULE_NAME}_LIBS harness-compat)

include(../../../CMakeCommon.txt)
