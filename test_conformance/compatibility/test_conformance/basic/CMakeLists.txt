set(MODULE_NAME COMPATIBILITY_BASIC)

set(${MODULE_NAME}_SOURCES
    main.c
    test_readimage.c test_readimage_int16.c test_readimage_fp32.c
    test_readimage3d.c test_readimage3d_int16.c test_readimage3d_fp32.c
    test_writeimage.c
    test_multireadimageonefmt.c test_multireadimagemultifmt.c
    test_imagedim.c
    test_image_multipass.c
    test_image_param.c
    test_imagenpot.c
    test_image_r8.c
    test_imagearraycopy3d.c
    test_async_copy.cpp
    test_sizeof.c
    test_bufferreadwriterect.c
    test_async_strided_copy.cpp
)

set(${MODULE_NAME}_LIBS harness-compat)

include(../../../CMakeCommon.txt)
