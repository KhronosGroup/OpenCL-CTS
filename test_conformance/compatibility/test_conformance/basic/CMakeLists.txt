set(MODULE_NAME COMPATIBILITY_BASIC)

set(${MODULE_NAME}_SOURCES
    main.c
    test_fpmath_float.c test_fpmath_float2.c test_fpmath_float4.c
    test_intmath_int.c test_intmath_int2.c test_intmath_int4.c
    test_intmath_long.c test_intmath_long2.c test_intmath_long4.c
    test_hiloeo.c test_local.c test_pointercast.c
    test_if.c test_loop.c
    test_readimage.c test_readimage_int16.c test_readimage_fp32.c
    test_readimage3d.c test_readimage3d_int16.c test_readimage3d_fp32.c
    test_writeimage.c test_writeimage_int16.c test_writeimage_fp32.c
    test_multireadimageonefmt.c test_multireadimagemultifmt.c
    test_imagedim.c
    test_vloadstore.c
    test_int2float.c test_float2int.c
    test_createkernelsinprogram.c
    test_hostptr.c
    test_explicit_s2v.cpp
    test_constant.c
    test_image_multipass.c
    test_imagereadwrite.c test_imagereadwrite3d.c
    test_image_param.c
    test_imagenpot.c
    test_image_r8.c
    test_barrier.c
    test_basic_parameter_types.c
    test_arrayreadwrite.c
    test_imagearraycopy3d.c
    test_imagecopy.c
    test_imagerandomcopy.c
    test_arrayimagecopy.c
    test_arrayimagecopy3d.c
    test_imagecopy3d.c
    test_enqueue_map.cpp
    test_work_item_functions.cpp
    test_astype.cpp
    test_async_copy.cpp
    test_sizeof.c
    test_vec_type_hint.c
    test_constant_source.cpp
    test_bufferreadwriterect.c
    test_async_strided_copy.cpp
    test_kernel_memory_alignment.cpp
    test_local_kernel_scope.cpp
)

set(${MODULE_NAME}_LIBS harness-compat)

include(../../../CMakeCommon.txt)
