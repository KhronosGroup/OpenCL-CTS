set(MODULE_NAME BASIC)

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
    test_arraycopy.c
    test_imagearraycopy.c
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
    test_async_copy2D.cpp
    test_async_copy3D.cpp
    test_async_copy_fence.cpp
    test_sizeof.c
    test_vector_creation.cpp
    test_vec_type_hint.c
    test_numeric_constants.cpp
    test_constant_source.cpp
    test_bufferreadwriterect.c
    test_async_strided_copy.cpp
    test_preprocessors.cpp
    test_kernel_memory_alignment.cpp
    test_global_work_offsets.cpp
    test_kernel_call_kernel_function.cpp
    test_local_kernel_scope.cpp
    test_progvar.cpp
    test_wg_barrier.c
    test_global_linear_id.c
    test_local_linear_id.c
    test_enqueued_local_size.c
    test_simple_image_pitch.c
    test_get_linear_ids.cpp
    test_rw_image_access_qualifier.c
    test_wg_barrier.c
    test_enqueued_local_size.c
    test_global_linear_id.c
    test_local_linear_id.c
    test_progvar.cpp
)

if(APPLE)
    list(APPEND ${MODULE_NAME}_SOURCES test_queue_priority.c)
endif(APPLE)

include(../CMakeCommon.txt)
