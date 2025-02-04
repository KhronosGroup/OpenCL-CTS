set(MODULE_NAME BASIC)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    add_cxx_flag_if_supported(-Wno-narrowing)
endif()

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_fpmath.cpp
    test_intmath.cpp
    test_hiloeo.cpp test_local.cpp test_pointercast.cpp
    test_if.cpp test_loop.cpp
    test_readimage.cpp
    test_writeimage.cpp
    test_multireadimageonefmt.cpp test_multireadimagemultifmt.cpp
    test_imagedim.cpp
    test_vloadstore.cpp
    test_int2fp.cpp
    test_createkernelsinprogram.cpp
    test_hostptr.cpp
    test_explicit_s2v.cpp
    test_constant.cpp
    test_image_multipass.cpp
    test_imagereadwrite.cpp test_imagereadwrite3d.cpp
    test_image_param.cpp
    test_imagenpot.cpp
    test_image_r8.cpp
    test_barrier.cpp
    test_basic_parameter_types.cpp
    test_arrayreadwrite.cpp
    test_arraycopy.cpp
    test_imagearraycopy.cpp
    test_imagecopy.cpp
    test_imagerandomcopy.cpp
    test_arrayimagecopy.cpp
    test_imagecopy3d.cpp
    test_enqueue_map.cpp
    test_work_item_functions.cpp
    test_astype.cpp
    test_async_copy.cpp
    test_async_copy2D.cpp
    test_async_copy3D.cpp
    test_async_copy_fence.cpp	
    test_sizeof.cpp
    test_vector_creation.cpp
    test_vector_swizzle.cpp
    test_vec_type_hint.cpp
    test_numeric_constants.cpp
    test_constant_source.cpp
    test_bufferreadwriterect.cpp
    test_async_strided_copy.cpp
    test_preprocessors.cpp
    test_kernel_memory_alignment.cpp
    test_global_work_offsets.cpp
    test_kernel_call_kernel_function.cpp
    test_local_kernel_scope.cpp
    test_progvar.cpp
    test_global_linear_id.cpp
    test_local_linear_id.cpp
    test_enqueued_local_size.cpp
    test_simple_image_pitch.cpp
    test_get_linear_ids.cpp
    test_rw_image_access_qualifier.cpp
    test_enqueued_local_size.cpp
    test_global_linear_id.cpp
    test_local_linear_id.cpp
    test_progvar.cpp
)

if(APPLE)
    list(APPEND ${MODULE_NAME}_SOURCES test_queue_priority.cpp)
endif(APPLE)

set_gnulike_module_compile_flags("-Wno-sign-compare -Wno-format")

include(../CMakeCommon.txt)
