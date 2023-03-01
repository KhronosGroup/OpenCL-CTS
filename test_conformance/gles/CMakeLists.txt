set (MODULE_NAME GLES)

set (${MODULE_NAME}_SOURCES
        main.cpp
        test_buffers.cpp
        test_images_2D.cpp
        test_images_3D.cpp
        test_renderbuffer.cpp
        test_images_2D_info.cpp
        test_images_3D_info.cpp
        test_renderbuffer_info.cpp
        test_fence_sync.cpp
        helpers.cpp
        setup_egl.cpp
        ../../test_common/gles/helpers.cpp
    )

list(APPEND CLConform_LIBRARIES EGL GLESv2)

include(../CMakeCommon.cmake)

if(DEFINED USE_GLES3)
    target_compile_definitions(${${MODULE_NAME}_OUT} PRIVATE GLES3)
endif()
if(MSVC)
    # Don't warn about using the portable "strdup" function.
    target_compile_definitions(${${MODULE_NAME}_OUT} PRIVATE _CRT_NONSTDC_NO_DEPRECATE)
endif()