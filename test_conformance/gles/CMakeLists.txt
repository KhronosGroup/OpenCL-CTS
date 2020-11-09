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

include(../CMakeCommon.txt)
