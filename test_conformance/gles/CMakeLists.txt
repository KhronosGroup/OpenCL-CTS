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

if(ANDROID)
    list(APPEND CLConform_LIBRARIES GLESv2)
elseif(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGLES3")
    list(APPEND CLConform_LIBRARIES libEGL libGLESv2 )
endif(ANDROID)

include(../CMakeCommon.txt)
