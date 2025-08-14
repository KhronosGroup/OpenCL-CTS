set(DIRECTX_WRAPPER_SOURCES directx_wrapper.cpp)

    add_library(directx_wrapper STATIC ${ DIRECTX_WRAPPER_SOURCES })

        include_directories(${ CMAKE_CURRENT_SOURCE_DIR })

            include_directories(${ CLConform_INCLUDE_DIR })

                if (WIN32) target_link_libraries(directx_wrapper d3d12) endif()
