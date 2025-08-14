if (WIN32)
include_directories(${ CLConform_SOURCE_DIR } / test_common
                    / harness ${ CLConform_INCLUDE_DIR }) link_directories(${
    CL_LIB_DIR })

    list(APPEND CLConform_LIBRARIES directx_wrapper)

        set(MODULE_NAME CL_KHR_EXTERNAL_SEMAPHORE_DX_FENCE)

            set(${
                MODULE_NAME } _SOURCES main.cpp test_external_semaphore_dx_fence
                    .cpp test_external_semaphore_dx_fence_negative_wait_signal
                    .cpp test_external_semaphore_dx_fence_queries
                    .cpp test_external_semaphore_dx_fence_export.cpp)

                set_source_files_properties(${
                    MODULE_NAME } _SOURCES PROPERTIES LANGUAGE CXX)

                    include_directories(${ CMAKE_CURRENT_SOURCE_DIR })
                        include_directories("../../common/directx_wrapper")
                            include(../../ CMakeCommon.txt) endif(WIN32)
