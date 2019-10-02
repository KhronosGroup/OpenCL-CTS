add_executable(test_headers 
    test_headers.c
)

add_executable(test_cl_h
    test_cl.h.c
)

add_executable(test_cl_platform_h
    test_cl_platform.h.c
)

add_executable(test_cl_gl_h
    test_cl_gl.h.c
)

add_executable(test_opencl_h
    test_opencl.h.c
)

add_executable(test_cl_h_c99
    test_cl.h.c
)

add_executable(test_cl_platform_h_c99
    test_cl_platform.h.c
)

add_executable(test_cl_gl_h_c99
    test_cl_gl.h.c
)

add_executable(test_opencl_h_c99
    test_opencl.h.c
)

set_target_properties(test_cl_h_c99
    test_cl_platform_h_c99
    test_cl_gl_h_c99
    test_opencl_h_c99
    PROPERTIES COMPILE_FLAGS "-std=c99"
)

TARGET_LINK_LIBRARIES(test_headers
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_cl_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_cl_platform_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_cl_gl_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_opencl_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_cl_h_c99
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_cl_platform_h_c99
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_cl_gl_h_c99
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(test_opencl_h_c99
        ${CLConform_LIBRARIES})

add_dependencies(OpenCLCTS test_headers test_cl_h
    test_cl_platform_h test_cl_gl_h
    test_opencl_h test_cl_h_c99
    test_cl_platform_h_c99 test_cl_gl_h_c99
    test_opencl_h_c99)
install(TARGETS test_headers test_cl_h
    test_cl_platform_h test_cl_gl_h
    test_opencl_h test_cl_h_c99
    test_cl_platform_h_c99 test_cl_gl_h_c99
    test_opencl_h_c99
    DESTINATION "${CLConf_OUT_DIR}"
    COMPONENT OpenCLCTS)
