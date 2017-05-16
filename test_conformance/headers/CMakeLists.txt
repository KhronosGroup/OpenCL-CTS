add_executable(conformance_test_headers 
    test_headers.c
)

add_executable(conformance_test_cl_h
    test_cl.h.c
)

add_executable(conformance_test_cl_platform_h
    test_cl_platform.h.c
)

add_executable(conformance_test_cl_gl_h
    test_cl_gl.h.c
)

add_executable(conformance_test_opencl_h
    test_opencl.h.c
)

set(CMAKE_C_FLAGS "" )
set(CMAKE_CXX_FLAGS "" )

TARGET_LINK_LIBRARIES(conformance_test_headers
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(conformance_test_cl_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(conformance_test_cl_platform_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(conformance_test_cl_gl_h
        ${CLConform_LIBRARIES})

TARGET_LINK_LIBRARIES(conformance_test_opencl_h
        ${CLConform_LIBRARIES})
