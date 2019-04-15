set(HEADERS_SOURCES
    test_headers.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
)

set_source_files_properties(${HEADERS_SOURCES} PROPERTIES LANGUAGE CXX)

set(HEADERS_OUT ${CONFORMANCE_PREFIX}headers${CONFORMANCE_SUFFIX})

add_executable(
    ${HEADERS_OUT}
    ${HEADERS_SOURCES}
)

set_property(TARGET ${HEADERS_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")

TARGET_LINK_LIBRARIES(${HEADERS_OUT} ${CLConform_LIBRARIES})

########################################################################################

set(CL_H_SOURCES
    test_cl.h.c
)

set_source_files_properties(${CL_H_SOURCES} PROPERTIES LANGUAGE CXX)

set(CL_H_OUT ${CONFORMANCE_PREFIX}cl_h${CONFORMANCE_SUFFIX})

add_executable(
    ${CL_H_OUT}
    ${CL_H_SOURCES}
)

set_property(TARGET ${CL_H_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")

TARGET_LINK_LIBRARIES(${CL_H_OUT} ${CLConform_LIBRARIES})

########################################################################################

set(CL_PLATFORM_H_SOURCES
    test_cl_platform.h.c
)

set_source_files_properties(${CL_PLATFORM_H_SOURCES} PROPERTIES LANGUAGE CXX)

set(CL_PLATFORM_H_OUT ${CONFORMANCE_PREFIX}cl_platform_h${CONFORMANCE_SUFFIX})

add_executable(
    ${CL_PLATFORM_H_OUT}
    ${CL_PLATFORM_H_SOURCES}
)

set_property(TARGET ${CL_PLATFORM_H_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")

TARGET_LINK_LIBRARIES(${CL_PLATFORM_H_OUT} ${CLConform_LIBRARIES})

########################################################################################

set(CL_GL_H_SOURCES
    test_cl_gl.h.c
)

set_source_files_properties(${CL_GL_H_SOURCES} PROPERTIES LANGUAGE CXX)

set(CL_GL_H_OUT ${CONFORMANCE_PREFIX}cl_gl_h${CONFORMANCE_SUFFIX})

add_executable(
    ${CL_GL_H_OUT}
    ${CL_GL_H_SOURCES}
)

set_property(TARGET ${CL_GL_H_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")

TARGET_LINK_LIBRARIES(${CL_GL_H_OUT} ${CLConform_LIBRARIES})

########################################################################################

set(OPENCL_H_SOURCES
    test_opencl.h.c
)

set_source_files_properties(${OPENCL_H_SOURCES} PROPERTIES LANGUAGE CXX)

set(OPENCL_H_OUT ${CONFORMANCE_PREFIX}opencl_h${CONFORMANCE_SUFFIX})

add_executable(
    ${OPENCL_H_OUT}
    ${OPENCL_H_SOURCES}
)

set_property(TARGET ${OPENCL_H_OUT} PROPERTY FOLDER "CONFORMANCE${CONFORMANCE_SUFFIX}")

TARGET_LINK_LIBRARIES(${OPENCL_H_OUT} ${CLConform_LIBRARIES})

########################################################################################

# end of file #
