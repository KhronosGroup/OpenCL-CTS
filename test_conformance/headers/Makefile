ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

EXECUTABLES = test_headers \
              test_cl_h \
              test_cl_platform_h \
              test_cl_gl_h \
              test_opencl_h \
              test_cl_h_c99 \
              test_cl_platform_h_c99 \
              test_cl_gl_h_c99 \
              test_opencl_h_c99

INCLUDE = 
HCFLAGS =  ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
CFLAGS = -Wall -pedantic ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
LIBRARIES = ${ATF}

all : debug
release : debug
debug:
	cc $(HCFLAGS) test_headers.c $(LIBRARIES) -o test_headers
	cc $(CFLAGS) test_cl.h.c $(LIBRARIES) -o test_cl_h
	cc $(CFLAGS) test_cl_platform.h.c $(LIBRARIES) -o test_cl_platform_h
	cc $(CFLAGS) test_cl_gl.h.c $(LIBRARIES) -o test_cl_gl_h
	cc $(CFLAGS) test_opencl.h.c $(LIBRARIES) -o test_opencl_h
	cc $(CFLAGS) -std=c99 test_cl.h.c $(LIBRARIES) -o test_cl_h_c99
	cc $(CFLAGS) -std=c99 test_cl_platform.h.c $(LIBRARIES) -o test_cl_platform_h_c99
	cc $(CFLAGS) -std=c99 test_cl_gl.h.c $(LIBRARIES) -o test_cl_gl_h_c99
	cc $(CFLAGS) -std=c99 test_opencl.h.c $(LIBRARIES) -o test_opencl_h_c99
	
install:

	
clean: 
	rm -f $(EXECUTABLES)
