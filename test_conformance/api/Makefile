ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
			test_retain_program.c \
			test_queries.cpp \
			test_create_kernels.c \
			test_kernels.c \
            test_kernel_arg_info.c \
			test_api_min_max.c \
			test_kernel_arg_changes.cpp \
			test_kernel_arg_multi_setup.cpp \
			test_binary.cpp \
			test_native_kernel.cpp \
			test_create_context_from_type.cpp \
			test_platform.cpp \
			test_retain.cpp \
			test_device_min_data_type_align_size_alignment.cpp \
			test_mem_objects.cpp \
            test_bool.c \
            test_null_buffer_arg.c \
            test_mem_object_info.cpp \
            ../../test_common/harness/errorHelpers.c \
			../../test_common/harness/threadTesting.c \
			../../test_common/harness/testHarness.c \
			../../test_common/harness/imageHelpers.cpp \
			../../test_common/harness/kernelHelpers.c \
			../../test_common/harness/typeWrappers.cpp \
			../../test_common/harness/mt19937.c \
			../../test_common/harness/conversions.c
		  
DEFINES = DONT_TEST_GARBAGE_POINTERS

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
HEADERS = 
TARGET = test_api
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32
CC = c++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
CXXFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
LIBRARIES = -framework OpenCL -framework OpenGL -framework GLUT -framework AppKit ${ATF}

OBJECTS := ${SOURCES:.c=.o}
OBJECTS := ${OBJECTS:.cpp=.o}

TARGETOBJECT =
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(RC_CFLAGS) $(OBJECTS) -o $@ $(LIBPATH) $(LIBRARIES)

clean:
	rm -f $(TARGET) $(OBJECTS)

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
