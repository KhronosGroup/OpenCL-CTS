ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
	test_allocate_shared_buffer.cpp \
	test_byte_granularity.cpp \
	test_cross_buffer_pointers.cpp \
	test_enqueue_api.cpp \
    test_fine_grain_memory_consistency.cpp \
    test_fine_grain_sync_buffers.cpp \
    test_pointer_passing.cpp \
    test_set_kernel_exec_info_svm_ptrs.cpp \
	test_shared_address_space_coarse_grain.cpp \
	test_shared_address_space_fine_grain_buffers.cpp \
	test_shared_address_space_fine_grain.cpp \
    test_shared_sub_buffers.cpp \
    test_migrate.cpp \
    ../../test_common/harness/errorHelpers.c \
	../../test_common/harness/threadTesting.c \
	../../test_common/harness/testHarness.c \
	../../test_common/harness/kernelHelpers.c \
	../../test_common/harness/typeWrappers.cpp \
	../../test_common/harness/mt19937.c \
		  
DEFINES = DONT_TEST_GARBAGE_POINTERS

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
HEADERS = 
TARGET = test_SVM
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
