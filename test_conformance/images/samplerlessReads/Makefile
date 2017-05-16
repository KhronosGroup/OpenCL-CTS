ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.cpp \
		test_iterations.cpp \
		test_loops.cpp \
        test_read_1D.cpp \
        test_read_1D_buffer.cpp \
        test_read_1D_array.cpp \
        test_read_2D_array.cpp \
		test_read_3D.cpp \
		../image_helpers.cpp  \
		../../../test_common/harness/errorHelpers.c \
		../../../test_common/harness/threadTesting.c \
		../../../test_common/harness/kernelHelpers.c \
		../../../test_common/harness/imageHelpers.cpp \
		../../../test_common/harness/conversions.c \
		../../../test_common/harness/testHarness.c \
                ../../../test_common/harness/mt19937.c \
		../../../test_common/harness/typeWrappers.cpp
		  
DEFINES = DONT_TEST_GARBAGE_POINTERS

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK =
HEADERS = 
TARGET = test_samplerless_reads
INCLUDE = -I../../test_common/harness
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32 -Os
CC = c++
CXX = c++
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
