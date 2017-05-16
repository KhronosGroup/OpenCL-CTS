ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
			test_popcount.c \
			test_int_basic_ops.c \
			test_integers.cpp \
			test_upsample.cpp \
			test_intmul24.c test_intmad24.c \
			test_sub_sat.c test_add_sat.c \
			test_abs.c test_absdiff.c  \
			test_unary_ops.cpp \
			verification_and_generation_functions.c \
			../../test_common/harness/conversions.c \
			../../test_common/harness/errorHelpers.c \
			../../test_common/harness/threadTesting.c \
			../../test_common/harness/testHarness.c \
			../../test_common/harness/mt19937.c \
			../../test_common/harness/ThreadPool.c \
			../../test_common/harness/kernelHelpers.c

DEFINES = 

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(SOURCES)
HEADERS = 
TARGET = test_integer_ops
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32 -Os
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
