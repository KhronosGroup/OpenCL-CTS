ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
		test_device_partition.cpp \
		../../test_common/harness/errorHelpers.c \
		../../test_common/harness/threadTesting.c \
		../../test_common/harness/testHarness.c \
		../../test_common/harness/kernelHelpers.c \
		../../test_common/harness/genericThread.cpp \
		../../test_common/harness/mt19937.c \
		../../test_common/harness/conversions.c \
		../../test_common/harness/typeWrappers.cpp \
		../../test_common/harness/msvc9.c
DEFINES = 

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(abspath $(SRCS))
HEADERS =  procs.h testBase.h
TARGET = test_device_partition
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Os -Wshorten-64-to-32
CC = c++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
CXXFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
LIBRARIES = -framework OpenCL -framework OpenGL -framework GLUT -framework AppKit ${RC_CFLAGS} ${ATF}

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
