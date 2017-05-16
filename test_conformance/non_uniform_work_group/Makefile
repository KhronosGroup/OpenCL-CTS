ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
	test_basic.cpp \
	test_advanced_2d.cpp \
	test_advanced_3d.cpp \
	test_advanced_other.cpp \
	TestNonUniformWorkGroup.cpp \
	tools.cpp \
	../../test_common/harness/errorHelpers.c \
	../../test_common/harness/threadTesting.c \
	../../test_common/harness/testHarness.c \
	../../test_common/harness/kernelHelpers.c \
	../../test_common/harness/typeWrappers.cpp \

DEFINES = DONT_TEST_GARBAGE_POINTERS

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
HEADERS = 
TARGET = test_non_uniform_workgroup
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
