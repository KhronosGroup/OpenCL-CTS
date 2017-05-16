ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = conversions.c \
	errorHelpers.c \
	genericThread.cpp \
	imageHelpers.cpp \
	kernelHelpers.c \
	mt19937.c \
	rounding_mode.c \
	testHarness.c \
	testHarness.cpp \
	ThreadPool.c \
	threadTesting.c \
	typeWrappers.cpp
		  
DEFINES = DONT_TEST_GARBAGE_POINTERS

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
HEADERS = 
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32
CC = c++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
CXXFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(DEFINES:%=-D%) $(INCLUDE)
LIBRARIES = -framework OpenCL -framework OpenGL -framework GLUT -framework AppKit ${ATF}

OBJECTS := ${SOURCES:.c=.o}
OBJECTS := ${OBJECTS:.cpp=.o}

all: $(OBJECTS)

clean:
	rm -f $(OBJECTS)

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
