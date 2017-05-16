ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.cpp datagen.cpp kernelargs.cpp run_build_test.cpp run_services.cpp \
			../../test_common/miniz/miniz.c \
			../../test_common/harness/testHarness.c \
			../../test_common/harness/errorHelpers.c \
			../../test_common/harness/typeWrappers.cpp \
			../../test_common/harness/mt19937.c \
			../../test_common/harness/os_helpers.c \
			../../test_common/harness/kernelHelpers.c

SOURCES = $(abspath $(SRCS))


LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
           
FRAMEWORK = ${SOURCES}
HEADERS = 
TARGET = test_spir
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32
#COMPILERFLAGS = -c -Wall -g -DUSE_LOCAL_THREADS
CC = c++
CFLAGS = $(COMPILERFLAGS) $(RC_CFLAGS) ${USE_ATF}
CXXFLAGS= $(COMPILERFLAGS) $(RC_CFLAGS) ${USE_ATF}
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
