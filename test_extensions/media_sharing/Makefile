ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.cpp wrappers.cpp utils.cpp test_create_context.cpp test_functions_api.cpp test_functions_kernel.cpp test_get_device_ids.cpp test_interop_sync.cpp test_memory_access.cpp test_other_data_types.cpp ../../test_common/harness/errorHelpers.c ../../test_common/harness/testHarness.c ../../test_common/harness/mt19937.c ../../test_common/harness/rounding_mode.c ../../test_common/harness/typeWrappers.cpp ../../test_common/harness/kernelHelpers.c
DEFINES = 
SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(SOURCES)
HEADERS = 
TARGET = media_sharing
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32
CC = g++ 
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
