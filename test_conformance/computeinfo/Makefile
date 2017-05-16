ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = main.c ../../test_common/harness/errorHelpers.c ../../test_common/harness/kernelHelpers.c 
HEADERS = 
TARGET = computeinfo
INCLUDE = -I./../../OpenCL/cl_headers/public
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32
CC = c++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF}
LIBRARIES = -framework OpenCL -framework OpenGL -framework GLUT -framework AppKit ${RC_CFLAGS} ${ATF}

OBJECTS = main.o errorHelpers.o kernelHelpers.o

TARGETOBJECT =
all: $(TARGET)

$(OBJECTS): $(FRAMEWORK) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDE) $(FRAMEWORK)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LIBPATH) $(LIBRARIES)

clean:
	rm -f $(TARGET) $(OBJECTS)

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
