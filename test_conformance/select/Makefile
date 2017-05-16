ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = test_select.c util_select.c ../../test_common/harness/mt19937.c ../../test_common/harness/kernelHelpers.c ../../test_common/harness/errorHelpers.c

LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(abspath $(SRCS))
HEADERS =  test_select.h
TARGET = test_select
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Os -Wshorten-64-to-32
CC = c++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF}
LIBRARIES = -framework OpenCL ${ATF}

OBJECTS = test_select.o util_select.o mt19937.o kernelHelpers.o errorHelpers.o
TARGETOBJECT =
all: $(TARGET)

$(OBJECTS): $(FRAMEWORK) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDE) $(FRAMEWORK)

$(TARGET): $(OBJECTS)
	$(CC) $(RC_CFLAGS) $(OBJECTS) -o $@ $(LIBPATH) $(LIBRARIES)

clean:
	rm -f $(TARGET) $(OBJECTS)

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
