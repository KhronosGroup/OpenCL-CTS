ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
			cl_utils.c \
			Test_vLoadHalf.c \
			Test_roundTrip.c \
			Test_vStoreHalf.c \
			../../test_common/harness/errorHelpers.c \
			../../test_common/harness/kernelHelpers.c \
			../../test_common/harness/mingw_compat.c \
			../../test_common/harness/testHarness.c

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(SOURCES)
HEADERS =
TARGET = Test_half
INCLUDE =
COMPILERFLAGS = -c -Wall -g
CC =  ${CROSS_COMPILE}g++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF} $(INCLUDE)
LIBRARIES = -framework OpenCL ${ATF}

OBJECTS := ${SOURCES:.c=.o}

TARGETOBJECT =
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(INCLUDE) $(RC_CFLAGS) $(OBJECTS) -o $@ $(LIBPATH) $(LIBRARIES)

clean:
	rm -f $(TARGET) $(OBJECTS)

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
