ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c test_clamp.c test_degrees.c \
			test_max.c test_maxf.c test_min.c test_minf.c \
            test_mix.c test_radians.c test_step.c test_stepf.c\
            test_smoothstep.c test_smoothstepf.c test_sign.c \
			test_fmax.c test_fmin.c test_fmaxf.c test_fminf.c test_binary_fn.c \
		  ../../test_common/harness/errorHelpers.c \
		  ../../test_common/harness/threadTesting.c \
		  ../../test_common/harness/testHarness.c \
                  ../../test_common/harness/conversions.c \
                  ../../test_common/harness/mt19937.c \
		  ../../test_common/harness/kernelHelpers.c 

SOURCES = $(abspath $(SRCS))

LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(SOURCES)
HEADERS = 
TARGET = test_commonfns
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -Wshorten-64-to-32 -Os
CC = c++
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF}
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
