ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.cpp \
		  helpers.cpp \
		  test_buffers.cpp \
		  test_fence_sync.cpp \
		  test_images_1D.cpp \
		  test_images_1Darray.cpp \
		  test_images_2Darray.cpp \
		  test_images_2D.cpp \
		  test_images_3D.cpp \
		  test_images_getinfo_common.cpp \
		  test_images_read_common.cpp \
		  test_images_write_common.cpp \
		  test_renderbuffer.cpp \
		  test_renderbuffer_info.cpp \
      test_images_depth.cpp \
      test_images_multisample.cpp \
			test_image_methods.cpp \
		  ../../test_common/gl/helpers.cpp \
		  ../../test_common/gl/setup_osx.cpp \
		  ../../test_common/harness/conversions.c \
		  ../../test_common/harness/errorHelpers.c \
		  ../../test_common/harness/genericThread.cpp \
		  ../../test_common/harness/imageHelpers.c \
		  ../../test_common/harness/kernelHelpers.c \
		  ../../test_common/harness/mt19937.c \
		  ../../test_common/harness/testHarness.c \
		  ../../test_common/harness/threadTesting.c 
		  
DEFINES = 

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(SOURCES)
HEADERS = 
TARGET = test_gl
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
