ifdef BUILD_WITH_ATF
ATF = -framework ATF
USE_ATF = -DUSE_ATF
endif

SRCS = main.c \
		test_fpmath_float.c test_fpmath_float2.c test_fpmath_float4.c \
		test_intmath_int.c test_intmath_int2.c test_intmath_int4.c  \
		test_intmath_long.c test_intmath_long2.c test_intmath_long4.c \
		test_hiloeo.c test_local.c test_local_kernel_scope.cpp test_pointercast.c \
		test_if.c test_sizeof.c test_loop.c \
		test_readimage.c test_readimage_int16.c test_readimage_fp32.c \
		test_readimage3d.c test_readimage3d_int16.c test_readimage3d_fp32.c \
		test_writeimage.c test_writeimage_int16.c test_writeimage_fp32.c \
		test_multireadimageonefmt.c test_multireadimagemultifmt.c \
		test_imagedim.c \
		test_vloadstore.c \
		test_int2float.c test_float2int.c \
		test_createkernelsinprogram.c \
		test_hostptr.c \
		test_explicit_s2v.cpp \
		test_constant.c \
		test_constant_source.cpp \
		test_image_multipass.c \
		test_imagereadwrite.c test_imagereadwrite3d.c \
		test_bufferreadwriterect.c \
		test_image_param.c \
		test_imagenpot.c \
		test_image_r8.c \
		test_barrier.c \
		test_arrayreadwrite.c \
		test_arraycopy.c \
		test_imagearraycopy.c \
		test_imagearraycopy3d.c \
		test_imagecopy.c \
		test_imagerandomcopy.c \
		test_arrayimagecopy.c \
		test_arrayimagecopy3d.c\
		test_imagecopy3d.c \
		test_enqueue_map.cpp \
		test_work_item_functions.cpp \
		test_astype.cpp \
		test_async_copy.cpp \
		test_async_strided_copy.cpp \
		test_numeric_constants.cpp \
		test_kernel_call_kernel_function.cpp \
		test_basic_parameter_types.c \
		test_vector_creation.cpp \
		test_vec_type_hint.c \
		test_preprocessors.cpp \
		test_kernel_memory_alignment.cpp \
		test_global_work_offsets.cpp \
		../../test_common/harness/errorHelpers.c \
		../../test_common/harness/threadTesting.c \
		../../test_common/harness/testHarness.c \
		../../test_common/harness/rounding_mode.c \
		../../test_common/harness/kernelHelpers.c \
		../../test_common/harness/typeWrappers.cpp \
		../../test_common/harness/imageHelpers.cpp \
                ../../test_common/harness/mt19937.c \
		../../test_common/harness/conversions.c 

DEFINES = 

SOURCES = $(abspath $(SRCS))
LIBPATH += -L/System/Library/Frameworks/OpenCL.framework/Libraries
LIBPATH += -L.
FRAMEWORK = $(SOURCES)
HEADERS = 
TARGET = test_basic
INCLUDE = 
COMPILERFLAGS = -c -Wall -g -O0 -Wshorten-64-to-32
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



