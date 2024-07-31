//
// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "testBase.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

#if !defined(__APPLE__)
#include <CL/cl_gl.h>
#endif

static const char *bufferKernelPattern =
    "__kernel void sample_test( __global %s%s *source, __global %s%s *clDest, "
    "__global %s%s *glDest )\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "     clDest[ tid ] = source[ tid ] + (%s%s)(1);\n"
    "     glDest[ tid ] = source[ tid ] + (%s%s)(2);\n"
    "}\n";

#define TYPE_CASE(enum, type, range, offset)                                   \
    case enum: {                                                               \
        cl_##type *ptr = (cl_##type *)outData;                                 \
        for (i = 0; i < count; i++)                                            \
            ptr[i] = (cl_##type)((genrand_int32(d) & range) - offset);         \
        break;                                                                 \
    }

void gen_input_data(ExplicitType type, size_t count, MTdata d, void *outData)
{
    size_t i;

    switch (type)
    {
        case kBool: {
            bool *boolPtr = (bool *)outData;
            for (i = 0; i < count; i++)
            {
                boolPtr[i] = (genrand_int32(d) & 1) ? true : false;
            }
            break;
        }

            TYPE_CASE(kChar, char, 250, 127)
            TYPE_CASE(kUChar, uchar, 250, 0)
            TYPE_CASE(kShort, short, 65530, 32767)
            TYPE_CASE(kUShort, ushort, 65530, 0)
            TYPE_CASE(kInt, int, 0x0fffffff, 0x70000000)
            TYPE_CASE(kUInt, uint, 0x0fffffff, 0)

        case kLong: {
            cl_long *longPtr = (cl_long *)outData;
            for (i = 0; i < count; i++)
            {
                longPtr[i] = (cl_long)genrand_int32(d)
                    | ((cl_ulong)genrand_int32(d) << 32);
            }
            break;
        }

        case kULong: {
            cl_ulong *ulongPtr = (cl_ulong *)outData;
            for (i = 0; i < count; i++)
            {
                ulongPtr[i] = (cl_ulong)genrand_int32(d)
                    | ((cl_ulong)genrand_int32(d) << 32);
            }
            break;
        }

        case kFloat: {
            cl_float *floatPtr = (float *)outData;
            for (i = 0; i < count; i++)
                floatPtr[i] = get_random_float(-100000.f, 100000.f, d);
            break;
        }

        default:
            log_error(
                "ERROR: Invalid type passed in to generate_random_data!\n");
            break;
    }
}

#define INC_CASE(enum, type)                                                   \
    case enum: {                                                               \
        cl_##type *src = (cl_##type *)inData;                                  \
        cl_##type *dst = (cl_##type *)outData;                                 \
        *dst = *src + 1;                                                       \
        break;                                                                 \
    }

void get_incremented_value(void *inData, void *outData, ExplicitType type)
{
    switch (type)
    {
        INC_CASE(kChar, char)
        INC_CASE(kUChar, uchar)
        INC_CASE(kShort, short)
        INC_CASE(kUShort, ushort)
        INC_CASE(kInt, int)
        INC_CASE(kUInt, uint)
        INC_CASE(kLong, long)
        INC_CASE(kULong, ulong)
        INC_CASE(kFloat, float)
        default: break;
    }
}

int test_buffer_kernel(cl_context context, cl_command_queue queue,
                       ExplicitType vecType, size_t vecSize, int numElements,
                       int validate_only, MTdata d)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[3];
    size_t dataSize = numElements * 16;
    std::vector<cl_long> inData(dataSize), outDataCL(dataSize),
        outDataGL(dataSize);

    glBufferWrapper inGLBuffer, outGLBuffer;
    int i;
    size_t bufferSize;

    int error;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4];

    /* Create the source */
    if (vecSize == 1)
        sizeName[0] = 0;
    else
        sprintf(sizeName, "%d", (int)vecSize);

    sprintf(kernelSource, bufferKernelPattern, get_explicit_type_name(vecType),
            sizeName, get_explicit_type_name(vecType), sizeName,
            get_explicit_type_name(vecType), sizeName,
            get_explicit_type_name(vecType), sizeName,
            get_explicit_type_name(vecType), sizeName);

    /* Create kernels */
    programPtr = kernelSource;
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
    {
        return -1;
    }

    bufferSize = numElements * vecSize * get_explicit_type_size(vecType);

    /* Generate some almost-random input data */
    gen_input_data(vecType, vecSize * numElements, d, inData.data());

    /* Generate some GL buffers to go against */
    glGenBuffers(1, &inGLBuffer);
    glGenBuffers(1, &outGLBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, inGLBuffer);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, inData.data(), GL_STATIC_DRAW);

    // Note: we need to bind the output buffer, even though we don't care about
    // its values yet, because CL needs it to get the buffer size
    glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, outDataGL.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();


    /* Generate some streams. The first and last ones are GL, middle one just
     * vanilla CL */
    streams[0] = (*clCreateFromGLBuffer_ptr)(context, CL_MEM_READ_ONLY,
                                             inGLBuffer, &error);
    test_error(error, "Unable to create input GL buffer");

    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, &error);
    test_error(error, "Unable to create output CL buffer");

    streams[2] = (*clCreateFromGLBuffer_ptr)(context, CL_MEM_WRITE_ONLY,
                                             outGLBuffer, &error);
    test_error(error, "Unable to create output GL buffer");


    /* Validate the info */
    if (validate_only)
    {
        int result = (CheckGLObjectInfo(streams[0], CL_GL_OBJECT_BUFFER,
                                        (GLuint)inGLBuffer, (GLenum)0, 0)
                      | CheckGLObjectInfo(streams[2], CL_GL_OBJECT_BUFFER,
                                          (GLuint)outGLBuffer, (GLenum)0, 0));
        for (i = 0; i < 3; i++)
        {
            streams[i].reset();
        }

        glDeleteBuffers(1, &inGLBuffer);
        inGLBuffer = 0;
        glDeleteBuffers(1, &outGLBuffer);
        outGLBuffer = 0;

        return result;
    }

    /* Assign streams and execute */
    for (int i = 0; i < 3; i++)
    {
        error = clSetKernelArg(kernel, i, sizeof(streams[i]), &streams[i]);
        test_error(error, "Unable to set kernel arguments");
    }
    error =
        (*clEnqueueAcquireGLObjects_ptr)(queue, 1, &streams[0], 0, NULL, NULL);
    test_error(error, "Unable to acquire GL obejcts");
    error =
        (*clEnqueueAcquireGLObjects_ptr)(queue, 1, &streams[2], 0, NULL, NULL);
    test_error(error, "Unable to acquire GL obejcts");

    /* Run the kernel */
    threads[0] = numElements;

    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    error =
        (*clEnqueueReleaseGLObjects_ptr)(queue, 1, &streams[0], 0, NULL, NULL);
    test_error(error, "clEnqueueReleaseGLObjects failed");
    error =
        (*clEnqueueReleaseGLObjects_ptr)(queue, 1, &streams[2], 0, NULL, NULL);
    test_error(error, "clEnqueueReleaseGLObjects failed");

    // Get the results from both CL and GL and make sure everything looks
    // correct
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, bufferSize,
                                outDataCL.data(), 0, NULL, NULL);
    test_error(error, "Unable to read output CL array!");

    glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer);
    void *glMem = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    memcpy(outDataGL.data(), glMem, bufferSize);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    char *inP = (char *)inData.data(), *glP = (char *)outDataGL.data(),
         *clP = (char *)outDataCL.data();
    error = 0;
    for (size_t i = 0; i < numElements * vecSize; i++)
    {
        cl_long expectedCLValue, expectedGLValue;
        get_incremented_value(inP, &expectedCLValue, vecType);
        get_incremented_value(&expectedCLValue, &expectedGLValue, vecType);

        if (memcmp(clP, &expectedCLValue, get_explicit_type_size(vecType)) != 0)
        {
            char scratch[64];
            log_error(
                "ERROR: Data sample %d from the CL output did not validate!\n",
                (int)i);
            log_error("\t   Input: %s\n",
                      GetDataVectorString(inP, get_explicit_type_size(vecType),
                                          1, scratch));
            log_error("\tExpected: %s\n",
                      GetDataVectorString(&expectedCLValue,
                                          get_explicit_type_size(vecType), 1,
                                          scratch));
            log_error("\t  Actual: %s\n",
                      GetDataVectorString(clP, get_explicit_type_size(vecType),
                                          1, scratch));
            error = -1;
        }

        if (memcmp(glP, &expectedGLValue, get_explicit_type_size(vecType)) != 0)
        {
            char scratch[64];
            log_error(
                "ERROR: Data sample %d from the GL output did not validate!\n",
                (int)i);
            log_error("\t   Input: %s\n",
                      GetDataVectorString(inP, get_explicit_type_size(vecType),
                                          1, scratch));
            log_error("\tExpected: %s\n",
                      GetDataVectorString(&expectedGLValue,
                                          get_explicit_type_size(vecType), 1,
                                          scratch));
            log_error("\t  Actual: %s\n",
                      GetDataVectorString(glP, get_explicit_type_size(vecType),
                                          1, scratch));
            error = -1;
        }

        if (error) return error;

        inP += get_explicit_type_size(vecType);
        glP += get_explicit_type_size(vecType);
        clP += get_explicit_type_size(vecType);
    }

    for (i = 0; i < 3; i++)
    {
        streams[i].reset();
    }

    glDeleteBuffers(1, &inGLBuffer);
    inGLBuffer = 0;
    glDeleteBuffers(1, &outGLBuffer);
    outGLBuffer = 0;

    return 0;
}

int test_buffers(cl_device_id device, cl_context context,
                 cl_command_queue queue, int numElements)
{
    ExplicitType vecType[] = {
        kChar, kUChar, kShort, kUShort, kInt,
        kUInt, kLong,  kULong, kFloat,  kNumExplicitTypes
    };
    unsigned int vecSizes[] = { 1, 2, 4, 8, 16, 0 };
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);


    for (typeIndex = 0; vecType[typeIndex] != kNumExplicitTypes; typeIndex++)
    {
        for (index = 0; vecSizes[index] != 0; index++)
        {
            // Test!
            if (test_buffer_kernel(context, queue, vecType[typeIndex],
                                   vecSizes[index], numElements, 0, seed)
                != 0)
            {
                char sizeNames[][4] = { "", "", "2", "", "4", "", "", "",  "8",
                                        "", "", "",  "", "",  "", "", "16" };
                log_error("   Buffer test %s%s FAILED\n",
                          get_explicit_type_name(vecType[typeIndex]),
                          sizeNames[vecSizes[index]]);
                retVal++;
            }
        }
    }

    return retVal;
}


int test_buffers_getinfo(cl_device_id device, cl_context context,
                         cl_command_queue queue, int numElements)
{
    ExplicitType vecType[] = {
        kChar, kUChar, kShort, kUShort, kInt,
        kUInt, kLong,  kULong, kFloat,  kNumExplicitTypes
    };
    unsigned int vecSizes[] = { 1, 2, 4, 8, 16, 0 };
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);


    for (typeIndex = 0; vecType[typeIndex] != kNumExplicitTypes; typeIndex++)
    {
        for (index = 0; vecSizes[index] != 0; index++)
        {
            // Test!
            if (test_buffer_kernel(context, queue, vecType[typeIndex],
                                   vecSizes[index], numElements, 1, seed)
                != 0)
            {
                char sizeNames[][4] = { "", "", "2", "", "4", "", "", "",  "8",
                                        "", "", "",  "", "",  "", "", "16" };
                log_error("   Buffer test %s%s FAILED\n",
                          get_explicit_type_name(vecType[typeIndex]),
                          sizeNames[vecSizes[index]]);
                retVal++;
            }
        }
    }

    return retVal;
}
