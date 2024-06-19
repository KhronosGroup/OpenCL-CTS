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
#if defined(__APPLE__)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#include <CL/cl_gl.h>
#endif

static int test_renderbuffer_object_info(cl_context context,
                                         cl_command_queue queue, GLsizei width,
                                         GLsizei height, GLenum attachment,
                                         GLenum format, GLenum internalFormat,
                                         GLenum glType, ExplicitType type,
                                         MTdata d)
{
    int error;

    if (type == kHalf)
        if (DetectFloatToHalfRoundingMode(queue)) return 1;

    // Create the GL render buffer
    glFramebufferWrapper glFramebuffer;
    glRenderbufferWrapper glRenderbuffer;
    BufferOwningPtr<char> inputBuffer(CreateGLRenderbuffer(
        width, height, attachment, format, internalFormat, glType, type,
        &glFramebuffer, &glRenderbuffer, &error, d, true));
    if (error != 0) return error;

    clMemWrapper image = (*clCreateFromGLRenderbuffer_ptr)(
        context, CL_MEM_READ_ONLY, glRenderbuffer, &error);
    test_error(error, "clCreateFromGLRenderbuffer failed");

    log_info("- Given a GL format of %s, input type was %s, size was %d x %d\n",
             GetGLFormatName(internalFormat), get_explicit_type_name(type),
             (int)width, (int)height);

    // Verify the expected information here.
    return CheckGLObjectInfo(image, CL_GL_OBJECT_RENDERBUFFER,
                             (GLuint)glRenderbuffer, internalFormat, 0);
}

int test_renderbuffer_getinfo(cl_device_id device, cl_context context,
                              cl_command_queue queue, int numElements)
{
    GLenum attachments[] = { GL_COLOR_ATTACHMENT0_EXT };

    struct
    {
        GLenum internal;
        GLenum format;
        GLenum datatype;
        ExplicitType type;

    } formats[] = { { GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, kUChar },
                    { GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, kUChar },
                    { GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, kUShort },
                    { GL_RGBA32F_ARB, GL_RGBA, GL_FLOAT, kFloat },
                    { GL_RGBA16F_ARB, GL_RGBA, GL_HALF_FLOAT, kHalf } };

    size_t fmtIdx, tgtIdx;
    int error = 0;
    size_t iter = 6;
    RandomSeed seed(gRandomSeed);

    // Check if images are supported
    if (checkForImageSupport(device))
    {
        log_info("Device does not support images. Skipping test.\n");
        return 0;
    }

    if (!gluCheckExtension((const GLubyte *)"GL_EXT_framebuffer_object",
                           glGetString(GL_EXTENSIONS)))
    {
        log_info("Renderbuffers are not supported by this OpenGL "
                 "implementation; skipping test\n");
        return 0;
    }

    // Loop through a set of GL formats, testing a set of sizes against each one
    for (fmtIdx = 0; fmtIdx < sizeof(formats) / sizeof(formats[0]); fmtIdx++)
    {
        for (tgtIdx = 0; tgtIdx < sizeof(attachments) / sizeof(attachments[0]);
             tgtIdx++)
        {
            log_info("Testing Renderbuffer object info for %s : %s : %s\n",
                     GetGLFormatName(formats[fmtIdx].internal),
                     GetGLBaseFormatName(formats[fmtIdx].format),
                     GetGLTypeName(formats[fmtIdx].datatype));

            size_t i;
            for (i = 0; i < iter; i++)
            {
                GLsizei width = random_in_range(16, 512, seed);
                GLsizei height = random_in_range(16, 512, seed);

                if (test_renderbuffer_object_info(
                        context, queue, (int)width, (int)height,
                        attachments[tgtIdx], formats[fmtIdx].format,
                        formats[fmtIdx].internal, formats[fmtIdx].datatype,
                        formats[fmtIdx].type, seed))
                {
                    log_error("ERROR: Renderbuffer write test failed for GL "
                              "format %s : %s\n\n",
                              GetGLFormatName(formats[fmtIdx].internal),
                              GetGLTypeName(formats[fmtIdx].datatype));

                    error++;
                    break; // Skip other sizes for this combination
                }
            }
            if (i == iter)
            {
                log_info("passed: Renderbuffer write test passed for GL format "
                         "%s : %s\n\n",
                         GetGLFormatName(formats[fmtIdx].internal),
                         GetGLTypeName(formats[fmtIdx].datatype));
            }
        }
    }

    return error;
}
