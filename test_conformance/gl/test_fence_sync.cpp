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
#include "gl/setup.h"
#include "harness/genericThread.h"

#if defined(__APPLE__)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#include <CL/cl_gl.h>
#if !defined(_WIN32) && !defined(__ANDROID__)
#include <GL/glx.h>
#endif
#endif

#ifndef GLsync
// For OpenGL before 3.2, we look for the ARB_sync extension and try to use that
#if !defined(_WIN32)
#include <inttypes.h>
#endif // !_WIN32
typedef int64_t GLint64;
typedef uint64_t GLuint64;
typedef struct __GLsync *GLsync;

#ifndef APIENTRY
#define APIENTRY
#endif

typedef GLsync(APIENTRY *glFenceSyncPtr)(GLenum condition, GLbitfield flags);
glFenceSyncPtr glFenceSyncFunc;

typedef bool(APIENTRY *glIsSyncPtr)(GLsync sync);
glIsSyncPtr glIsSyncFunc;

typedef void(APIENTRY *glDeleteSyncPtr)(GLsync sync);
glDeleteSyncPtr glDeleteSyncFunc;

typedef GLenum(APIENTRY *glClientWaitSyncPtr)(GLsync sync, GLbitfield flags,
                                              GLuint64 timeout);
glClientWaitSyncPtr glClientWaitSyncFunc;

typedef void(APIENTRY *glWaitSyncPtr)(GLsync sync, GLbitfield flags,
                                      GLuint64 timeout);
glWaitSyncPtr glWaitSyncFunc;

typedef void(APIENTRY *glGetInteger64vPtr)(GLenum pname, GLint64 *params);
glGetInteger64vPtr glGetInteger64vFunc;

typedef void(APIENTRY *glGetSyncivPtr)(GLsync sync, GLenum pname,
                                       GLsizei bufSize, GLsizei *length,
                                       GLint *values);
glGetSyncivPtr glGetSyncivFunc;

#define CHK_GL_ERR() printf("%s\n", gluErrorString(glGetError()))

static void InitSyncFns(void)
{
    glFenceSyncFunc = (glFenceSyncPtr)glutGetProcAddress("glFenceSync");
    glIsSyncFunc = (glIsSyncPtr)glutGetProcAddress("glIsSync");
    glDeleteSyncFunc = (glDeleteSyncPtr)glutGetProcAddress("glDeleteSync");
    glClientWaitSyncFunc =
        (glClientWaitSyncPtr)glutGetProcAddress("glClientWaitSync");
    glWaitSyncFunc = (glWaitSyncPtr)glutGetProcAddress("glWaitSync");
    glGetInteger64vFunc =
        (glGetInteger64vPtr)glutGetProcAddress("glGetInteger64v");
    glGetSyncivFunc = (glGetSyncivPtr)glutGetProcAddress("glGetSynciv");
}
#ifndef GL_ARB_sync
#define GL_MAX_SERVER_WAIT_TIMEOUT 0x9111

#define GL_OBJECT_TYPE 0x9112
#define GL_SYNC_CONDITION 0x9113
#define GL_SYNC_STATUS 0x9114
#define GL_SYNC_FLAGS 0x9115

#define GL_SYNC_FENCE 0x9116

#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117

#define GL_UNSIGNALED 0x9118
#define GL_SIGNALED 0x9119

#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001

#define GL_TIMEOUT_IGNORED 0xFFFFFFFFFFFFFFFFull

#define GL_ALREADY_SIGNALED 0x911A
#define GL_TIMEOUT_EXPIRED 0x911B
#define GL_CONDITION_SATISFIED 0x911C
#define GL_WAIT_FAILED 0x911D
#endif

#define USING_ARB_sync 1
#endif

typedef cl_event(CL_API_CALL *clCreateEventFromGLsyncKHR_fn)(
    cl_context context, GLsync sync, cl_int *errCode_ret);

clCreateEventFromGLsyncKHR_fn clCreateEventFromGLsyncKHR_ptr;


// clang-format off
static const char *updateBuffersKernel[] = {
    "__kernel void update( __global float4 * vertices, __global float4 "
    "*colors, int horizWrap, int rowIdx )\n"
    "{\n"
    "    size_t tid = get_global_id(0);\n"
    "\n"
    "    size_t xVal = ( tid & ( horizWrap - 1 ) );\n"
    "    vertices[ tid * 2 + 0 ] = (float4)( xVal, rowIdx*16.f, 0.0f, 1.f );\n"
    "    vertices[ tid * 2 + 1 ] = (float4)( xVal, rowIdx*16.f + 4.0f, 0.0f, "
    "1.f );\n"
    "\n"
    "    int rowV = rowIdx + 1;\n"
    "    colors[ tid * 2 + 0 ] = (float4)( ( rowV & 1 ) / 255.f, ( ( rowV & 2 "
    ") >> 1 ) / 255.f, ( ( rowV & 4 ) >> 2 ) / 255.f, 1.f );\n"
    "    //colors[ tid * 2 + 0 ] = (float4)( (float)xVal/(float)horizWrap, "
    "1.0f, 1.0f, 1.0f );\n"
    "    colors[ tid * 2 + 1 ] = colors[ tid * 2 + 0 ];\n"
    "}\n"
};
// clang-format on

// Passthrough VertexShader
static const char *vertexshader = "#version 150\n"
                                  "uniform mat4 projMatrix;\n"
                                  "in vec4 inPosition;\n"
                                  "in vec4 inColor;\n"
                                  "out vec4 vertColor;\n"
                                  "void main (void) {\n"
                                  "    gl_Position = projMatrix*inPosition;\n"
                                  "   vertColor = inColor;\n"
                                  "}\n";

// Passthrough FragmentShader
static const char *fragmentshader = "#version 150\n"
                                    "in vec4 vertColor;\n"
                                    "out vec4 outColor;\n"
                                    "void main (void) {\n"
                                    "    outColor = vertColor;\n"
                                    "}\n";

GLuint createShaderProgram(GLint *posLoc, GLint *colLoc)
{
    GLint logLength, status;
    GLuint program = glCreateProgram();
    GLuint vpShader;

    vpShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vpShader, 1, (const GLchar **)&vertexshader, NULL);
    glCompileShader(vpShader);
    glGetShaderiv(vpShader, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetShaderInfoLog(vpShader, logLength, &logLength, log);
        log_info("Vtx Shader compile log:\n%s", log);
        free(log);
    }

    glGetShaderiv(vpShader, GL_COMPILE_STATUS, &status);
    if (status == 0)
    {
        log_error("Failed to compile vtx shader:\n");
        return 0;
    }

    glAttachShader(program, vpShader);

    GLuint fpShader;
    fpShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fpShader, 1, (const GLchar **)&fragmentshader, NULL);
    glCompileShader(fpShader);

    glGetShaderiv(fpShader, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetShaderInfoLog(fpShader, logLength, &logLength, log);
        log_info("Frag Shader compile log:\n%s", log);
        free(log);
    }

    glAttachShader(program, fpShader);
    glGetShaderiv(fpShader, GL_COMPILE_STATUS, &status);
    if (status == 0)
    {
        log_error("Failed to compile frag shader:\n\n");
        return 0;
    }

    glLinkProgram(program);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetProgramInfoLog(program, logLength, &logLength, log);
        log_info("Program link log:\n%s", log);
        free(log);
    }

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == 0)
    {
        log_error("Failed to link program\n");
        return 0;
    }

    *posLoc = glGetAttribLocation(program, "inPosition");
    *colLoc = glGetAttribLocation(program, "inColor");

    return program;
}

void destroyShaderProgram(GLuint program)
{
    GLuint shaders[2];
    GLsizei count;
    glUseProgram(0);
    glGetAttachedShaders(program, 2, &count, shaders);
    int i;
    for (i = 0; i < count; i++)
    {
        glDetachShader(program, shaders[i]);
        glDeleteShader(shaders[i]);
    }
    glDeleteProgram(program);
}

// This function queues up and runs the above CL kernel that writes the vertex
// data
cl_int run_cl_kernel(cl_kernel kernel, cl_command_queue queue, cl_mem stream0,
                     cl_mem stream1, cl_int rowIdx, cl_event fenceEvent,
                     size_t numThreads)
{
    cl_int error = clSetKernelArg(kernel, 3, sizeof(rowIdx), &rowIdx);
    test_error(error, "Unable to set kernel arguments");

    clEventWrapper acqEvent1, acqEvent2, kernEvent, relEvent1, relEvent2;
    int numEvents = (fenceEvent != NULL) ? 1 : 0;
    cl_event *fence_evt = (fenceEvent != NULL) ? &fenceEvent : NULL;

    error = (*clEnqueueAcquireGLObjects_ptr)(queue, 1, &stream0, numEvents,
                                             fence_evt, &acqEvent1);
    test_error(error, "Unable to acquire GL obejcts");
    error = (*clEnqueueAcquireGLObjects_ptr)(queue, 1, &stream1, numEvents,
                                             fence_evt, &acqEvent2);
    test_error(error, "Unable to acquire GL obejcts");

    cl_event evts[2] = { acqEvent1, acqEvent2 };

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &numThreads, NULL, 2,
                                   evts, &kernEvent);
    test_error(error, "Unable to execute test kernel");

    error = (*clEnqueueReleaseGLObjects_ptr)(queue, 1, &stream0, 1, &kernEvent,
                                             &relEvent1);
    test_error(error, "clEnqueueReleaseGLObjects failed");
    error = (*clEnqueueReleaseGLObjects_ptr)(queue, 1, &stream1, 1, &kernEvent,
                                             &relEvent2);
    test_error(error, "clEnqueueReleaseGLObjects failed");

    evts[0] = relEvent1;
    evts[1] = relEvent2;
    error = clWaitForEvents(2, evts);
    test_error(error, "Unable to wait for release events");

    return 0;
}

class RunThread : public genericThread {
public:
    cl_kernel mKernel;
    cl_command_queue mQueue;
    cl_mem mStream0, mStream1;
    cl_int mRowIdx;
    cl_event mFenceEvent;
    size_t mNumThreads;

    RunThread(cl_kernel kernel, cl_command_queue queue, cl_mem stream0,
              cl_mem stream1, size_t numThreads)
        : mKernel(kernel), mQueue(queue), mStream0(stream0), mStream1(stream1),
          mNumThreads(numThreads)
    {}

    void SetRunData(cl_int rowIdx, cl_event fenceEvent)
    {
        mRowIdx = rowIdx;
        mFenceEvent = fenceEvent;
    }

    virtual void *IRun(void)
    {
        cl_int error = run_cl_kernel(mKernel, mQueue, mStream0, mStream1,
                                     mRowIdx, mFenceEvent, mNumThreads);
        return (void *)(uintptr_t)error;
    }
};


int test_fence_sync_single(cl_device_id device, cl_context context,
                           cl_command_queue queue, bool separateThreads,
                           GLint rend_vs, GLint read_vs,
                           cl_device_id rend_device)
{
    int error;
    const int framebufferSize = 512;


    if (!is_extension_available(device, "cl_khr_gl_event"))
    {
        log_info("NOTE: cl_khr_gl_event extension not present on this device; "
                 "skipping fence sync test\n");
        return 0;
    }

    // Ask OpenCL for the platforms.  Warn if more than one platform found,
    // since this might not be the platform we want.  By default, we simply
    // use the first returned platform.

    cl_uint nplatforms;
    cl_platform_id platform;
    clGetPlatformIDs(0, NULL, &nplatforms);
    clGetPlatformIDs(1, &platform, NULL);

    if (nplatforms > 1)
    {
        log_info("clGetPlatformIDs returned multiple values.  This is not "
                 "an error, but might result in obtaining incorrect function "
                 "pointers if you do not want the first returned platform.\n");

        // Show them the platform name, in case it is a problem.

        size_t size;
        char *name;

        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &size);
        name = (char *)malloc(size);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, name, NULL);

        log_info("Using platform with name: %s \n", name);
        free(name);
    }

    clCreateEventFromGLsyncKHR_ptr =
        (clCreateEventFromGLsyncKHR_fn)clGetExtensionFunctionAddressForPlatform(
            platform, "clCreateEventFromGLsyncKHR");
    if (clCreateEventFromGLsyncKHR_ptr == NULL)
    {
        log_error("ERROR: Unable to run fence_sync test "
                  "(clCreateEventFromGLsyncKHR function not discovered!)\n");
        clCreateEventFromGLsyncKHR_ptr = (clCreateEventFromGLsyncKHR_fn)
            clGetExtensionFunctionAddressForPlatform(
                platform, "clCreateEventFromGLsyncAPPLE");
        return -1;
    }

#ifdef USING_ARB_sync
    char *gl_version_str = (char *)glGetString(GL_VERSION);
    float glCoreVersion;
    sscanf(gl_version_str, "%f", &glCoreVersion);
    if (glCoreVersion < 3.0f)
    {
        log_info(
            "OpenGL version %f does not support fence/sync! Skipping test.\n",
            glCoreVersion);
        return 0;
    }

#ifdef __APPLE__
    CGLContextObj currCtx = CGLGetCurrentContext();
    CGLPixelFormatObj pixFmt = CGLGetPixelFormat(currCtx);
    GLint val, screen;
    CGLGetVirtualScreen(currCtx, &screen);
    CGLDescribePixelFormat(pixFmt, screen, kCGLPFAOpenGLProfile, &val);
    if (val != kCGLOGLPVersion_3_2_Core)
    {
        log_error(
            "OpenGL context was not created with OpenGL version >= 3.0 profile "
            "even though platform supports it"
            "OpenGL profile %f does not support fence/sync! Skipping test.\n",
            glCoreVersion);
        return -1;
    }
#else
#ifdef _WIN32
    HDC hdc = wglGetCurrentDC();
    HGLRC hglrc = wglGetCurrentContext();
#else
    Display *dpy = glXGetCurrentDisplay();
    GLXDrawable drawable = glXGetCurrentDrawable();
    GLXContext ctx = glXGetCurrentContext();
#endif
#endif

    InitSyncFns();
#endif

#ifdef __APPLE__
    CGLSetVirtualScreen(CGLGetCurrentContext(), rend_vs);
#else
#ifdef _WIN32
    wglMakeCurrent(hdc, hglrc);
#else
    glXMakeCurrent(dpy, drawable, ctx);
#endif
#endif

    GLint posLoc, colLoc;
    GLuint shaderprogram = createShaderProgram(&posLoc, &colLoc);
    if (!shaderprogram)
    {
        log_error("Failed to create shader program\n");
        return -1;
    }

    float l = 0.0f;
    float r = framebufferSize;
    float b = 0.0f;
    float t = framebufferSize;

    float projMatrix[16] = { 2.0f / (r - l),
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             2.0f / (t - b),
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             -1.0f,
                             0.0f,
                             -(r + l) / (r - l),
                             -(t + b) / (t - b),
                             0.0f,
                             1.0f };

    glUseProgram(shaderprogram);
    GLuint projMatLoc = glGetUniformLocation(shaderprogram, "projMatrix");
    glUniformMatrix4fv(projMatLoc, 1, 0, projMatrix);
    glUseProgram(0);

    // Note: the framebuffer is just the target to verify our results against,
    // so we don't really care to go through all the possible formats in this
    // case
    glFramebufferWrapper glFramebuffer;
    glRenderbufferWrapper glRenderbuffer;
    error = CreateGLRenderbufferRaw(
        framebufferSize, 128, GL_COLOR_ATTACHMENT0_EXT, GL_RGBA, GL_RGBA,
        GL_UNSIGNED_INT_8_8_8_8_REV, &glFramebuffer, &glRenderbuffer);
    if (error != 0) return error;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBufferWrapper vtxBuffer, colorBuffer;
    glGenBuffers(1, &vtxBuffer);
    glGenBuffers(1, &colorBuffer);

    const int numHorizVertices = (framebufferSize * 64) + 1;

    glBindBuffer(GL_ARRAY_BUFFER, vtxBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * numHorizVertices * 2 * 4,
                 NULL, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * numHorizVertices * 2 * 4,
                 NULL, GL_STATIC_DRAW);

    // Now that the requisite objects are bound, we can attempt program
    // validation:

    glValidateProgram(shaderprogram);

    GLint logLength, status;
    glGetProgramiv(shaderprogram, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0)
    {
        GLchar *log = (GLchar *)malloc(logLength);
        glGetProgramInfoLog(shaderprogram, logLength, &logLength, log);
        log_info("Program validate log:\n%s", log);
        free(log);
    }

    glGetProgramiv(shaderprogram, GL_VALIDATE_STATUS, &status);
    if (status == 0)
    {
        log_error("Failed to validate program\n");
        return 0;
    }

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];

    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    updateBuffersKernel, "update"))
        return -1;

    streams[0] = (*clCreateFromGLBuffer_ptr)(context, CL_MEM_READ_WRITE,
                                             vtxBuffer, &error);
    test_error(error, "Unable to create CL buffer from GL vertex buffer");

    streams[1] = (*clCreateFromGLBuffer_ptr)(context, CL_MEM_READ_WRITE,
                                             colorBuffer, &error);
    test_error(error, "Unable to create CL buffer from GL color buffer");

    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set kernel arguments");

    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set kernel arguments");

    cl_int horizWrap = (cl_int)framebufferSize;
    error = clSetKernelArg(kernel, 2, sizeof(horizWrap), &horizWrap);
    test_error(error, "Unable to set kernel arguments");

    glViewport(0, 0, framebufferSize, framebufferSize);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    clEventWrapper fenceEvent;
    GLsync glFence = 0;

    // Do a loop through 8 different horizontal stripes against the framebuffer
    RunThread thread(kernel, queue, streams[0], streams[1],
                     (size_t)numHorizVertices);

    for (int i = 0; i < 8; i++)
    {
        // if current rendering device is not the compute device and
        // separateThreads == false which means compute is going on same
        // thread and we are using implicit synchronization (no GLSync obj used)
        // then glFlush by clEnqueueAcquireGLObject is not sufficient ... we
        // need to wait for rendering to finish on other device before CL can
        // start writing to CL/GL shared mem objects. When separateThreads is
        // true i.e. we are using GLSync obj to synchronize then we dont need to
        // call glFinish here since CL should wait for rendering on other device
        // before this GLSync object to finish before it starts writing to
        // shared mem object. Also rend_device == compute_device no need to call
        // glFinish
        if (rend_device != device && !separateThreads) glFinish();

        if (separateThreads)
        {
            glDeleteSyncFunc(glFence);

            glFence = glFenceSyncFunc(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            fenceEvent =
                clCreateEventFromGLsyncKHR_ptr(context, glFence, &error);
            test_error(error, "Unable to create CL event from GL fence");

            // in case of explicit synchronization, we just wait for the sync
            // object to complete in clEnqueueAcquireGLObject but we dont flush.
            // Its application's responsibility to flush on the context on which
            // glSync is created
            glFlush();

            thread.SetRunData((cl_int)i, fenceEvent);
            thread.Start();

            error = (cl_int)(size_t)thread.Join();
        }
        else
        {
            error =
                run_cl_kernel(kernel, queue, streams[0], streams[1], (cl_int)i,
                              fenceEvent, (size_t)numHorizVertices);
        }
        test_error(error, "Unable to run CL kernel");

        glUseProgram(shaderprogram);
        glEnableVertexAttribArray(posLoc);
        glEnableVertexAttribArray(colLoc);
        glBindBuffer(GL_ARRAY_BUFFER, vtxBuffer);
        glVertexAttribPointer(posLoc, 4, GL_FLOAT, GL_FALSE,
                              4 * sizeof(GLfloat), 0);
        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glVertexAttribPointer(colLoc, 4, GL_FLOAT, GL_FALSE,
                              4 * sizeof(GLfloat), 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, numHorizVertices * 2);

        glDisableVertexAttribArray(posLoc);
        glDisableVertexAttribArray(colLoc);
        glUseProgram(0);

        if (separateThreads)
        {
            // If we're on the same thread, then we're testing implicit syncing,
            // so we don't need the actual fence code
            glDeleteSyncFunc(glFence);


            glFence = glFenceSyncFunc(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            fenceEvent =
                clCreateEventFromGLsyncKHR_ptr(context, glFence, &error);
            test_error(error, "Unable to create CL event from GL fence");

            // in case of explicit synchronization, we just wait for the sync
            // object to complete in clEnqueueAcquireGLObject but we dont flush.
            // Its application's responsibility to flush on the context on which
            // glSync is created
            glFlush();
        }
        else
            glFinish();
    }

    if (glFence != 0)
        // Don't need the final release for fenceEvent, because the wrapper will
        // take care of that
        glDeleteSyncFunc(glFence);

#ifdef __APPLE__
    CGLSetVirtualScreen(CGLGetCurrentContext(), read_vs);
#else
#ifdef _WIN32
    wglMakeCurrent(hdc, hglrc);
#else
    glXMakeCurrent(dpy, drawable, ctx);
#endif
#endif
    // Grab the contents of the final framebuffer
    BufferOwningPtr<char> resultData(ReadGLRenderbuffer(
        glFramebuffer, glRenderbuffer, GL_COLOR_ATTACHMENT0_EXT, GL_RGBA,
        GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, kUChar, framebufferSize, 128));

    // Check the contents now. We should end up with solid color bands 32 pixels
    // high and the full width of the framebuffer, at values (128,128,128) due
    // to the additive blending
    for (int i = 0; i < 8; i++)
    {
        for (int y = 0; y < 4; y++)
        {
            // Note: coverage will be double because the 63-0 triangle
            // overwrites again at the end of the pass
            cl_uchar valA =
                (((i + 1) & 1)) * numHorizVertices * 2 / framebufferSize;
            cl_uchar valB =
                (((i + 1) & 2) >> 1) * numHorizVertices * 2 / framebufferSize;
            cl_uchar valC =
                (((i + 1) & 4) >> 2) * numHorizVertices * 2 / framebufferSize;

            cl_uchar *row =
                (cl_uchar *)&resultData[(i * 16 + y) * framebufferSize * 4];
            for (int x = 0; x < (framebufferSize - 1) - 1; x++)
            {
                if ((row[x * 4] != valA) || (row[x * 4 + 1] != valB)
                    || (row[x * 4 + 2] != valC))
                {
                    log_error("ERROR: Output framebuffer did not validate!\n");
                    DumpGLBuffer(GL_UNSIGNED_BYTE, framebufferSize, 128,
                                 resultData);
                    log_error("RUNS:\n");
                    uint32_t *p = (uint32_t *)(char *)resultData;
                    size_t a = 0;
                    for (size_t t = 1; t < framebufferSize * framebufferSize;
                         t++)
                    {
                        if (p[a] != 0)
                        {
                            if (p[t] == 0)
                            {
                                log_error(
                                    "RUN: %ld to %ld (%d,%d to %d,%d) 0x%08x\n",
                                    a, t - 1, (int)(a % framebufferSize),
                                    (int)(a / framebufferSize),
                                    (int)((t - 1) % framebufferSize),
                                    (int)((t - 1) / framebufferSize), p[a]);
                                a = t;
                            }
                        }
                        else
                        {
                            if (p[t] != 0)
                            {
                                a = t;
                            }
                        }
                    }
                    return -1;
                }
            }
        }
    }

    destroyShaderProgram(shaderprogram);
    glDeleteVertexArrays(1, &vao);
    return 0;
}

int test_fence_sync(cl_device_id device, cl_context context,
                    cl_command_queue queue, int numElements)
{
    GLint vs_count = 0;
    cl_device_id *device_list = NULL;

    if (!is_extension_available(device, "cl_khr_gl_event"))
    {
        log_info("NOTE: cl_khr_gl_event extension not present on this device; "
                 "skipping fence sync test\n");
        return 0;
    }
#ifdef __APPLE__
    CGLContextObj ctx = CGLGetCurrentContext();
    CGLPixelFormatObj pix = CGLGetPixelFormat(ctx);
    CGLError err =
        CGLDescribePixelFormat(pix, 0, kCGLPFAVirtualScreenCount, &vs_count);

    device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * vs_count);
    clGetGLContextInfoAPPLE(context, ctx,
                            CL_CGL_DEVICES_FOR_SUPPORTED_VIRTUAL_SCREENS_APPLE,
                            sizeof(cl_device_id) * vs_count, device_list, NULL);
#else
    // Need platform specific way of getting devices from CL context to which
    // OpenGL can render If not available it can be replaced with
    // clGetContextInfo with CL_CONTEXT_DEVICES
    size_t device_cb;
    cl_int err =
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &device_cb);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Unable to get device count from context");
        return -1;
    }
    vs_count = (GLint)device_cb / sizeof(cl_device_id);

    if (vs_count < 1)
    {
        log_error("No devices found.\n");
        return -1;
    }

    device_list = (cl_device_id *)malloc(device_cb);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, device_cb, device_list,
                           NULL);
    if (err != CL_SUCCESS)
    {
        free(device_list);
        print_error(err, "Unable to get device list from context");
        return -1;
    }

#endif

    GLint rend_vs, read_vs;
    int error = 0;
    int any_failed = 0;

    // Loop through all the devices capable to OpenGL rendering
    // and set them as current rendering target
    for (rend_vs = 0; rend_vs < vs_count; rend_vs++)
    {
        // Loop through all the devices and set them as current
        // compute target
        for (read_vs = 0; read_vs < vs_count; read_vs++)
        {
            cl_device_id rend_device = device_list[rend_vs],
                         read_device = device_list[read_vs];
            char rend_name[200], read_name[200];

            clGetDeviceInfo(rend_device, CL_DEVICE_NAME, sizeof(rend_name),
                            rend_name, NULL);
            clGetDeviceInfo(read_device, CL_DEVICE_NAME, sizeof(read_name),
                            read_name, NULL);

            log_info("Rendering on: %s, read back on: %s\n", rend_name,
                     read_name);
            error = test_fence_sync_single(device, context, queue, false,
                                           rend_vs, read_vs, rend_device);
            any_failed |= error;
            if (error != 0)
                log_error(
                    "ERROR: Implicit syncing with GL sync events failed!\n\n");
            else
                log_info("Implicit syncing Passed\n");

            error = test_fence_sync_single(device, context, queue, true,
                                           rend_vs, read_vs, rend_device);
            any_failed |= error;
            if (error != 0)
                log_error(
                    "ERROR: Explicit syncing with GL sync events failed!\n\n");
            else
                log_info("Explicit syncing Passed\n");
        }
    }

    free(device_list);

    return any_failed;
}
