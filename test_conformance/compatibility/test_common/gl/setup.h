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
#ifndef _setup_h
#define _setup_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gl_headers.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


// Note: the idea here is to have every platform define their own setup.cpp file that implements a GLEnvironment
// subclass internally, then return it as a definition for GLEnvironment::Create

class GLEnvironment
{
    public:
        GLEnvironment() {}
        virtual ~GLEnvironment() {}

         virtual int Init( int *argc, char **argv, int use_opengl_32 ) = 0;
        virtual cl_context CreateCLContext( void ) = 0;
        virtual int SupportsCLGLInterop( cl_device_type device_type) = 0;

        static GLEnvironment *    Instance( void );


};

#endif // _setup_h
