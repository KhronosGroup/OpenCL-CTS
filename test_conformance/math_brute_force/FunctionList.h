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
#ifndef FUNCTIONLIST_H
#define FUNCTIONLIST_H

#include "harness/compat.h"

#ifndef WIN32
#include <unistd.h>
#endif

#if defined( __APPLE__ )
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include "harness/mt19937.h"

typedef union fptr
{
    void    *p;
    double  (*f_f)(double);
    double  (*f_u)(cl_uint);
    int     (*i_f)(double);
    int     (*i_f_f)(float);
    float   (*f_ff_f)(float, float);
    double  (*f_ff)(double, double);
    int     (*i_ff)(double, double);
    double  (*f_fi)(double, int);
    double  (*f_fpf)(double, double*);
    double  (*f_fpI)(double, int*);
    double  (*f_ffpI)(double, double, int*);
    double  (*f_fff)(double, double, double );
    float   (*f_fma)(float, float, float, int);
}fptr;

typedef union dptr
{
    void            *p;
    long double     (*f_f)(long double);
    long double     (*f_u)(cl_ulong);
    int             (*i_f)(long double);
    long double     (*f_ff)(long double, long double);
    int             (*i_ff)(long double, long double);
    long double     (*f_fi)(long double, int);
    long double     (*f_fpf)(long double, long double*);
    long double     (*f_fpI)(long double, int*);
    long double     (*f_ffpI)(long double, long double, int*);
    long double     (*f_fff)(long double, long double, long double);
}dptr;

struct Func;

typedef struct vtbl
{
    const char  *type_name;
    int         (*TestFunc)( const struct Func *, MTdata );
    int         (*DoubleTestFunc)( const struct Func *, MTdata);        // may be NULL if function is single precision only
}vtbl;

typedef struct Func
{
  const char      *name;              // common name, to be used as an argument in the shell
  const char      *nameInCode;        // name as it appears in the __kernel, usually the same as name, but different for multiplication
  fptr            func;
  dptr            dfunc;
  fptr            rfunc;
  float           float_ulps;
  float           double_ulps;
  float           float_embedded_ulps;
  float           relaxed_error;
  int             ftz;
  int             relaxed;
  const vtbl      *vtbl_ptr;
}Func;


extern const Func  functionList[];

extern const size_t functionListCount;

#endif


