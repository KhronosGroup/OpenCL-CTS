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
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#if defined( __APPLE__ )
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

//
// An atomic add operator
cl_int     ThreadPool_AtomicAdd( volatile cl_int *a, cl_int b );    // returns old value

// Your function prototype
//
// A function pointer to the function you want to execute in a multithreaded context.  No
// synchronization primitives are provided, other than the atomic add above. You may not
// call ThreadPool_Do from your function. ThreadPool_AtomicAdd() and GetThreadCount() should
// work, however.
//
// job ids and thread ids are 0 based.  If number of jobs or threads was 8, they will numbered be 0 through 7.
// Note that while every job will be run, it is not guaranteed that every thread will wake up before
// the work is done.
typedef cl_int (*TPFuncPtr)( cl_uint /*job_id*/, cl_uint /* thread_id */, void *userInfo );

// returns first non-zero result from func_ptr, or CL_SUCCESS if all are zero.
// Some workitems may not run if a non-zero result is returned from func_ptr().
// This function may not be called from a TPFuncPtr.
cl_int      ThreadPool_Do(  TPFuncPtr func_ptr,
                            cl_uint count,
                            void *userInfo );

// Returns the number of worker threads that underlie the threadpool.  The value passed
// as the TPFuncPtrs thread_id will be between 0 and this value less one, inclusive.
// This is safe to call from a TPFuncPtr.
cl_uint     GetThreadCount( void );

// SetThreadCount() may be used to artifically set the number of worker threads
// If the value is 0 (the default) the number of threads will be determined based on
// the number of CPU cores.  If it is a unicore machine, then 2 will be used, so
// that we still get some testing for thread safety.
//
// If count < 2 or the CL_TEST_SINGLE_THREADED environment variable is set then the
// code will run single threaded, but will report an error to indicate that the test
// is invalid.  This option is intended for debugging purposes only. It is suggested
// as a convention that test apps set the thread count to 1 in response to the -m flag.
//
// SetThreadCount() must be called before the first call to GetThreadCount() or ThreadPool_Do(),
// otherwise the behavior is indefined. It may not be called from a TPFuncPtr.
void        SetThreadCount( int count );


#endif  /* THREAD_POOL_H  */
