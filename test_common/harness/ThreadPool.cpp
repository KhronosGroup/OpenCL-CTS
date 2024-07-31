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
#include "ThreadPool.h"
#include "errorHelpers.h"
#include "fpcontrol.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
// or any other POSIX system

#include <atomic>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "mingw_compat.h"
#include <process.h>
#else // !_WIN32
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
#endif // !_WIN32

// declarations
#ifdef _WIN32
void ThreadPool_WorkerFunc(void *p);
#else
void *ThreadPool_WorkerFunc(void *p);
#endif
void ThreadPool_Init(void);
void ThreadPool_Exit(void);

#if defined(__MINGW32__)
// Mutex for implementing super heavy atomic operations if you don't have GCC or
// MSVC
CRITICAL_SECTION gAtomicLock;
#elif defined(__GNUC__) || defined(_MSC_VER)
#else
pthread_mutex_t gAtomicLock;
#endif

#if !defined(_WIN32)
// Keep track of pthread_t's created in ThreadPool_Init() so they can be joined
// in ThreadPool_Exit() and avoid thread leaks.
static std::vector<pthread_t> pthreads;
#endif

// Atomic add operator with mem barrier.  Mem barrier needed to protect state
// modified by the worker functions.
cl_int ThreadPool_AtomicAdd(volatile cl_int *a, cl_int b)
{
#if defined(__MINGW32__)
    // No atomics on Mingw32
    EnterCriticalSection(&gAtomicLock);
    cl_int old = *a;
    *a = old + b;
    LeaveCriticalSection(&gAtomicLock);
    return old;
#elif defined(__GNUC__)
    // GCC extension:
    // http://gcc.gnu.org/onlinedocs/gcc/Atomic-Builtins.html#Atomic-Builtins
    return __sync_fetch_and_add(a, b);
    // do we need __sync_synchronize() here, too?  GCC docs are unclear whether
    // __sync_fetch_and_add does a synchronize
#elif defined(_MSC_VER)
    return (cl_int)_InterlockedExchangeAdd((volatile LONG *)a, (LONG)b);
#else
#warning Please add a atomic add implementation here, with memory barrier.  Fallback code is slow.
    if (pthread_mutex_lock(&gAtomicLock))
        log_error("Atomic operation failed. pthread_mutex_lock(&gAtomicLock) "
                  "returned an error\n");
    cl_int old = *a;
    *a = old + b;
    if (pthread_mutex_unlock(&gAtomicLock))
        log_error("Failed to release gAtomicLock. Further atomic operations "
                  "may deadlock!\n");
    return old;
#endif
}

#if defined(_WIN32)
// Uncomment the following line if Windows XP support is not required.
// #define HAS_INIT_ONCE_EXECUTE_ONCE 1

#if defined(HAS_INIT_ONCE_EXECUTE_ONCE)
#define _INIT_ONCE INIT_ONCE
#define _PINIT_ONCE PINIT_ONCE
#define _InitOnceExecuteOnce InitOnceExecuteOnce
#else // !HAS_INIT_ONCE_EXECUTE_ONCE

typedef volatile LONG _INIT_ONCE;
typedef _INIT_ONCE *_PINIT_ONCE;
typedef BOOL(CALLBACK *_PINIT_ONCE_FN)(_PINIT_ONCE, PVOID, PVOID *);

#define _INIT_ONCE_UNINITIALIZED 0
#define _INIT_ONCE_IN_PROGRESS 1
#define _INIT_ONCE_DONE 2

static BOOL _InitOnceExecuteOnce(_PINIT_ONCE InitOnce, _PINIT_ONCE_FN InitFn,
                                 PVOID Parameter, LPVOID *Context)
{
    while (*InitOnce != _INIT_ONCE_DONE)
    {
        if (*InitOnce != _INIT_ONCE_IN_PROGRESS
            && _InterlockedCompareExchange(InitOnce, _INIT_ONCE_IN_PROGRESS,
                                           _INIT_ONCE_UNINITIALIZED)
                == _INIT_ONCE_UNINITIALIZED)
        {
            InitFn(InitOnce, Parameter, Context);
            *InitOnce = _INIT_ONCE_DONE;
            return TRUE;
        }
        Sleep(1);
    }
    return TRUE;
}
#endif // !HAS_INIT_ONCE_EXECUTE_ONCE

// Uncomment the following line if Windows XP support is not required.
// #define HAS_CONDITION_VARIABLE 1

#if defined(HAS_CONDITION_VARIABLE)
#define _CONDITION_VARIABLE CONDITION_VARIABLE
#define _InitializeConditionVariable InitializeConditionVariable
#define _SleepConditionVariableCS SleepConditionVariableCS
#define _WakeAllConditionVariable WakeAllConditionVariable
#else // !HAS_CONDITION_VARIABLE
typedef struct
{
    HANDLE mEvent; // Used to park the thread.
    // Used to protect mWaiters, mGeneration and mReleaseCount:
    CRITICAL_SECTION mLock[1];
    volatile cl_int mWaiters; // Number of threads waiting on this cond var.
    volatile cl_int mGeneration; // Wait generation count.
    volatile cl_int mReleaseCount; // Number of releases to execute before
                                   // reseting the event.
} _CONDITION_VARIABLE;

typedef _CONDITION_VARIABLE *_PCONDITION_VARIABLE;

static void _InitializeConditionVariable(_PCONDITION_VARIABLE cond_var)
{
    cond_var->mEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    InitializeCriticalSection(cond_var->mLock);
    cond_var->mWaiters = 0;
    cond_var->mGeneration = 0;
#if !defined(NDEBUG)
    cond_var->mReleaseCount = 0;
#endif // !NDEBUG
}

static void _SleepConditionVariableCS(_PCONDITION_VARIABLE cond_var,
                                      PCRITICAL_SECTION cond_lock,
                                      DWORD ignored)
{
    EnterCriticalSection(cond_var->mLock);
    cl_int generation = cond_var->mGeneration;
    ++cond_var->mWaiters;
    LeaveCriticalSection(cond_var->mLock);
    LeaveCriticalSection(cond_lock);

    while (TRUE)
    {
        WaitForSingleObject(cond_var->mEvent, INFINITE);
        EnterCriticalSection(cond_var->mLock);
        BOOL done =
            cond_var->mReleaseCount > 0 && cond_var->mGeneration != generation;
        LeaveCriticalSection(cond_var->mLock);
        if (done)
        {
            break;
        }
    }

    EnterCriticalSection(cond_lock);
    EnterCriticalSection(cond_var->mLock);
    if (--cond_var->mReleaseCount == 0)
    {
        ResetEvent(cond_var->mEvent);
    }
    --cond_var->mWaiters;
    LeaveCriticalSection(cond_var->mLock);
}

static void _WakeAllConditionVariable(_PCONDITION_VARIABLE cond_var)
{
    EnterCriticalSection(cond_var->mLock);
    if (cond_var->mWaiters > 0)
    {
        ++cond_var->mGeneration;
        cond_var->mReleaseCount = cond_var->mWaiters;
        SetEvent(cond_var->mEvent);
    }
    LeaveCriticalSection(cond_var->mLock);
}
#endif // !HAS_CONDITION_VARIABLE
#endif // _WIN32

#define MAX_COUNT (1 << 29)

// Global state to coordinate whether the threads have been launched
// successfully or not
#if defined(_MSC_VER) && (_WIN32_WINNT >= 0x600)
static _INIT_ONCE threadpool_init_control;
#elif defined(_WIN32) // MingW of XP
static int threadpool_init_control;
#else // Posix platforms
pthread_once_t threadpool_init_control = PTHREAD_ONCE_INIT;
#endif
cl_int threadPoolInitErr = -1; // set to CL_SUCCESS on successful thread launch

// critical region lock around ThreadPool_Do.  We can only run one ThreadPool_Do
// at a time, because we are too lazy to set up a queue here, and don't expect
// to need one.
#if defined(_WIN32)
CRITICAL_SECTION gThreadPoolLock[1];
#else // !_WIN32
pthread_mutex_t gThreadPoolLock;
#endif // !_WIN32

// Condition variable to park ThreadPool threads when not working
#if defined(_WIN32)
CRITICAL_SECTION cond_lock[1];
_CONDITION_VARIABLE cond_var[1];
#else // !_WIN32
pthread_mutex_t cond_lock;
pthread_cond_t cond_var;
#endif // !_WIN32

// Condition variable state. How many iterations on the function left to run,
// set to CL_INT_MAX to cause worker threads to exit. Note: this value might
// go negative.
std::atomic<cl_int> gRunCount{ 0 };

// State that only changes when the threadpool is not working.
volatile TPFuncPtr gFunc_ptr = NULL;
volatile void *gUserInfo = NULL;
volatile cl_int gJobCount = 0;

// State that may change while the thread pool is working
volatile cl_int jobError = CL_SUCCESS; // err code return for the job as a whole

// Condition variable to park caller while waiting
#if defined(_WIN32)
HANDLE caller_event;
#else // !_WIN32
pthread_mutex_t caller_cond_lock;
pthread_cond_t caller_cond_var;
#endif // !_WIN32

// # of threads intended to be running. Running threads will decrement this
// as they discover they've run out of work to do.
std::atomic<cl_int> gRunning{ 0 };

// The total number of threads launched.
std::atomic<cl_int> gThreadCount{ 0 };

#ifdef _WIN32
void ThreadPool_WorkerFunc(void *p)
#else
void *ThreadPool_WorkerFunc(void *p)
#endif
{
    auto &tid = *static_cast<std::atomic<cl_uint> *>(p);
    cl_uint threadID = tid++;
    cl_int item = gRunCount--;

    while (MAX_COUNT > item)
    {
        cl_int err;

        // check for more work to do
        if (0 >= item)
        {
            // No work to do. Attempt to block waiting for work
#if defined(_WIN32)
            EnterCriticalSection(cond_lock);
#else // !_WIN32
            if ((err = pthread_mutex_lock(&cond_lock)))
            {
                log_error(
                    "Error %d from pthread_mutex_lock. Worker %d unable to "
                    "block waiting for work. ThreadPool_WorkerFunc failed.\n",
                    err, threadID);
                goto exit;
            }
#endif // !_WIN32

            cl_int remaining = gRunning--;
            if (1 == remaining)
            { // last thread out signal the main thread to wake up
#if defined(_WIN32)
                SetEvent(caller_event);
#else // !_WIN32
                if ((err = pthread_mutex_lock(&caller_cond_lock)))
                {
                    log_error("Error %d from pthread_mutex_lock. Unable to "
                              "wake caller.\n",
                              err);
                    goto exit;
                }
                if ((err = pthread_cond_broadcast(&caller_cond_var)))
                {
                    log_error(
                        "Error %d from pthread_cond_broadcast. Unable to wake "
                        "up main thread. ThreadPool_WorkerFunc failed.\n",
                        err);
                    goto exit;
                }
                if ((err = pthread_mutex_unlock(&caller_cond_lock)))
                {
                    log_error("Error %d from pthread_mutex_lock. Unable to "
                              "wake caller.\n",
                              err);
                    goto exit;
                }
#endif // !_WIN32
            }

            // loop in case we are woken only to discover that some other thread
            // already did all the work
            while (0 >= item)
            {
#if defined(_WIN32)
                _SleepConditionVariableCS(cond_var, cond_lock, INFINITE);
#else // !_WIN32
                if ((err = pthread_cond_wait(&cond_var, &cond_lock)))
                {
                    log_error(
                        "Error %d from pthread_cond_wait. Unable to block for "
                        "waiting for work. ThreadPool_WorkerFunc failed.\n",
                        err);
                    pthread_mutex_unlock(&cond_lock);
                    goto exit;
                }
#endif // !_WIN32

                // try again to get a valid item id
                item = gRunCount--;
                if (MAX_COUNT <= item) // exit if we are done
                {
#if defined(_WIN32)
                    LeaveCriticalSection(cond_lock);
#else // !_WIN32
                    pthread_mutex_unlock(&cond_lock);
#endif // !_WIN32
                    goto exit;
                }
            }

            gRunning++;

#if defined(_WIN32)
            LeaveCriticalSection(cond_lock);
#else // !_WIN32
            if ((err = pthread_mutex_unlock(&cond_lock)))
            {
                log_error(
                    "Error %d from pthread_mutex_unlock. Unable to block for "
                    "waiting for work. ThreadPool_WorkerFunc failed.\n",
                    err);
                goto exit;
            }
#endif // !_WIN32
        }

        // we have a valid item, so do the work
        // but only if we haven't already encountered an error
        if (CL_SUCCESS == jobError)
        {
            // log_info("Thread %d doing job %d\n", threadID, item - 1);

#if defined(__APPLE__) && defined(__arm__)
            // On most platforms which support denorm, default is FTZ off.
            // However, on some hardware where the reference is computed,
            // default might be flush denorms to zero e.g. arm. This creates
            // issues in result verification. Since spec allows the
            // implementation to either flush or not flush denorms to zero, an
            // implementation may choose not be flush i.e. return denorm result
            // whereas reference result may be zero (flushed denorm). Hence we
            // need to disable denorm flushing on host side where reference is
            // being computed to make sure we get non-flushed reference result.
            // If implementation returns flushed result, we correctly take care
            // of that in verification code.
            FPU_mode_type oldMode;
            DisableFTZ(&oldMode);
#endif

            // Call the user's function with this item ID
            err = gFunc_ptr(item - 1, threadID, (void *)gUserInfo);
#if defined(__APPLE__) && defined(__arm__)
            // Restore FP state
            RestoreFPState(&oldMode);
#endif

            if (err)
            {
#if (__MINGW32__)
                EnterCriticalSection(&gAtomicLock);
                if (jobError == CL_SUCCESS) jobError = err;
                gRunCount = 0;
                LeaveCriticalSection(&gAtomicLock);
#elif defined(__GNUC__)
                // GCC extension:
                // http://gcc.gnu.org/onlinedocs/gcc/Atomic-Builtins.html#Atomic-Builtins
                // set the new error if we are the first one there.
                __sync_val_compare_and_swap(&jobError, CL_SUCCESS, err);

                // drop run count to 0
                gRunCount = 0;
                __sync_synchronize();
#elif defined(_MSC_VER)
                // set the new error if we are the first one there.
                _InterlockedCompareExchange((volatile LONG *)&jobError, err,
                                            CL_SUCCESS);

                // drop run count to 0
                gRunCount = 0;
                _mm_mfence();
#else
                if (pthread_mutex_lock(&gAtomicLock))
                    log_error(
                        "Atomic operation failed. "
                        "pthread_mutex_lock(&gAtomicLock) returned an error\n");
                if (jobError == CL_SUCCESS) jobError = err;
                gRunCount = 0;
                if (pthread_mutex_unlock(&gAtomicLock))
                    log_error("Failed to release gAtomicLock. Further atomic "
                              "operations may deadlock\n");
#endif
            }
        }

        // get the next item
        item = gRunCount--;
    }

exit:
    log_info("ThreadPool: thread %d exiting.\n", threadID);
    gThreadCount--;
#if !defined(_WIN32)
    return NULL;
#endif
}

// SetThreadCount() may be used to artifically set the number of worker threads
// If the value is 0 (the default) the number of threads will be determined
// based on the number of CPU cores.  If it is a unicore machine, then 2 will be
// used, so that we still get some testing for thread safety.
//
// If count < 2 or the CL_TEST_SINGLE_THREADED environment variable is set then
// the code will run single threaded, but will report an error to indicate that
// the test is invalid.  This option is intended for debugging purposes only. It
// is suggested as a convention that test apps set the thread count to 1 in
// response to the -m flag.
//
// SetThreadCount() must be called before the first call to GetThreadCount() or
// ThreadPool_Do(), otherwise the behavior is indefined.
void SetThreadCount(int count)
{
    if (threadPoolInitErr == CL_SUCCESS)
    {
        log_error("Error: It is illegal to set the thread count after the "
                  "first call to ThreadPool_Do or GetThreadCount\n");
        abort();
    }

    gThreadCount = count;
}

void ThreadPool_Init(void)
{
    cl_int i;
    int err;
    std::atomic<cl_uint> threadID{ 0 };

    // Check for manual override of multithreading code. We add this for better
    // debuggability.
    if (getenv("CL_TEST_SINGLE_THREADED"))
    {
        log_error("ERROR: CL_TEST_SINGLE_THREADED is set in the environment. "
                  "Running single threaded.\n*** TEST IS INVALID! ***\n");
        gThreadCount = 1;
        return;
    }

    // Figure out how many threads to run -- check first for non-zero to give
    // the implementation the chance
    if (0 == gThreadCount)
    {
#if defined(_MSC_VER) || defined(__MINGW64__)
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
        DWORD length = 0;

        GetLogicalProcessorInformation(NULL, &length);
        buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(length);
        if (buffer != NULL)
        {
            if (GetLogicalProcessorInformation(buffer, &length) == TRUE)
            {
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
                while (
                    ptr
                    < &buffer[length
                              / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)])
                {
                    if (ptr->Relationship == RelationProcessorCore)
                    {
                        // Count the number of bits in ProcessorMask (number of
                        // logical cores)
                        ULONG_PTR mask = ptr->ProcessorMask;
                        while (mask)
                        {
                            ++gThreadCount;
                            mask &= mask - 1; // Remove 1 bit at a time
                        }
                    }
                    ++ptr;
                }
            }
            free(buffer);
        }
#elif defined(__MINGW32__)
        {
#warning How about this, instead of hard coding it to 2?
            SYSTEM_INFO sysinfo;
            GetSystemInfo(&sysinfo);
            gThreadCount = sysinfo.dwNumberOfProcessors;
        }
#elif defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t affinity;
        if (0 == sched_getaffinity(0, sizeof(cpu_set_t), &affinity))
        {
#if !(defined(CPU_COUNT))
            gThreadCount = 1;
#else
            gThreadCount = CPU_COUNT(&affinity);
#endif
        }
        else
        {
            // Hopefully your system returns logical cpus here, as does MacOS X
            gThreadCount = (cl_int)sysconf(_SC_NPROCESSORS_CONF);
        }
#else /* !_WIN32 */
        // Hopefully your system returns logical cpus here, as does MacOS X
        gThreadCount = (cl_int)sysconf(_SC_NPROCESSORS_CONF);
#endif // !_WIN32

        // Multithreaded tests are required to run multithreaded even on unicore
        // systems so as to test thread safety
        if (1 == gThreadCount) gThreadCount = 2;
    }

// When working in 32 bit limit the thread number to 12
// This fix was made due to memory issues in integer_ops test
// When running integer_ops, the test opens as many threads as the
// machine has and each thread allocates a fixed amount of memory
// When running this test on dual socket machine in 32-bit, the
// process memory is not sufficient and the test fails
#if defined(_WIN32) && !defined(_M_X64)
    if (gThreadCount > 12)
    {
        gThreadCount = 12;
    }
#endif

    // Allow the app to set thread count to <0 for debugging purposes.
    // This will cause the test to run single threaded.
    if (gThreadCount < 2)
    {
        log_error("ERROR: Running single threaded because thread count < 2. "
                  "\n*** TEST IS INVALID! ***\n");
        gThreadCount = 1;
        return;
    }

#if defined(_WIN32)
    InitializeCriticalSection(gThreadPoolLock);
    InitializeCriticalSection(cond_lock);
    _InitializeConditionVariable(cond_var);
    caller_event = CreateEvent(NULL, FALSE, FALSE, NULL);
#elif defined(__GNUC__)
    // Dont rely on PTHREAD_MUTEX_INITIALIZER for intialization of a mutex since
    // it might cause problem with some flavors of gcc compilers.
    pthread_cond_init(&cond_var, NULL);
    pthread_mutex_init(&cond_lock, NULL);
    pthread_cond_init(&caller_cond_var, NULL);
    pthread_mutex_init(&caller_cond_lock, NULL);
    pthread_mutex_init(&gThreadPoolLock, NULL);
#endif

#if !(defined(__GNUC__) || defined(_MSC_VER) || defined(__MINGW32__))
    pthread_mutex_initialize(gAtomicLock);
#elif defined(__MINGW32__)
    InitializeCriticalSection(&gAtomicLock);
#endif
    // Make sure the last thread done in the work pool doesn't signal us to wake
    // before we get to the point where we are supposed to wait
    //  That would cause a deadlock.
#if !defined(_WIN32)
    if ((err = pthread_mutex_lock(&caller_cond_lock)))
    {
        log_error("Error %d from pthread_mutex_lock. Unable to block for work "
                  "to finish. ThreadPool_Init failed.\n",
                  err);
        gThreadCount = 1;
        return;
    }
#endif // !_WIN32

    gRunning = gThreadCount.load();
    // init threads
    for (i = 0; i < gThreadCount; i++)
    {
#if defined(_WIN32)
        uintptr_t handle =
            _beginthread(ThreadPool_WorkerFunc, 0, (void *)&threadID);
        err = (handle == 0);
#else // !_WIN32
        pthread_t tid = 0;
        err = pthread_create(&tid, NULL, ThreadPool_WorkerFunc,
                             (void *)&threadID);
#endif // !_WIN32
        if (err)
        {
            log_error("Error %d launching thread %d\n", err, i);
            threadPoolInitErr = err;
            gThreadCount = i;
            break;
        }
#if !defined(_WIN32)
        pthreads.push_back(tid);
#endif // !_WIN32
    }

    atexit(ThreadPool_Exit);

    // block until they are done launching.
    do
    {
#if defined(_WIN32)
        WaitForSingleObject(caller_event, INFINITE);
#else // !_WIN32
        if ((err = pthread_cond_wait(&caller_cond_var, &caller_cond_lock)))
        {
            log_error("Error %d from pthread_cond_wait. Unable to block for "
                      "work to finish. ThreadPool_Init failed.\n",
                      err);
            pthread_mutex_unlock(&caller_cond_lock);
            return;
        }
#endif // !_WIN32
    } while (gRunCount != -gThreadCount);
#if !defined(_WIN32)
    if ((err = pthread_mutex_unlock(&caller_cond_lock)))
    {
        log_error("Error %d from pthread_mutex_unlock. Unable to block for "
                  "work to finish. ThreadPool_Init failed.\n",
                  err);
        return;
    }
#endif // !_WIN32

    threadPoolInitErr = CL_SUCCESS;
}

#if defined(_MSC_VER)
static BOOL CALLBACK _ThreadPool_Init(_PINIT_ONCE InitOnce, PVOID Parameter,
                                      PVOID *lpContex)
{
    ThreadPool_Init();
    return TRUE;
}
#endif

void ThreadPool_Exit(void)
{
    gRunCount = CL_INT_MAX;

#if defined(__GNUC__)
    // GCC extension:
    // http://gcc.gnu.org/onlinedocs/gcc/Atomic-Builtins.html#Atomic-Builtins
    __sync_synchronize();
#elif defined(_MSC_VER)
    _mm_mfence();
#else
#warning If this is a weakly ordered memory system, please add a memory barrier here to force this and everything else to memory before we proceed
#endif

    // spin waiting for threads to die
    for (int count = 0; 0 != gThreadCount && count < 1000; count++)
    {
#if defined(_WIN32)
        _WakeAllConditionVariable(cond_var);
        Sleep(1);
#else // !_WIN32
        if (int err = pthread_cond_broadcast(&cond_var))
        {
            log_error("Error %d from pthread_cond_broadcast. Unable to wake up "
                      "work threads. ThreadPool_Exit failed.\n",
                      err);
            break;
        }
        usleep(1000);
#endif // !_WIN32
    }

    if (gThreadCount)
        log_error("Error: Thread pool timed out after 1 second with %d threads "
                  "still active.\n",
                  gThreadCount.load());
    else
    {
#if !defined(_WIN32)
        for (pthread_t pthread : pthreads)
        {
            if (int err = pthread_join(pthread, nullptr))
            {
                log_error("Error from %d from pthread_join. Unable to join "
                          "work threads. ThreadPool_Exit failed.\n",
                          err);
            }
        }
#endif
        log_info("Thread pool exited in a orderly fashion.\n");
    }
}


// Blocking API that farms out count jobs to a thread pool.
// It may return with some work undone if func_ptr() returns a non-zero
// result.
//
// This function obviously has its shortcommings. Only one call to ThreadPool_Do
// can be running at a time. It is not intended for general purpose use.
// If clEnqueueNativeKernelFn, out of order queues and a CL_DEVICE_TYPE_CPU were
// all available then it would make more sense to use those features.
cl_int ThreadPool_Do(TPFuncPtr func_ptr, cl_uint count, void *userInfo)
{
#ifndef _WIN32
    cl_int newErr;
#endif
    cl_int err = 0;
    // Lazily set up our threads
#if defined(_MSC_VER) && (_WIN32_WINNT >= 0x600)
    err = !_InitOnceExecuteOnce(&threadpool_init_control, _ThreadPool_Init,
                                NULL, NULL);
#elif defined(_WIN32)
    if (threadpool_init_control == 0)
    {
#warning This is buggy and race prone.  Find a better way.
        ThreadPool_Init();
        threadpool_init_control = 1;
    }
#else // posix platform
    err = pthread_once(&threadpool_init_control, ThreadPool_Init);
    if (err)
    {
        log_error("Error %d from pthread_once. Unable to init threads. "
                  "ThreadPool_Do failed.\n",
                  err);
        return err;
    }
#endif
    // Single threaded code to handle case where threadpool wasn't allocated or
    // was disabled by environment variable
    if (threadPoolInitErr)
    {
        cl_uint currentJob = 0;
        cl_int result = CL_SUCCESS;

#if defined(__APPLE__) && defined(__arm__)
        // On most platforms which support denorm, default is FTZ off. However,
        // on some hardware where the reference is computed, default might be
        // flush denorms to zero e.g. arm. This creates issues in result
        // verification. Since spec allows the implementation to either flush or
        // not flush denorms to zero, an implementation may choose not be flush
        // i.e. return denorm result whereas reference result may be zero
        // (flushed denorm). Hence we need to disable denorm flushing on host
        // side where reference is being computed to make sure we get
        // non-flushed reference result. If implementation returns flushed
        // result, we correctly take care of that in verification code.
        FPU_mode_type oldMode;
        DisableFTZ(&oldMode);
#endif
        for (currentJob = 0; currentJob < count; currentJob++)
            if ((result = func_ptr(currentJob, 0, userInfo)))
            {
#if defined(__APPLE__) && defined(__arm__)
                // Restore FP state before leaving
                RestoreFPState(&oldMode);
#endif
                return result;
            }

#if defined(__APPLE__) && defined(__arm__)
        // Restore FP state before leaving
        RestoreFPState(&oldMode);
#endif

        return CL_SUCCESS;
    }

    if (count >= MAX_COUNT)
    {
        log_error(
            "Error: ThreadPool_Do count %d >= max threadpool count of %d\n",
            count, MAX_COUNT);
        return -1;
    }

    // Enter critical region
#if defined(_WIN32)
    EnterCriticalSection(gThreadPoolLock);
#else // !_WIN32
    if ((err = pthread_mutex_lock(&gThreadPoolLock)))
    {
        switch (err)
        {
            case EDEADLK:
                log_error(
                    "Error EDEADLK returned in ThreadPool_Do(). ThreadPool_Do "
                    "is not designed to work recursively!\n");
                break;
            case EINVAL:
                log_error("Error EINVAL returned in ThreadPool_Do(). How did "
                          "we end up with an invalid gThreadPoolLock?\n");
                break;
            default: break;
        }
        return err;
    }
#endif // !_WIN32

    // Start modifying the job state observable by worker threads
#if defined(_WIN32)
    EnterCriticalSection(cond_lock);
#else // !_WIN32
    if ((err = pthread_mutex_lock(&cond_lock)))
    {
        log_error("Error %d from pthread_mutex_lock. Unable to wake up work "
                  "threads. ThreadPool_Do failed.\n",
                  err);
        goto exit;
    }
#endif // !_WIN32

    // Make sure the last thread done in the work pool doesn't signal us to wake
    // before we get to the point where we are supposed to wait
    //  That would cause a deadlock.
#if !defined(_WIN32)
    if ((err = pthread_mutex_lock(&caller_cond_lock)))
    {
        log_error("Error %d from pthread_mutex_lock. Unable to block for work "
                  "to finish. ThreadPool_Do failed.\n",
                  err);
        goto exit;
    }
#endif // !_WIN32

    // Prime the worker threads to get going
    jobError = CL_SUCCESS;
    gRunCount = gJobCount = count;
    gFunc_ptr = func_ptr;
    gUserInfo = userInfo;

#if defined(_WIN32)
    ResetEvent(caller_event);
    _WakeAllConditionVariable(cond_var);
    LeaveCriticalSection(cond_lock);
#else // !_WIN32
    if ((err = pthread_cond_broadcast(&cond_var)))
    {
        log_error("Error %d from pthread_cond_broadcast. Unable to wake up "
                  "work threads. ThreadPool_Do failed.\n",
                  err);
        goto exit;
    }
    if ((err = pthread_mutex_unlock(&cond_lock)))
    {
        log_error("Error %d from pthread_mutex_unlock. Unable to wake up work "
                  "threads. ThreadPool_Do failed.\n",
                  err);
        goto exit;
    }
#endif // !_WIN32

    // block until they are done.  It would be slightly more efficient to do
    // some of the work here though.
    do
    {
#if defined(_WIN32)
        WaitForSingleObject(caller_event, INFINITE);
#else // !_WIN32
        if ((err = pthread_cond_wait(&caller_cond_var, &caller_cond_lock)))
        {
            log_error("Error %d from pthread_cond_wait. Unable to block for "
                      "work to finish. ThreadPool_Do failed.\n",
                      err);
            pthread_mutex_unlock(&caller_cond_lock);
            goto exit;
        }
#endif // !_WIN32
    } while (gRunning);
#if !defined(_WIN32)
    if ((err = pthread_mutex_unlock(&caller_cond_lock)))
    {
        log_error("Error %d from pthread_mutex_unlock. Unable to block for "
                  "work to finish. ThreadPool_Do failed.\n",
                  err);
        goto exit;
    }
#endif // !_WIN32

    err = jobError;

#ifndef _WIN32
exit:
#endif
    // exit critical region
#if defined(_WIN32)
    LeaveCriticalSection(gThreadPoolLock);
#else // !_WIN32
    newErr = pthread_mutex_unlock(&gThreadPoolLock);
    if (newErr)
    {
        log_error("Error %d from pthread_mutex_unlock. Unable to exit critical "
                  "region. ThreadPool_Do failed.\n",
                  newErr);
        return err;
    }
#endif // !_WIN32

    return err;
}

cl_uint GetThreadCount(void)
{
    // Lazily set up our threads
#if defined(_MSC_VER) && (_WIN32_WINNT >= 0x600)
    cl_int err = !_InitOnceExecuteOnce(&threadpool_init_control,
                                       _ThreadPool_Init, NULL, NULL);
#elif defined(_WIN32)
    if (threadpool_init_control == 0)
    {
#warning This is buggy and race prone.  Find a better way.
        ThreadPool_Init();
        threadpool_init_control = 1;
    }
#else
    cl_int err = pthread_once(&threadpool_init_control, ThreadPool_Init);
    if (err)
    {
        log_error("Error %d from pthread_once. Unable to init threads. "
                  "ThreadPool_Do failed.\n",
                  err);
        return err;
    }
#endif // !_WIN32

    if (gThreadCount < 1) return 1;

    return gThreadCount;
}

#else

#ifndef MY_OS_REALLY_REALLY_DOESNT_SUPPORT_THREADS
#error ThreadPool implementation has not been multithreaded for this operating system. You must multithread this section.
#endif
//
// We require multithreading in parts of the test as a means of simultaneously
// testing reentrancy requirements of OpenCL API, while also checking
//
// A sample single threaded implementation follows, for documentation /
// bootstrapping purposes. It is not okay to use this for conformance testing!!!
//
// Exception:  If your operating system does not support multithreaded execution
// of any kind, then you may use this code.
//

cl_int ThreadPool_AtomicAdd(volatile cl_int *a, cl_int b)
{
    cl_uint r = *a;

    // since this fallback code path is not multithreaded, we just do a regular
    // add here. If your operating system supports memory-barrier-atomics, use
    // those here.
    *a = r + b;

    return r;
}

// Blocking API that farms out count jobs to a thread pool.
// It may return with some work undone if func_ptr() returns a non-zero
// result.
cl_int ThreadPool_Do(TPFuncPtr func_ptr, cl_uint count, void *userInfo)
{
    cl_uint currentJob = 0;
    cl_int result = CL_SUCCESS;

#ifndef MY_OS_REALLY_REALLY_DOESNT_SUPPORT_THREADS
    // THIS FUNCTION IS NOT INTENDED FOR USE!!
    log_error("ERROR:  Test must be multithreaded!\n");
    exit(-1);
#else
    static int spewCount = 0;

    if (0 == spewCount)
    {
        log_info("\nWARNING:  The operating system is claimed not to support "
                 "threads of any sort. Running single threaded.\n");
        spewCount = 1;
    }
#endif

    // The multithreaded code should mimic this behavior:
    for (currentJob = 0; currentJob < count; currentJob++)
        if ((result = func_ptr(currentJob, 0, userInfo))) return result;

    return CL_SUCCESS;
}

cl_uint GetThreadCount(void) { return 1; }

void SetThreadCount(int count)
{
    if (count > 1) log_info("WARNING: SetThreadCount(%d) ignored\n", count);
}

#endif
