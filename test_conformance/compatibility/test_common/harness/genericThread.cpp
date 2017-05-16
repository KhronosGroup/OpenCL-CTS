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
#include "genericThread.h"

#if defined(_WIN32)
#include <windows.h>
#else // !_WIN32
#include <pthread.h>
#endif

void * genericThread::IStaticReflector( void * data )
{
    genericThread *t = (genericThread *)data;
    return t->IRun();
}

bool genericThread::Start( void )
{
#if defined(_WIN32)
    mHandle = CreateThread( NULL, 0, (LPTHREAD_START_ROUTINE) IStaticReflector, this, 0, NULL );
    return ( mHandle != NULL );
#else // !_WIN32
    int error = pthread_create( (pthread_t*)&mHandle, NULL, IStaticReflector, (void *)this );
    return ( error == 0 );
#endif // !_WIN32
}

void * genericThread::Join( void )
{
#if defined(_WIN32)
    WaitForSingleObject( (HANDLE)mHandle, INFINITE );
    return NULL;
#else // !_WIN32
    void * retVal;
    int error = pthread_join( (pthread_t)mHandle, &retVal );
    if( error != 0 )
        retVal = NULL;
    return retVal;
#endif // !_WIN32
}
