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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"



int test_array_info_size( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem          memobj;
    cl_int          err;
    size_t          w = 32, h = 32, d = 32;
    size_t          retSize;
    size_t          elementSize = sizeof( cl_int );

    memobj = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  elementSize * w*h*d, NULL, &err);
    test_error(err, "clCreateBuffer failed.");

    err = clGetMemObjectInfo(memobj, CL_MEM_SIZE, sizeof( size_t ), (void *)&retSize, NULL);
    if ( err ){
        log_error( "Error calling clGetMemObjectInfo(): %d\n", err );
        clReleaseMemObject(memobj);
        return -1;
    }
    if ( (elementSize * w * h * d) != retSize ) {
        log_error( "Error in clGetMemObjectInfo() check of size\n" );
        clReleaseMemObject(memobj);
        return -1;
    }
    else{
        log_info( " CL_MEM_SIZE passed.\n" );
    }

    // cleanup
    clReleaseMemObject(memobj);

    return err;

}   // end testArrayElementSize()


// FIXME: need to test other flags

