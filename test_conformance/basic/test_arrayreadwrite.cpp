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


int
test_arrayreadwrite(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_uint                *inptr, *outptr;
    cl_mem              streams[1];
    int                 num_tries = 400;
    num_elements = 1024 * 1024 * 4;
    int                 i, j, err;
    MTdata              d;

    inptr = (cl_uint*)malloc(num_elements*sizeof(cl_uint));
    outptr = (cl_uint*)malloc(num_elements*sizeof(cl_uint));

    // randomize data
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        inptr[i] = (cl_uint)(genrand_int32(d) & 0x7FFFFFFF);

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_uint) * num_elements, NULL, &err);
    test_error(err, "clCreateBuffer failed");

    for (i=0; i<num_tries; i++)
    {
        int        offset;
        int        cb;

        do {
            offset = (int)(genrand_int32(d) & 0x7FFFFFFF);
            if (offset > 0 && offset < num_elements)
                break;
        } while (1);
        cb = (int)(genrand_int32(d) & 0x7FFFFFFF);
        if (cb > (num_elements - offset))
            cb = num_elements - offset;

        err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, offset*sizeof(cl_uint), sizeof(cl_uint)*cb,&inptr[offset], 0, NULL, NULL);
        test_error(err, "clEnqueueWriteBuffer failed");

        err = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, offset*sizeof(cl_uint), cb*sizeof(cl_uint), &outptr[offset], 0, NULL, NULL );
        test_error(err, "clEnqueueReadBuffer failed");

        for (j=offset; j<offset+cb; j++)
        {
            if (inptr[j] != outptr[j])
            {
                log_error("ARRAY read, write test failed\n");
                err = -1;
                break;
            }
        }

        if (err)
            break;
    }

    free_mtdata(d);
    clReleaseMemObject(streams[0]);
    free(inptr);
    free(outptr);

    if (!err)
        log_info("ARRAY read, write test passed\n");

    return err;
}



