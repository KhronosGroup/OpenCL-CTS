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



cl_int get_type_size( cl_context context, cl_command_queue queue, const char *type, cl_ulong *size, cl_device_id device  )
{
    const char *sizeof_kernel_code[4] =
    {
        "", /* optional pragma string */
        "__kernel __attribute__((reqd_work_group_size(1,1,1))) void test_sizeof(__global uint *dst) \n"
        "{\n"
        "   dst[0] = (uint) sizeof( ", type, " );\n"
        "}\n"
    };

    clProgramWrapper p;
    clKernelWrapper k;
    clMemWrapper m;
    cl_uint        temp;


    if (!strncmp(type, "double", 6))
    {
        sizeof_kernel_code[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }
    else if (!strncmp(type, "half", 4))
    {
        sizeof_kernel_code[0] = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    cl_int err = create_single_kernel_helper_with_build_options(
        context, &p, &k, 4, sizeof_kernel_code, "test_sizeof", nullptr);
    test_error(err, "Failed to build kernel/program.");

    m = clCreateBuffer( context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof( cl_ulong ), size, &err );
    test_error(err, "clCreateBuffer failed.");

    err = clSetKernelArg( k, 0, sizeof( cl_mem ), &m );
    test_error(err, "clSetKernelArg failed.");

    err = clEnqueueTask( queue, k, 0, NULL, NULL );
    test_error(err, "clEnqueueTask failed.");

    err = clEnqueueReadBuffer( queue, m, CL_TRUE, 0, sizeof( cl_uint ), &temp, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed.");

    *size = (cl_ulong) temp;

    return err;
}

typedef struct size_table
{
    const char *name;
    cl_ulong   size;
    cl_ulong   cl_size;
}size_table;

const size_table  scalar_table[] =
{
    // Fixed size entries from table 6.1
    {  "char",              1,  sizeof( cl_char )   },
    {  "uchar",             1,  sizeof( cl_uchar)   },
    {  "unsigned char",     1,  sizeof( cl_uchar)   },
    {  "short",             2,  sizeof( cl_short)   },
    {  "ushort",            2,  sizeof( cl_ushort)  },
    {  "unsigned short",    2,  sizeof( cl_ushort)  },
    {  "int",               4,  sizeof( cl_int )    },
    {  "uint",              4,  sizeof( cl_uint)    },
    {  "unsigned int",      4,  sizeof( cl_uint)    },
    {  "float",             4,  sizeof( cl_float)   },
    {  "long",              8,  sizeof( cl_long )   },
    {  "ulong",             8,  sizeof( cl_ulong)   },
    {  "unsigned long",     8,  sizeof( cl_ulong)   }
};

const size_table  vector_table[] =
{
    // Fixed size entries from table 6.1
    {  "char",      1,  sizeof( cl_char )   },
    {  "uchar",     1,  sizeof( cl_uchar)   },
    {  "short",     2,  sizeof( cl_short)   },
    {  "ushort",    2,  sizeof( cl_ushort)  },
    {  "int",       4,  sizeof( cl_int )    },
    {  "uint",      4,  sizeof( cl_uint)    },
    {  "float",     4,  sizeof( cl_float)   },
    {  "long",      8,  sizeof( cl_long )   },
    {  "ulong",     8,  sizeof( cl_ulong)   }
};

const char  *ptr_table[] =
{
    "global void*",
    "size_t",
    "sizeof(int)",      // check return type of sizeof
    "ptrdiff_t"
};

const char *other_types[] =
{
    "event_t",
    "image2d_t",
    "image3d_t",
    "sampler_t"
};

static int IsPowerOfTwo( cl_ulong x ){ return 0 == (x & (x-1)); }

int test_sizeof(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t i, j;
    cl_ulong test;
    cl_uint ptr_size = CL_UINT_MAX;
    cl_int err = CL_SUCCESS;

    // Check address space size
    err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(ptr_size), &ptr_size, NULL);
    if( err || ptr_size > 64)
    {
        log_error( "FAILED:  Unable to get CL_DEVICE_ADDRESS_BITS for device %p\n", device );
        return -1;
    }
    log_info( "\tCL_DEVICE_ADDRESS_BITS = %u\n", ptr_size );
    ptr_size /= 8;

    // Test standard scalar sizes
    for( i = 0; i < sizeof( scalar_table ) / sizeof( scalar_table[0] ); i++ )
    {
        if( ! gHasLong &&
           (0 == strcmp(scalar_table[i].name, "long") ||
            0 == strcmp(scalar_table[i].name, "ulong") ||
            0 == strcmp(scalar_table[i].name, "unsigned long")))
        {
            log_info("\nLongs are not supported by this device. Skipping test.\t");
            continue;
        }

        test = CL_ULONG_MAX;
        err = get_type_size( context, queue, scalar_table[i].name, &test, device);
        if( err )
            return err;
        if( test != scalar_table[i].size )
        {
            log_error( "\nFAILED: Type %s has size %lld, but expected size %lld!\n", scalar_table[i].name, test, scalar_table[i].size );
            return -1;
        }
        if( test != scalar_table[i].cl_size )
        {
            log_error( "\nFAILED: Type %s has size %lld, but cl_ size is %lld!\n", scalar_table[i].name, test, scalar_table[i].cl_size );
            return -2;
        }
        log_info( "%16s", scalar_table[i].name );
    }
    log_info( "\n" );

    // Test standard vector sizes
    for( j = 2; j <= 16; j *= 2 )
    {
        // For each vector size, iterate through types
        for( i = 0; i < sizeof( vector_table ) / sizeof( vector_table[0] ); i++ )
        {
            if( !gHasLong &&
               (0 == strcmp(vector_table[i].name, "long") ||
                0 == strcmp(vector_table[i].name, "ulong")))
            {
                log_info("\nLongs are not supported by this device. Skipping test.\t");
                continue;
            }

            char name[32];
            sprintf( name, "%s%ld", vector_table[i].name, j );

            test = CL_ULONG_MAX;
            err = get_type_size( context, queue, name, &test, device  );
            if( err )
                return err;
            if( test != j * vector_table[i].size )
            {
                log_error( "\nFAILED: Type %s has size %lld, but expected size %lld!\n", name, test, j * vector_table[i].size );
                return -1;
            }
            if( test != j * vector_table[i].cl_size )
            {
                log_error( "\nFAILED: Type %s has size %lld, but cl_ size is %lld!\n", name, test, j * vector_table[i].cl_size );
                return -2;
            }
            log_info( "%16s", name );
        }
        log_info( "\n" );
    }

    //Check that pointer sizes are correct
    for( i = 0; i < sizeof( ptr_table ) / sizeof( ptr_table[0] ); i++ )
    {
        test = CL_ULONG_MAX;
        err = get_type_size( context, queue, ptr_table[i], &test, device );
        if( err )
            return err;
        if( test != ptr_size )
        {
            log_error( "\nFAILED: Type %s has size %lld, but expected size %u!\n", ptr_table[i], test, ptr_size );
            return -1;
        }
        log_info( "%16s", ptr_table[i] );
    }

    // Check that intptr_t is large enough
    test = CL_ULONG_MAX;
    err = get_type_size( context, queue, "intptr_t", &test, device );
    if( err )
        return err;
    if( test < ptr_size )
    {
        log_error( "\nFAILED: intptr_t has size %lld, but must be at least %u!\n", test, ptr_size );
        return -1;
    }
    if( ! IsPowerOfTwo( test ) )
    {
        log_error( "\nFAILED: sizeof(intptr_t) is %lld, but must be a power of two!\n", test );
        return -2;
    }
    log_info( "%16s", "intptr_t" );

    // Check that uintptr_t is large enough
    test = CL_ULONG_MAX;
    err = get_type_size( context, queue, "uintptr_t", &test, device );
    if( err )
        return err;
    if( test < ptr_size )
    {
        log_error( "\nFAILED: uintptr_t has size %lld, but must be at least %u!\n", test, ptr_size );
        return -1;
    }
    if( ! IsPowerOfTwo( test ) )
    {
        log_error( "\nFAILED: sizeof(uintptr_t) is %lld, but must be a power of two!\n", test );
        return -2;
    }
    log_info( "%16s\n", "uintptr_t" );

    //Check that other types are powers of two
    for( i = 0; i < sizeof( other_types ) / sizeof( other_types[0] ); i++ )
    {
        if( 0 == strcmp(other_types[i], "image2d_t") &&
           checkForImageSupport( device ) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
            log_info("\nimages are not supported by this device. Skipping test.\t");
            continue;
        }

        if (0 == strcmp(other_types[i], "image3d_t")
            && checkFor3DImageSupport(device) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
            log_info("\n3D images are not supported by this device. "
                     "Skipping test.\t");
            continue;
        }

        if( 0 == strcmp(other_types[i], "sampler_t") &&
           checkForImageSupport( device ) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
          log_info("\nimages are not supported by this device. Skipping test.\t");
          continue;
        }

        test = CL_ULONG_MAX;
        err = get_type_size( context, queue, other_types[i], &test, device );
        if( err )
            return err;
        if( ! IsPowerOfTwo( test ) )
        {
            log_error( "\nFAILED: Type %s has size %lld, which is not a power of two (section 6.1.5)!\n", other_types[i], test );
            return -1;
        }
        log_info( "%16s", other_types[i] );
    }
    log_info( "\n" );


    //Check double
    if( is_extension_available( device, "cl_khr_fp64" ) )
    {
        log_info( "\tcl_khr_fp64:" );
        test = CL_ULONG_MAX;
        err = get_type_size( context, queue, "double", &test, device );
        if( err )
            return err;
        if( test != 8 )
        {
            log_error( "\nFAILED: double has size %lld, but must be 8!\n", test );
            return -1;
        }
        log_info( "%16s", "double" );

        // Test standard vector sizes
        for( j = 2; j <= 16; j *= 2 )
        {
            char name[32];
            sprintf( name, "double%ld", j );

            test = CL_ULONG_MAX;
            err = get_type_size( context, queue, name, &test, device );
            if( err )
                return err;
            if( test != 8*j )
            {
                log_error( "\nFAILED: %s has size %lld, but must be %ld!\n", name, test, 8 * j);
                return -1;
            }
            log_info( "%16s", name );
        }
        log_info( "\n" );
    }

    //Check half
    if( is_extension_available( device, "cl_khr_fp16" ) )
    {
        log_info( "\tcl_khr_fp16:" );
        test = CL_ULONG_MAX;
        err = get_type_size( context, queue, "half", &test, device );
        if( err )
            return err;
        if( test != 2 )
        {
            log_error( "\nFAILED: half has size %lld, but must be 2!\n", test );
            return -1;
        }
        log_info( "%16s", "half" );

        // Test standard vector sizes
        for( j = 2; j <= 16; j *= 2 )
        {
            char name[32];
            sprintf( name, "half%ld", j );

            test = CL_ULONG_MAX;
            err = get_type_size( context, queue, name, &test, device );
            if( err )
                return err;
            if( test != 2*j )
            {
                log_error( "\nFAILED: %s has size %lld, but must be %ld!\n", name, test, 2 * j);
                return -1;
            }
            log_info( "%16s", name );
        }
        log_info( "\n" );
    }

    return err;
}


