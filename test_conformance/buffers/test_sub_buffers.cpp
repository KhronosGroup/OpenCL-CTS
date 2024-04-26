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
#include "procs.h"

#include <algorithm>
#include <vector>

// Design:
// To test sub buffers, we first create one main buffer. We then create several sub-buffers and
// queue Actions on each one. Each Action is encapsulated in a class so it can keep track of
// what results it expects, and so we can test scaling degrees of Actions on scaling numbers of
// sub-buffers.

class SubBufferWrapper : public clMemWrapper
{
public:
    cl_mem mParentBuffer;
    size_t mOrigin;
    size_t mSize;

    cl_int Allocate( cl_mem parent, cl_mem_flags flags, size_t origin, size_t size )
    {
        mParentBuffer = parent;
        mOrigin = origin;
        mSize = size;

        cl_buffer_region region;
        region.origin = mOrigin;
        region.size = mSize;

        cl_int error;
        reset(clCreateSubBuffer(mParentBuffer, flags,
                                CL_BUFFER_CREATE_TYPE_REGION, &region, &error));
        return error;
    }
};

class Action
{
public:
    virtual ~Action() {}
    virtual cl_int Execute( cl_context context, cl_command_queue queue, cl_char tag, SubBufferWrapper &buffer1, SubBufferWrapper &buffer2, cl_char *parentBufferState ) = 0;
    virtual const char * GetName( void ) const = 0;

    static MTdata d;
    static MTdata GetRandSeed( void )
    {
        if ( d == 0 )
            d = init_genrand( gRandomSeed );
        return d;
    }
    static void FreeRandSeed() {
        if ( d != 0 ) {
            free_mtdata(d);
            d = 0;
        }
    }
};

MTdata Action::d = 0;

class ReadWriteAction : public Action
{
public:
    virtual ~ReadWriteAction() {}
    virtual const char * GetName( void ) const { return "ReadWrite";}

    virtual cl_int Execute( cl_context context, cl_command_queue queue, cl_char tag, SubBufferWrapper &buffer1, SubBufferWrapper &buffer2, cl_char *parentBufferState )
    {
        cl_char *tempBuffer = (cl_char*)malloc(buffer1.mSize);
        if (!tempBuffer) {
            log_error("Out of memory\n");
            return -1;
        }
        cl_int error = clEnqueueReadBuffer( queue, buffer1, CL_TRUE, 0, buffer1.mSize, tempBuffer, 0, NULL, NULL );
        test_error( error, "Unable to enqueue buffer read" );

        size_t start = get_random_size_t( 0, buffer1.mSize / 2, GetRandSeed() );
        size_t end = get_random_size_t( start, buffer1.mSize, GetRandSeed() );

        for ( size_t i = start; i < end; i++ )
        {
            tempBuffer[ i ] |= tag;
            parentBufferState[ i + buffer1.mOrigin ] |= tag;
        }

        error = clEnqueueWriteBuffer( queue, buffer1, CL_TRUE, 0, buffer1.mSize, tempBuffer, 0, NULL, NULL );
        test_error( error, "Unable to enqueue buffer write" );
        free(tempBuffer);
        return CL_SUCCESS;
    }
};

class CopyAction : public Action
{
public:
    virtual ~CopyAction() {}
    virtual const char * GetName( void ) const { return "Copy";}

    virtual cl_int Execute( cl_context context, cl_command_queue queue, cl_char tag, SubBufferWrapper &buffer1, SubBufferWrapper &buffer2, cl_char *parentBufferState )
    {
        // Copy from sub-buffer 1 to sub-buffer 2
        size_t size = get_random_size_t(
            0, std::min(buffer1.mSize, buffer2.mSize), GetRandSeed());

        size_t startOffset = get_random_size_t( 0, buffer1.mSize - size, GetRandSeed() );
        size_t endOffset = get_random_size_t( 0, buffer2.mSize - size, GetRandSeed() );

        cl_int error = clEnqueueCopyBuffer( queue, buffer1, buffer2, startOffset, endOffset, size, 0, NULL, NULL );
        test_error( error, "Unable to enqueue buffer copy" );

        memcpy( parentBufferState + buffer2.mOrigin + endOffset, parentBufferState + buffer1.mOrigin + startOffset, size );

        return CL_SUCCESS;
    }
};

class MapAction : public Action
{
public:
    virtual ~MapAction() {}
    virtual const char * GetName( void ) const { return "Map";}

    virtual cl_int Execute( cl_context context, cl_command_queue queue, cl_char tag, SubBufferWrapper &buffer1, SubBufferWrapper &buffer2, cl_char *parentBufferState )
    {
        size_t size = get_random_size_t( 0, buffer1.mSize, GetRandSeed() );
        size_t start = get_random_size_t( 0, buffer1.mSize - size, GetRandSeed() );

        cl_int error;
        void * mappedPtr = clEnqueueMapBuffer( queue, buffer1, CL_TRUE, (cl_map_flags)( CL_MAP_READ | CL_MAP_WRITE ),
                                               start, size, 0, NULL, NULL, &error );
        test_error( error, "Unable to map buffer" );

        cl_char *cPtr = (cl_char *)mappedPtr;
        for ( size_t i = 0; i < size; i++ )
        {
            cPtr[ i ] |= tag;
            parentBufferState[ i + start + buffer1.mOrigin ] |= tag;
        }

        error = clEnqueueUnmapMemObject( queue, buffer1, mappedPtr, 0, NULL, NULL );
        test_error( error, "Unable to unmap buffer" );

        return CL_SUCCESS;
    }
};

class KernelReadWriteAction : public Action
{
public:
    virtual ~KernelReadWriteAction() {}
    virtual const char * GetName( void ) const { return "KernelReadWrite";}

    virtual cl_int Execute( cl_context context, cl_command_queue queue, cl_char tag, SubBufferWrapper &buffer1, SubBufferWrapper &buffer2, cl_char *parentBufferState )
    {
        const char *kernelCode[] = {
            "__kernel void readTest( __global char *inBuffer, char tag )\n"
            "{\n"
            "    int tid = get_global_id(0);\n"
            "    inBuffer[ tid ] |= tag;\n"
            "}\n" };

        clProgramWrapper program;
        clKernelWrapper kernel;
        cl_int error;

        if ( create_single_kernel_helper( context, &program, &kernel, 1, kernelCode, "readTest" ) )
        {
            return -1;
        }

        size_t threads[1] = { buffer1.mSize };

        error = clSetKernelArg( kernel, 0, sizeof( cl_mem ), &buffer1 );
        test_error( error, "Unable to set kernel argument" );
        error = clSetKernelArg( kernel, 1, sizeof( tag ), &tag );
        test_error( error, "Unable to set kernel argument" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Unable to queue kernel" );

        for ( size_t i = 0; i < buffer1.mSize; i++ )
            parentBufferState[ i + buffer1.mOrigin ] |= tag;

        return CL_SUCCESS;
    }
};

cl_int get_reasonable_buffer_size( cl_device_id device, size_t &outSize )
{
    cl_ulong maxAllocSize;
    cl_int error;

    // Get the largest possible buffer we could allocate
    error = clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get max alloc size" );

    // Don't create a buffer quite that big, just so we have some space left over for other work
    outSize = (size_t)( maxAllocSize / 5 );

    // Cap at 32M so tests complete in a reasonable amount of time.
    if ( outSize > 32 << 20 )
        outSize = 32 << 20;

    return CL_SUCCESS;
}

size_t find_subbuffer_by_index( SubBufferWrapper * subBuffers, size_t numSubBuffers, size_t index )
{
    for ( size_t i = 0; i < numSubBuffers; i++ )
    {
        if ( subBuffers[ i ].mOrigin > index )
            return numSubBuffers;
        if ( ( subBuffers[ i ].mOrigin <= index ) && ( ( subBuffers[ i ].mOrigin + subBuffers[ i ].mSize ) > index ) )
            return i;
    }
    return numSubBuffers;
}

// This tests the read/write capabilities of sub buffers (if we are read/write, the sub buffers
// can't overlap)
int test_sub_buffers_read_write_core( cl_context context, cl_command_queue queueA, cl_command_queue queueB, size_t mainSize, size_t addressAlign )
{
    clMemWrapper mainBuffer;
    SubBufferWrapper subBuffers[ 8 ];
    size_t numSubBuffers;
    cl_int error;
    size_t i;
    MTdata m = init_genrand( 22 );


    cl_char * mainBufferContents = (cl_char*)calloc(1,mainSize);
    cl_char * actualResults      = (cl_char*)calloc(1,mainSize);

    for ( i = 0; i < mainSize / 4; i++ )
        ((cl_uint*) mainBufferContents)[i] = genrand_int32(m);

    free_mtdata( m );

    // Create the main buffer to test against
    mainBuffer = clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mainSize, mainBufferContents, &error );
    test_error( error, "Unable to create test main buffer" );

    // Create some sub-buffers to use
    size_t toStartFrom = 0;
    for ( numSubBuffers = 0; numSubBuffers < 8; numSubBuffers++ )
    {
        size_t endRange = toStartFrom + ( mainSize / 4 );
        if ( endRange > mainSize )
            endRange = mainSize;

        size_t offset = get_random_size_t( toStartFrom / addressAlign, endRange / addressAlign, Action::GetRandSeed() ) * addressAlign;
        size_t size =
            get_random_size_t(
                1, (std::min(mainSize / 8, mainSize - offset)) / addressAlign,
                Action::GetRandSeed())
            * addressAlign;
        error = subBuffers[ numSubBuffers ].Allocate( mainBuffer, CL_MEM_READ_WRITE, offset, size );
        test_error( error, "Unable to allocate sub buffer" );

        toStartFrom = offset + size;
        if ( toStartFrom > ( mainSize - ( addressAlign * 256 ) ) )
            break;
    }

    ReadWriteAction rwAction;
    MapAction mapAction;
    CopyAction copyAction;
    KernelReadWriteAction kernelAction;

    Action * actions[] = { &rwAction, &mapAction, &copyAction, &kernelAction };
    int numErrors = 0;

    // Do the following steps twice, to make sure the parent gets updated *and* we can
    // still work on the sub-buffers
    cl_command_queue prev_queue = queueA;
    for ( int time = 0; time < 2; time++ )
    {
        // Randomly apply actions to the set of sub buffers
        size_t i;
        for (  i = 0; i < 64; i++ )
        {
            int which = random_in_range( 0, 3, Action::GetRandSeed() );
            int whichQueue = random_in_range( 0, 1, Action::GetRandSeed() );
            int whichBufferA = random_in_range( 0, (int)numSubBuffers - 1, Action::GetRandSeed() );
            int whichBufferB;
            do
            {
                whichBufferB = random_in_range( 0, (int)numSubBuffers - 1, Action::GetRandSeed() );
            } while ( whichBufferB == whichBufferA );

            cl_command_queue queue = ( whichQueue == 1 ) ? queueB : queueA;
            if (queue != prev_queue) {
                error = clFinish( prev_queue );
                test_error( error, "Error finishing other queue." );

                prev_queue = queue;
            }

            error = actions[ which ]->Execute( context, queue, (cl_int)i, subBuffers[ whichBufferA ], subBuffers[ whichBufferB ], mainBufferContents );
            test_error( error, "Unable to execute action against sub buffers" );
        }

        error = clFinish( queueA );
        test_error( error, "Error finishing queueA." );

        error = clFinish( queueB );
        test_error( error, "Error finishing queueB." );

        // Validate by reading the final contents of the main buffer and
        // validating against our ref copy we generated
        error = clEnqueueReadBuffer( queueA, mainBuffer, CL_TRUE, 0, mainSize, actualResults, 0, NULL, NULL );
        test_error( error, "Unable to enqueue buffer read" );

        for ( i = 0; i < mainSize; i += 65536 )
        {
            size_t left = 65536;
            if ( ( i + left ) > mainSize )
                left = mainSize - i;

            if ( memcmp( actualResults + i, mainBufferContents + i, left ) == 0 )
                continue;

            // The fast compare failed, so we need to determine where exactly the failure is

            for ( size_t j = 0; j < left; j++ )
            {
                if ( actualResults[ i + j ] != mainBufferContents[ i + j ] )
                {
                    // Hit a failure; report the subbuffer at this address as having failed
                    size_t sbThatFailed = find_subbuffer_by_index( subBuffers, numSubBuffers, i + j );
                    if ( sbThatFailed == numSubBuffers )
                    {
                        log_error( "ERROR: Validation failure outside of a sub-buffer! (Shouldn't be possible, but it happened at index %ld out of %ld...)\n", i + j, mainSize );
                        // Since this is a nonsensical, don't bother continuing to check
                        // (we will, however, print our map of sub-buffers for comparison)
                        for ( size_t k = 0; k < numSubBuffers; k++ )
                        {
                            log_error( "\tBuffer %ld: %ld to %ld (length %ld)\n", k, subBuffers[ k ].mOrigin, subBuffers[ k ].mOrigin + subBuffers[ k ].mSize, subBuffers[ k ].mSize );
                        }
                        return -1;
                    }
                    log_error( "ERROR: Validation failure on sub-buffer %ld (start: %ld, length: %ld)\n", sbThatFailed, subBuffers[ sbThatFailed ].mOrigin, subBuffers[ sbThatFailed ].mSize );
                    size_t newPos = subBuffers[ sbThatFailed ].mOrigin + subBuffers[ sbThatFailed ].mSize - 1;
                    i = newPos & ~65535;
                    j = newPos - i;
                    numErrors++;
                }
            }
        }
    }

    free(mainBufferContents);
    free(actualResults);
    Action::FreeRandSeed();

    return numErrors;
}

int test_sub_buffers_read_write( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_int error;
    size_t mainSize;
    cl_uint addressAlignBits;

    // Get the size of the main buffer to use
    error = get_reasonable_buffer_size( deviceID, mainSize );
    test_error( error, "Unable to get reasonable buffer size" );

    // Determine the alignment of the device so we can make sure sub buffers are valid
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof( addressAlignBits ), &addressAlignBits, NULL );
    test_error( error, "Unable to get device's address alignment" );

    size_t addressAlign = addressAlignBits/8;

    return test_sub_buffers_read_write_core( context, queue, queue, mainSize, addressAlign );
}

// This test performs the same basic operations as sub_buffers_read_write, but instead of a single
// device, it creates a context and buffer shared between two devices, then executes commands
// on queues for each device to ensure that everything still operates as expected.
int test_sub_buffers_read_write_dual_devices( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_int error;


    // First obtain the second device
    cl_device_id otherDevice = GetOpposingDevice( deviceID );
    if ( otherDevice == NULL )
    {
        log_error( "ERROR: Unable to obtain a second device for sub-buffer dual-device test.\n" );
        return -1;
    }
    if ( otherDevice == deviceID )
    {
        log_info( "Note: Unable to run dual-device sub-buffer test (only one device available). Skipping test (implicitly passing).\n" );
        return 0;
    }

    // Determine the device id.
    size_t param_size;
    error = clGetDeviceInfo(otherDevice, CL_DEVICE_NAME, 0, NULL, &param_size );
    test_error( error, "Error obtaining device name" );
    std::vector<char> device_name(param_size);

    error = clGetDeviceInfo(otherDevice, CL_DEVICE_NAME, param_size, &device_name[0], NULL );
    test_error( error, "Error obtaining device name" );

    log_info("\tOther device obtained for dual device test is type %s\n",
             device_name.data());

    // Create a shared context for these two devices
    cl_device_id devices[ 2 ] = { deviceID, otherDevice };
    clContextWrapper testingContext = clCreateContext( NULL, 2, devices, NULL, NULL, &error );
    test_error( error, "Unable to create shared context" );

    // Create two queues (can't use the existing one, because it's on the wrong context)
    clCommandQueueWrapper queue1 = clCreateCommandQueue( testingContext, deviceID, 0, &error );
    test_error( error, "Unable to create command queue on main device" );

    clCommandQueueWrapper queue2 = clCreateCommandQueue( testingContext, otherDevice, 0, &error );
    test_error( error, "Unable to create command queue on secondary device" );

    // Determine the reasonable buffer size and address alignment that applies to BOTH devices
    size_t maxBuffer1, maxBuffer2;
    error = get_reasonable_buffer_size( deviceID, maxBuffer1 );
    test_error( error, "Unable to get buffer size for main device" );

    error = get_reasonable_buffer_size( otherDevice, maxBuffer2 );
    test_error( error, "Unable to get buffer size for secondary device" );
    maxBuffer1 = std::min(maxBuffer1, maxBuffer2);

    cl_uint addressAlign1Bits, addressAlign2Bits;
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof( addressAlign1Bits ), &addressAlign1Bits, NULL );
    test_error( error, "Unable to get main device's address alignment" );

    error = clGetDeviceInfo( otherDevice, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof( addressAlign2Bits ), &addressAlign2Bits, NULL );
    test_error( error, "Unable to get secondary device's address alignment" );

    cl_uint addressAlign1 = std::max(addressAlign1Bits, addressAlign2Bits) / 8;
    // Finally time to run!
    return test_sub_buffers_read_write_core( testingContext, queue1, queue2, maxBuffer1, addressAlign1 );
}

cl_int read_buffer_via_kernel( cl_context context, cl_command_queue queue, cl_mem buffer, size_t length, cl_char *outResults )
{
    const char *kernelCode[] = {
        "__kernel void readTest( __global char *inBuffer, __global char *outBuffer )\n"
        "{\n"
        "    int tid = get_global_id(0);\n"
        "    outBuffer[ tid ] = inBuffer[ tid ];\n"
        "}\n" };

    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_int error;

    if ( create_single_kernel_helper( context, &program, &kernel, 1, kernelCode, "readTest" ) )
    {
        return -1;
    }

    size_t threads[1] = { length };

    clMemWrapper outStream = clCreateBuffer( context, CL_MEM_READ_WRITE, length, NULL, &error );
    test_error( error, "Unable to create output stream" );

    error = clSetKernelArg( kernel, 0, sizeof( buffer ), &buffer );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 1, sizeof( outStream ), &outStream );
    test_error( error, "Unable to set kernel argument" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Unable to queue kernel" );

    error = clEnqueueReadBuffer( queue, outStream, CL_TRUE, 0, length, outResults, 0, NULL, NULL );
    test_error( error, "Unable to read results from kernel" );

    return CL_SUCCESS;
}


int test_sub_buffers_overlapping( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_int error;
    size_t mainSize;
    cl_uint addressAlign;

    clMemWrapper mainBuffer;
    SubBufferWrapper subBuffers[ 16 ];


    // Create the main buffer to test against
    error = get_reasonable_buffer_size( deviceID, mainSize );
    test_error( error, "Unable to get reasonable buffer size" );

    mainBuffer = clCreateBuffer( context, CL_MEM_READ_WRITE, mainSize, NULL, &error );
    test_error( error, "Unable to create test main buffer" );

    // Determine the alignment of the device so we can make sure sub buffers are valid
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof( addressAlign ), &addressAlign, NULL );
    test_error( error, "Unable to get device's address alignment" );

    // Create some sub-buffers to use. Note: they don't have to not overlap (we actually *want* them to overlap)
    for ( size_t i = 0; i < 16; i++ )
    {
        size_t offset = get_random_size_t( 0, mainSize / addressAlign, Action::GetRandSeed() ) * addressAlign;
        size_t size = get_random_size_t( 1, ( mainSize - offset ) / addressAlign, Action::GetRandSeed() ) * addressAlign;

        error = subBuffers[ i ].Allocate( mainBuffer, CL_MEM_READ_ONLY, offset, size );
        test_error( error, "Unable to allocate sub buffer" );
    }

    /// For logging, we determine the amount of overlap we just generated
    // Build a fast in-out map to help with generating the stats
    int sbMap[ 32 ], mapSize = 0;
    for ( int i = 0; i < 16; i++ )
    {
        int j;
        for ( j = 0; j < mapSize; j++ )
        {
            size_t pt = ( sbMap[ j ] < 0 ) ? ( subBuffers[ -sbMap[ j ] ].mOrigin + subBuffers[ -sbMap[ j ] ].mSize )
                        : subBuffers[ sbMap[ j ] ].mOrigin;
            if ( subBuffers[ i ].mOrigin < pt )
            {
                // Origin is before this part of the map, so move map forward so we can insert
                memmove( &sbMap[ j + 1 ], &sbMap[ j ], sizeof( int ) * ( mapSize - j ) );
                sbMap[ j ] = i;
                mapSize++;
                break;
            }
        }
        if ( j == mapSize )
        {
            sbMap[ j ] = i;
            mapSize++;
        }

        size_t endPt = subBuffers[ i ].mOrigin + subBuffers[ i ].mSize;
        for ( j = 0; j < mapSize; j++ )
        {
            size_t pt = ( sbMap[ j ] < 0 ) ? ( subBuffers[ -sbMap[ j ] ].mOrigin + subBuffers[ -sbMap[ j ] ].mSize )
                        : subBuffers[ sbMap[ j ] ].mOrigin;
            if ( endPt < pt )
            {
                // Origin is before this part of the map, so move map forward so we can insert
                memmove( &sbMap[ j + 1 ], &sbMap[ j ], sizeof( int ) * ( mapSize - j ) );
                sbMap[ j ] = -( i + 1 );
                mapSize++;
                break;
            }
        }
        if ( j == mapSize )
        {
            sbMap[ j ] = -( i + 1 );
            mapSize++;
        }
    }
    long long delta = 0;
    size_t maxOverlap = 1, overlap = 0;
    for ( int i = 0; i < 32; i++ )
    {
        if ( sbMap[ i ] >= 0 )
        {
            overlap++;
            if ( overlap > 1 )
                delta -= (long long)( subBuffers[ sbMap[ i ] ].mOrigin );
            if ( overlap > maxOverlap )
                maxOverlap = overlap;
        }
        else
        {
            if ( overlap > 1 )
                delta += (long long)( subBuffers[ -sbMap[ i ] - 1 ].mOrigin + subBuffers[ -sbMap[ i ] - 1 ].mSize );
            overlap--;
        }
    }

    log_info( "\tTesting %d sub-buffers with %lld overlapping Kbytes (%d%%; as many as %ld buffers overlapping at once)\n",
              16, ( delta / 1024LL ), (int)( delta * 100LL / (long long)mainSize ), maxOverlap );

    // Write some random contents to the main buffer
    cl_char * contents = new cl_char[ mainSize ];
    generate_random_data( kChar, mainSize, Action::GetRandSeed(), contents );

    error = clEnqueueWriteBuffer( queue, mainBuffer, CL_TRUE, 0, mainSize, contents, 0, NULL, NULL );
    test_error( error, "Unable to write to main buffer" );

    // Now read from each sub-buffer and check to make sure that they make sense w.r.t. the main contents
    cl_char * tempBuffer = new cl_char[ mainSize ];

    int numErrors = 0;
    for ( size_t i = 0; i < 16; i++ )
    {
        // Read from this buffer
        int which = random_in_range( 0, 1, Action::GetRandSeed() );
        if ( which )
            error = clEnqueueReadBuffer( queue, subBuffers[ i ], CL_TRUE, 0, subBuffers[ i ].mSize, tempBuffer, 0, NULL, NULL );
        else
            error = read_buffer_via_kernel( context, queue, subBuffers[ i ], subBuffers[ i ].mSize, tempBuffer );
        test_error( error, "Unable to read sub buffer contents" );

        if ( memcmp( tempBuffer, contents + subBuffers[ i ].mOrigin, subBuffers[ i ].mSize ) != 0 )
        {
            log_error( "ERROR: Validation for sub-buffer %ld failed!\n", i );
            numErrors++;
        }
    }

    delete [] contents;
    delete [] tempBuffer;
    Action::FreeRandSeed();

    return numErrors;
}

