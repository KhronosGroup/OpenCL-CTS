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

#define CL_EXIT_ERROR(cmd,format,...)                \
{                                \
if ((cmd) != CL_SUCCESS) {                    \
log_error("CL ERROR: %s %u: ", __FILE__,__LINE__);    \
log_error(format,## __VA_ARGS__ );            \
log_error("\n");                        \
/*abort();*/                \
}                                \
}

typedef unsigned char BufferType;

// Globals for test
cl_command_queue queue;

// Width and height of each pair of images.
enum { TotalImages = 8 };
size_t width  [TotalImages];
size_t height [TotalImages];
size_t depth  [TotalImages];

// cl buffer and host buffer.
cl_mem buffer [TotalImages];
BufferType* verify[TotalImages];
BufferType* backing[TotalImages];

// Temporary buffer used for read and write operations.
BufferType* tmp_buffer;
size_t tmp_buffer_size;

size_t num_tries   = 50; // Number of randomly selected operations to perform.
size_t alloc_scale = 2;   // Scale term applied buffer allocation size.
MTdata mt;

// Initialize a buffer in host memory containing random values of the specified size.
static void initialize_image(BufferType* ptr, size_t w, size_t h, size_t d, MTdata mt)
{
    enum { ElementSize = sizeof(BufferType)/sizeof(unsigned char) };

    unsigned char* buf = (unsigned char*)ptr;
    size_t size = w*h*d*ElementSize;

    for (size_t i = 0; i != size; i++) {
        buf[i] = (unsigned char)(genrand_int32(mt) % 0xff);
    }
}

// This function prints the contents of a buffer to standard error.
void print_buffer(BufferType* buf, size_t w, size_t h, size_t d) {
    log_error("Size = %lux%lux%lu (%lu total)\n",w,h,d,w*h*d);
    for (unsigned k=0; k!=d;++k) {
        log_error("Slice: %u\n",k);
        for (unsigned j=0; j!=h;++j) {
            for (unsigned i=0;i!=w;++i) {
                log_error("%02x",buf[k*(w*h)+j*w+i]);
            }
            log_error("\n");
        }
        log_error("\n");
    }
}

// Returns true if the two specified regions overlap.
bool check_overlap_rect(size_t src_offset[3],
                        size_t dst_offset[3],
                        size_t region[3],
                        size_t row_pitch,
                        size_t slice_pitch)
{
    const size_t src_min[] = { src_offset[0], src_offset[1], src_offset[2] };
    const size_t src_max[] = { src_offset[0] + region[0], src_offset[1] + region[1], src_offset[2] + region[2] };

    const size_t dst_min[] = { dst_offset[0], dst_offset[1], dst_offset[2] };
    const size_t dst_max[] = { dst_offset[0] + region[0],
                               dst_offset[1] + region[1],
                               dst_offset[2] + region[2]};
// Check for overlap
        bool overlap = true;
        unsigned i;
        for (i = 0; i != 3; ++i)
        {
            overlap = overlap && (src_min[i] < dst_max[i]) && (src_max[i] > dst_min[i]);
        }

    size_t dst_start = dst_offset[2] * slice_pitch + dst_offset[1] * row_pitch + dst_offset[0];
    size_t dst_end = dst_start + (region[2] * slice_pitch +
                                  region[1] * row_pitch + region[0]);
    size_t src_start = src_offset[2] * slice_pitch + src_offset[1] * row_pitch + src_offset[0];
    size_t src_end = src_start + (region[2] * slice_pitch +
                                  region[1] * row_pitch + region[0]);
    if (!overlap) {
        size_t delta_src_x = (src_offset[0] + region[0] > row_pitch) ?
            src_offset[0] + region[0] - row_pitch : 0; size_t delta_dst_x = (dst_offset[0] + region[0] > row_pitch) ?
            dst_offset[0] + region[0] - row_pitch : 0;
        if ((delta_src_x > 0 && delta_src_x > dst_offset[0]) ||
            (delta_dst_x > 0 && delta_dst_x > src_offset[0])) {
            if ((src_start <= dst_start && dst_start < src_end) || (dst_start <= src_start && src_start < dst_end)) overlap = true;
        }
        if (region[2] > 1) {
            size_t src_height = slice_pitch / row_pitch; size_t dst_height = slice_pitch / row_pitch;
            size_t delta_src_y = (src_offset[1] + region[1] > src_height) ? src_offset[1] + region[1] - src_height : 0;
            size_t delta_dst_y = (dst_offset[1] + region[1] > dst_height) ? dst_offset[1] + region[1] - dst_height : 0;
            if ((delta_src_y > 0 && delta_src_y > dst_offset[1]) ||
                (delta_dst_y > 0 && delta_dst_y > src_offset[1])) {
                if ((src_start <= dst_start && dst_start < src_end) || (dst_start <= src_start && src_start < dst_end))
                    overlap = true;
            }
        }
    }
    return overlap;
}



// This function invokes the CopyBufferRect CL command and then mirrors the operation on the host side verify buffers.
int copy_region(size_t src, size_t soffset[3], size_t sregion[3], size_t dst, size_t doffset[3], size_t dregion[3]) {

    // Copy between cl buffers.
    size_t src_slice_pitch = (width[src]*height[src] != 1) ? width[src]*height[src] : 0;
    size_t dst_slice_pitch = (width[dst]*height[dst] != 1) ? width[dst]*height[dst] : 0;
    size_t src_row_pitch = width[src];

    cl_int err;
    if (check_overlap_rect(soffset,doffset,sregion,src_row_pitch, src_slice_pitch)) {
        log_info( "Copy overlap reported, skipping copy buffer rect\n" );
        return CL_SUCCESS;
    } else {
        if ((err = clEnqueueCopyBufferRect(queue,
                                         buffer[src],buffer[dst],
                                         soffset, doffset,
                                         sregion,/*dregion,*/
                                         width[src], src_slice_pitch,
                                         width[dst], dst_slice_pitch,
                                         0, NULL, NULL)) != CL_SUCCESS)
        {
            CL_EXIT_ERROR(err, "clEnqueueCopyBufferRect failed between %u and %u",(unsigned)src,(unsigned)dst);
        }
    }

    // Copy between host buffers.
    size_t total = sregion[0] * sregion[1] * sregion[2];

    size_t spitch = width[src];
    size_t sslice = width[src]*height[src];

    size_t dpitch = width[dst];
    size_t dslice = width[dst]*height[dst];

    for (size_t i = 0; i != total; ++i) {

        // Compute the coordinates of the element within the source and destination regions.
        size_t rslice = sregion[0]*sregion[1];
        size_t sz = i / rslice;
        size_t sy = (i % rslice) / sregion[0];
        size_t sx = (i % rslice) % sregion[0];

        size_t dz = sz;
        size_t dy = sy;
        size_t dx = sx;

        // Compute the offset in bytes of the source and destination.
        size_t s_idx = (soffset[2]+sz)*sslice + (soffset[1]+sy)*spitch + soffset[0]+sx;
        size_t d_idx = (doffset[2]+dz)*dslice + (doffset[1]+dy)*dpitch + doffset[0]+dx;

        verify[dst][d_idx] = verify[src][s_idx];
    }

    return 0;
}

// This function compares the destination region in the buffer pointed
// to by device, to the source region of the specified verify buffer.
int verify_region(BufferType* device, size_t src, size_t soffset[3], size_t sregion[3], size_t dst, size_t doffset[3]) {

    // Copy between host buffers.
    size_t spitch = width[src];
    size_t sslice = width[src]*height[src];

    size_t dpitch = width[dst];
    size_t dslice = width[dst]*height[dst];

    size_t total = sregion[0] * sregion[1] * sregion[2];
    for (size_t i = 0; i != total; ++i) {

        // Compute the coordinates of the element within the source and destination regions.
        size_t rslice = sregion[0]*sregion[1];
        size_t sz = i / rslice;
        size_t sy = (i % rslice) / sregion[0];
        size_t sx = (i % rslice) % sregion[0];

        // Compute the offset in bytes of the source and destination.
        size_t s_idx = (soffset[2]+sz)*sslice + (soffset[1]+sy)*spitch + soffset[0]+sx;
        size_t d_idx = (doffset[2]+sz)*dslice + (doffset[1]+sy)*dpitch + doffset[0]+sx;

        if (device[d_idx] != verify[src][s_idx]) {
            log_error("Verify failed on comparsion %lu: coordinate (%lu, %lu, %lu) of region\n",i,sx,sy,sz);
            log_error("0x%02x != 0x%02x\n", device[d_idx], verify[src][s_idx]);
#if 0
            // Uncomment this section to print buffers.
            log_error("Device (copy): [%lu]\n",dst);
            print_buffer(device,width[dst],height[dst],depth[dst]);
            log_error("\n");
            log_error("Verify: [%lu]\n",src);
            print_buffer(verify[src],width[src],height[src],depth[src]);
            log_error("\n");
            abort();
#endif
            return -1;
        }
    }

    return 0;
}


// This function invokes ReadBufferRect to read a region from the
// specified source buffer into a temporary destination buffer. The
// contents of the temporary buffer are then compared to the source
// region of the corresponding verify buffer.
int read_verify_region(size_t src, size_t soffset[3], size_t sregion[3], size_t dst, size_t doffset[3], size_t dregion[3]) {

    // Clear the temporary destination host buffer.
    memset(tmp_buffer, 0xff, tmp_buffer_size);

    size_t src_slice_pitch = (width[src]*height[src] != 1) ? width[src]*height[src] : 0;
    size_t dst_slice_pitch = (width[dst]*height[dst] != 1) ? width[dst]*height[dst] : 0;

    // Copy the source region of the cl buffer, to the destination region of the temporary buffer.
    CL_EXIT_ERROR(clEnqueueReadBufferRect(queue,
                                          buffer[src],
                                          CL_TRUE,
                                          soffset,doffset,
                                          sregion,
                                          width[src], src_slice_pitch,
                                          width[dst], dst_slice_pitch,
                                          tmp_buffer,
                                          0, NULL, NULL), "clEnqueueCopyBufferRect failed between %u and %u",(unsigned)src,(unsigned)dst);

    return verify_region(tmp_buffer,src,soffset,sregion,dst,doffset);
}

// This function performs the same verification check as
// read_verify_region, except a MapBuffer command is used to access the
// device buffer data instead of a ReadBufferRect, and the whole
// buffer is checked.
int map_verify_region(size_t src) {

    size_t size_bytes = width[src]*height[src]*depth[src]*sizeof(BufferType);

    // Copy the source region of the cl buffer, to the destination region of the temporary buffer.
    cl_int err;
    BufferType* mapped = (BufferType*)clEnqueueMapBuffer(queue,buffer[src],CL_TRUE,CL_MAP_READ,0,size_bytes,0,NULL,NULL,&err);
    CL_EXIT_ERROR(err, "clEnqueueMapBuffer failed for buffer %u",(unsigned)src);

    size_t soffset[] = { 0, 0, 0 };
    size_t sregion[] = { width[src], height[src], depth[src] };

    int ret = verify_region(mapped,src,soffset,sregion,src,soffset);

    CL_EXIT_ERROR(clEnqueueUnmapMemObject(queue,buffer[src],mapped,0,NULL,NULL),
                  "clEnqueueUnmapMemObject failed for buffer %u",(unsigned)src);

    return ret;
}

// This function generates a new temporary buffer and then writes a
// region of it to a region in the specified destination buffer.
int write_region(size_t src, size_t soffset[3], size_t sregion[3], size_t dst, size_t doffset[3], size_t dregion[3]) {

    initialize_image(tmp_buffer, tmp_buffer_size, 1, 1, mt);
    // memset(tmp_buffer, 0xf0, tmp_buffer_size);

    size_t src_slice_pitch = (width[src]*height[src] != 1) ? width[src]*height[src] : 0;
    size_t dst_slice_pitch = (width[dst]*height[dst] != 1) ? width[dst]*height[dst] : 0;

    // Copy the source region of the cl buffer, to the destination region of the temporary buffer.
    CL_EXIT_ERROR(clEnqueueWriteBufferRect(queue,
                                           buffer[dst],
                                           CL_TRUE,
                                           doffset,soffset,
    /*sregion,*/dregion,
                                           width[dst], dst_slice_pitch,
                                           width[src], src_slice_pitch,
                                           tmp_buffer,
                                           0, NULL, NULL), "clEnqueueWriteBufferRect failed between %u and %u",(unsigned)src,(unsigned)dst);

    // Copy from the temporary buffer to the host buffer.
    size_t spitch = width[src];
    size_t sslice = width[src]*height[src];
    size_t dpitch = width[dst];
    size_t dslice = width[dst]*height[dst];

    size_t total = sregion[0] * sregion[1] * sregion[2];
    for (size_t i = 0; i != total; ++i) {

        // Compute the coordinates of the element within the source and destination regions.
        size_t rslice = sregion[0]*sregion[1];
        size_t sz = i / rslice;
        size_t sy = (i % rslice) / sregion[0];
        size_t sx = (i % rslice) % sregion[0];

        size_t dz = sz;
        size_t dy = sy;
        size_t dx = sx;

        // Compute the offset in bytes of the source and destination.
        size_t s_idx = (soffset[2]+sz)*sslice + (soffset[1]+sy)*spitch + soffset[0]+sx;
        size_t d_idx = (doffset[2]+dz)*dslice + (doffset[1]+dy)*dpitch + doffset[0]+dx;

        verify[dst][d_idx] = tmp_buffer[s_idx];
    }
    return 0;
}

void CL_CALLBACK mem_obj_destructor_callback( cl_mem, void *data )
{
    free( data );
}

// This is the main test function for the conformance test.
int
test_bufferreadwriterect(cl_device_id device, cl_context context, cl_command_queue queue_, int num_elements)
{
    queue = queue_;
    cl_int err;

    // Initialize the random number generator.
    mt = init_genrand( gRandomSeed );

    // Compute a maximum buffer size based on the number of test images and the device maximum.
    cl_ulong max_mem_alloc_size = 0;
    CL_EXIT_ERROR(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL),"Could not get device info");
    log_info("CL_DEVICE_MAX_MEM_ALLOC_SIZE = %llu bytes.\n", max_mem_alloc_size);

    // Confirm that the maximum allocation size is not zero.
    if (max_mem_alloc_size == 0) {
        log_error("Error: CL_DEVICE_MAX_MEM_ALLOC_SIZE is zero bytes\n");
        return -1;
    }

    // Guess at a reasonable maximum dimension.
    size_t max_mem_alloc_dim = (size_t)cbrt((double)(max_mem_alloc_size/sizeof(BufferType)))/alloc_scale;
    if (max_mem_alloc_dim == 0) {
        max_mem_alloc_dim = max_mem_alloc_size;
    }

    log_info("Using maximum dimension      = %lu.\n", max_mem_alloc_dim);

    // Create pairs of cl buffers and host buffers on which operations will be mirrored.
    log_info("Creating %u pairs of random sized host and cl buffers.\n", TotalImages);

    size_t max_size = 0;
    size_t total_bytes = 0;

    for (unsigned i=0; i != TotalImages; ++i) {

        // Determine a width and height for this buffer.
        size_t size_bytes;
        size_t tries = 0;
        size_t max_tries = 1048576;
        do {
            width[i]   = get_random_size_t(1, max_mem_alloc_dim, mt);
            height[i]  = get_random_size_t(1, max_mem_alloc_dim, mt);
            depth[i]   = get_random_size_t(1, max_mem_alloc_dim, mt);
            ++tries;
        } while ((tries < max_tries) && (size_bytes = width[i]*height[i]*depth[i]*sizeof(BufferType)) > max_mem_alloc_size);

        // Check to see if adequately sized buffers were found.
        if (tries >= max_tries) {
            log_error("Error: Could not find random buffer sized less than %llu bytes in %lu tries.\n",
                      max_mem_alloc_size, max_tries);
            return -1;
        }

        // Keep track of the dimensions of the largest buffer.
        max_size = (size_bytes > max_size) ? size_bytes : max_size;
        total_bytes += size_bytes;

        log_info("Buffer[%u] is (%lu,%lu,%lu) = %lu MB (truncated)\n",i,width[i],height[i],depth[i],(size_bytes)/1048576);
    }

    log_info( "Total size: %lu MB (truncated)\n", total_bytes/1048576 );

    // Allocate a temporary buffer for read and write operations.
    tmp_buffer_size  = max_size;
    tmp_buffer = (BufferType*)malloc(tmp_buffer_size);

    // Initialize cl buffers
    log_info( "Initializing buffers\n" );
    for (unsigned i=0; i != TotalImages; ++i) {

        size_t size_bytes = width[i]*height[i]*depth[i]*sizeof(BufferType);

        // Allocate a host copy of the buffer for verification.
        verify[i] = (BufferType*)malloc(size_bytes);
        CL_EXIT_ERROR(verify[i] ? CL_SUCCESS : -1, "malloc of host buffer failed for buffer %u", i);

        // Allocate the buffer in host memory.
        backing[i] = (BufferType*)malloc(size_bytes);
        CL_EXIT_ERROR(backing[i] ? CL_SUCCESS : -1, "malloc of backing buffer failed for buffer %u", i);

        // Generate a random buffer.
        log_info( "Initializing buffer %u\n", i );
        initialize_image(verify[i], width[i], height[i], depth[i], mt);

        // Copy the image into a buffer which will passed to CL.
        memcpy(backing[i], verify[i], size_bytes);

        // Create the CL buffer.
        buffer[i] = clCreateBuffer (context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, size_bytes, backing[i], &err);
        CL_EXIT_ERROR(err,"clCreateBuffer failed for buffer %u", i);

        // Make sure buffer is cleaned up appropriately if we encounter an error in the rest of the calls.
        err = clSetMemObjectDestructorCallback( buffer[i], mem_obj_destructor_callback, backing[i] );
        CL_EXIT_ERROR(err, "Unable to set mem object destructor callback" );
    }

    // Main test loop, run num_tries times.
    log_info( "Executing %u test operations selected at random.\n", (unsigned)num_tries );
    for (size_t iter = 0; iter < num_tries; ++iter) {

        // Determine a source and a destination.
        size_t src = get_random_size_t(0,TotalImages,mt);
        size_t dst = get_random_size_t(0,TotalImages,mt);

        // Determine the minimum dimensions.
        size_t min_width = width[src] < width[dst] ? width[src] : width[dst];
        size_t min_height = height[src] < height[dst] ? height[src] : height[dst];
        size_t min_depth = depth[src] < depth[dst] ? depth[src] : depth[dst];

        // Generate a random source rectangle within the minimum dimensions.
        size_t mx = get_random_size_t(0, min_width-1, mt);
        size_t my = get_random_size_t(0, min_height-1, mt);
        size_t mz = get_random_size_t(0, min_depth-1, mt);

        size_t sw = get_random_size_t(1, (min_width - mx), mt);
        size_t sh = get_random_size_t(1, (min_height - my), mt);
        size_t sd = get_random_size_t(1, (min_depth - mz), mt);

        size_t sx = get_random_size_t(0, width[src]-sw, mt);
        size_t sy = get_random_size_t(0, height[src]-sh, mt);
        size_t sz = get_random_size_t(0, depth[src]-sd, mt);

        size_t soffset[] = { sx, sy, sz };
        size_t sregion[] = { sw, sh, sd };

        // Generate a destination rectangle of the same size.
        size_t dw = sw;
        size_t dh = sh;
        size_t dd = sd;

        // Generate a random destination offset within the buffer.
        size_t dx = get_random_size_t(0, (width[dst] - dw), mt);
        size_t dy = get_random_size_t(0, (height[dst] - dh), mt);
        size_t dz = get_random_size_t(0, (depth[dst] - dd), mt);
        size_t doffset[] = { dx, dy, dz };
        size_t dregion[] = { dw, dh, dd };

        // Execute one of three operations:
        // - Copy: Copies between src and dst within each set of host, buffer, and images.
        // - Read & verify: Reads src region from buffer and image, and compares to host.
        // - Write: Generates new buffer with src dimensions, and writes to cl buffer and image.

        enum { TotalOperations = 3 };
        size_t operation = get_random_size_t(0,TotalOperations,mt);

        switch (operation) {
            case 0:
                log_info("%lu Copy %lu offset (%lu,%lu,%lu) -> %lu offset (%lu,%lu,%lu) region (%lux%lux%lu = %lu)\n",
                         iter,
                         src, soffset[0], soffset[1], soffset[2],
                         dst, doffset[0], doffset[1], doffset[2],
                         sregion[0], sregion[1], sregion[2],
                         sregion[0]*sregion[1]*sregion[2]);
                if ((err = copy_region(src, soffset, sregion, dst, doffset, dregion)))
                    return err;
                break;
            case 1:
                log_info("%lu Read %lu offset (%lu,%lu,%lu) -> %lu offset (%lu,%lu,%lu) region (%lux%lux%lu = %lu)\n",
                         iter,
                         src, soffset[0], soffset[1], soffset[2],
                         dst, doffset[0], doffset[1], doffset[2],
                         sregion[0], sregion[1], sregion[2],
                         sregion[0]*sregion[1]*sregion[2]);
                if ((err = read_verify_region(src, soffset, sregion, dst, doffset, dregion)))
                    return err;
                break;
            case 2:
                log_info("%lu Write %lu offset (%lu,%lu,%lu) -> %lu offset (%lu,%lu,%lu) region (%lux%lux%lu = %lu)\n",
                         iter,
                         src, soffset[0], soffset[1], soffset[2],
                         dst, doffset[0], doffset[1], doffset[2],
                         sregion[0], sregion[1], sregion[2],
                         sregion[0]*sregion[1]*sregion[2]);
                if ((err = write_region(src, soffset, sregion, dst, doffset, dregion)))
                    return err;
                break;
        }

#if 0
        // Uncomment this section to verify each operation.
        // If commented out, verification won't occur until the end of the
        // test, and it will not be possible to determine which operation failed.
        log_info("Verify src %lu offset (%u,%u,%u) region (%lux%lux%lu)\n", src, 0, 0, 0, width[src], height[src], depth[src]);
        if (err = map_verify_region(src))
            return err;

        log_info("Verify dst %lu offset (%u,%u,%u) region (%lux%lux%lu)\n", dst, 0, 0, 0, width[dst], height[dst], depth[dst]);
        if (err = map_verify_region(dst))
            return err;


#endif

    } // end main for loop.

    for (unsigned i=0;i<TotalImages;++i) {
        log_info("Verify %u offset (%u,%u,%u) region (%lux%lux%lu)\n", i, 0, 0, 0, width[i], height[i], depth[i]);
        if ((err = map_verify_region(i)))
            return err;
    }

    // Clean-up.
    free_mtdata(mt);
    for (unsigned i=0;i<TotalImages;++i) {
        free( verify[i] );
        clReleaseMemObject( buffer[i] );
    }
    free( tmp_buffer );

    if (!err) {
        log_info("RECT read, write test passed\n");
    }

    return err;
}



