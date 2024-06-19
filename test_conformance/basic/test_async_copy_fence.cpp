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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../../test_common/harness/conversions.h"
#include "procs.h"

static const char *import_after_export_aliased_local_kernel =
    "#pragma OPENCL EXTENSION cl_khr_async_work_group_copy_fence : enable\n"
    "%s\n" // optional pragma string
    "__kernel void test_fn( const __global %s *exportSrc, __global %s "
    "*exportDst,\n"
    "                       const __global %s *importSrc, __global %s "
    "*importDst,\n"
    "                       __local %s *localBuffer, /* there isn't another "
    "__local %s local buffer since export src and import dst are aliased*/\n"
    "                       int exportSrcLocalSize, int "
    "exportCopiesPerWorkItem,\n"
    "                       int importSrcLocalSize, int "
    "importCopiesPerWorkItem )\n"
    "{\n"
    "    int i;\n"
    "    int localImportOffset = exportSrcLocalSize - importSrcLocalSize;\n"
    // Zero the local storage first
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        localBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] = "
    "(%s)(%s)0;\n"
    "    }\n"
    "    // no need to set another local buffer values to (%s)(%s)0 since "
    "export src and import dst are aliased (use the same buffer)\n"
    // Do this to verify all kernels are done zeroing the local buffer before we
    // try the export and import
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        localBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] = "
    "exportSrc[ get_global_id( 0 )*exportCopiesPerWorkItem+i ];\n"
    "    }\n"
    // Do this to verify all kernels are done copying to the local buffer before
    // we try the export and import
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    event_t events;\n"
    "    events = async_work_group_copy((__global "
    "%s*)(exportDst+exportSrcLocalSize*get_group_id(0)), (__local const "
    "%s*)localBuffer, (size_t)exportSrcLocalSize, 0 );\n"
    "    async_work_group_copy_fence( CLK_LOCAL_MEM_FENCE );\n"
    "    events = async_work_group_copy( (__local "
    "%s*)(localBuffer+localImportOffset), (__global const "
    "%s*)(importSrc+importSrcLocalSize*get_group_id(0)), "
    "(size_t)importSrcLocalSize, events );\n"
    // Wait for the export and import to complete, then verify by manually
    // copying to the dest
    "    wait_group_events( 2, &events );\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importDst[ get_global_id( 0 )*importCopiesPerWorkItem+i ] = "
    "(localBuffer+localImportOffset)[ get_local_id( 0 "
    ")*importCopiesPerWorkItem+i ];\n"
    "    }\n"
    "}\n";

static const char *import_after_export_aliased_global_kernel =
    "#pragma OPENCL EXTENSION cl_khr_async_work_group_copy_fence : enable\n"
    "%s\n" // optional pragma string
    "__kernel void test_fn( const __global %s *exportSrc, __global %s "
    "*exportDstImportSrc,\n"
    "                       __global %s *importDst, /* there isn't a dedicated "
    "__global %s buffer for import src since export dst and import src are "
    "aliased*/\n"
    "                       __local %s *exportLocalBuffer, __local %s "
    "*importLocalBuffer,\n"
    "                       int exportSrcLocalSize, int "
    "exportCopiesPerWorkItem,\n"
    "                       int importSrcLocalSize, int "
    "importCopiesPerWorkItem )\n"
    "{\n"
    "    int i;\n"
    // Zero the local storage first
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        exportLocalBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] "
    "= (%s)(%s)0;\n"
    "    }\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importLocalBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ] "
    "= (%s)(%s)0;\n"
    "    }\n"
    // Do this to verify all kernels are done zeroing the local buffer before we
    // try the export and import
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        exportLocalBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] "
    "= exportSrc[ get_global_id( 0 )*exportCopiesPerWorkItem+i ];\n"
    "    }\n"
    // Do this to verify all kernels are done copying to the local buffer before
    // we try the export and import
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    event_t events;\n"
    "    events = async_work_group_copy((__global "
    "%s*)(exportDstImportSrc+exportSrcLocalSize*get_group_id(0)), (__local "
    "const %s*)exportLocalBuffer, (size_t)exportSrcLocalSize, 0 );\n"
    "    async_work_group_copy_fence( CLK_GLOBAL_MEM_FENCE );\n"
    "    events = async_work_group_copy( (__local %s*)importLocalBuffer, "
    "(__global const "
    "%s*)(exportDstImportSrc+exportSrcLocalSize*get_group_id(0) + "
    "(exportSrcLocalSize - importSrcLocalSize)), (size_t)importSrcLocalSize, "
    "events );\n"
    // Wait for the export and import to complete, then verify by manually
    // copying to the dest
    "    wait_group_events( 2, &events );\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importDst[ get_global_id( 0 )*importCopiesPerWorkItem+i ] = "
    "importLocalBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ];\n"
    "    }\n"
    "}\n";

static const char *import_after_export_aliased_global_and_local_kernel =
    "#pragma OPENCL EXTENSION cl_khr_async_work_group_copy_fence : enable\n"
    "%s\n" // optional pragma string
    "__kernel void test_fn( const __global %s *exportSrc, __global %s "
    "*exportDstImportSrc,\n"
    "                       __global %s *importDst, /* there isn't a dedicated "
    "__global %s buffer for import src since export dst and import src are "
    "aliased*/\n"
    "                       __local %s *localBuffer, /* there isn't another "
    "__local %s local buffer since export src and import dst are aliased*/\n"
    "                       int exportSrcLocalSize, int "
    "exportCopiesPerWorkItem,\n"
    "                       int importSrcLocalSize, int "
    "importCopiesPerWorkItem )\n"
    "{\n"
    "    int i;\n"
    "    int localImportOffset = exportSrcLocalSize - importSrcLocalSize;\n"
    // Zero the local storage first
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        localBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] = "
    "(%s)(%s)0;\n"
    "    }\n"
    "    // no need to set another local buffer values to (%s)(%s)0 since "
    "export src and import dst are aliased (use the same buffer)\n"
    // Do this to verify all kernels are done zeroing the local buffer before we
    // try the export and import
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        localBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] = "
    "exportSrc[ get_global_id( 0 )*exportCopiesPerWorkItem+i ];\n"
    "    }\n"
    // Do this to verify all kernels are done copying to the local buffer before
    // we try the export and import
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    event_t events;\n"
    "    events = async_work_group_copy((__global "
    "%s*)(exportDstImportSrc+exportSrcLocalSize*get_group_id(0)), (__local "
    "const %s*)localBuffer, (size_t)exportSrcLocalSize, 0 );\n"
    "    async_work_group_copy_fence( CLK_GLOBAL_MEM_FENCE | "
    "CLK_LOCAL_MEM_FENCE );\n"
    "    events = async_work_group_copy( (__local "
    "%s*)(localBuffer+localImportOffset), (__global const "
    "%s*)(exportDstImportSrc+exportSrcLocalSize*get_group_id(0) + "
    "(exportSrcLocalSize - importSrcLocalSize)), (size_t)importSrcLocalSize, "
    "events );\n"
    // Wait for the export and import to complete, then verify by manually
    // copying to the dest
    "    wait_group_events( 2, &events );\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importDst[ get_global_id( 0 )*importCopiesPerWorkItem+i ] = "
    "(localBuffer+localImportOffset)[ get_local_id( 0 "
    ")*importCopiesPerWorkItem+i ];\n"
    "    }\n"
    "}\n";

static const char *export_after_import_aliased_local_kernel =
    "#pragma OPENCL EXTENSION cl_khr_async_work_group_copy_fence : enable\n"
    "%s\n" // optional pragma string
    "__kernel void test_fn( const __global %s *importSrc, __global %s "
    "*importDst,\n"
    "                       const __global %s *exportDst, /* there isn't a "
    "dedicated __global %s buffer for export src since the local memory is "
    "aliased, so the export src is taken from it */\n"
    "                       __local %s *localBuffer, /* there isn't another "
    "__local %s local buffer since import dst and export src are aliased*/\n"
    "                       int importSrcLocalSize, int "
    "importCopiesPerWorkItem,\n"
    "                       int exportSrcLocalSize, int "
    "exportCopiesPerWorkItem )\n"
    "{\n"
    "    int i;\n"
    // Zero the local storage first
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        localBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ] = "
    "(%s)(%s)0;\n"
    "    }\n"
    "    // no need to set another local buffer values to (%s)(%s)0 since "
    "import dst and export src are aliased (use the same buffer)\n"
    // Do this to verify all kernels are done zeroing the local buffer before we
    // try the import and export
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    event_t events;\n"
    "    events = async_work_group_copy( (__local %s*)localBuffer, (__global "
    "const %s*)(importSrc+importSrcLocalSize*get_group_id(0)), "
    "(size_t)importSrcLocalSize, 0 );\n"
    "    async_work_group_copy_fence( CLK_LOCAL_MEM_FENCE );\n"
    "    events = async_work_group_copy((__global "
    "%s*)(exportDst+exportSrcLocalSize*get_group_id(0)), (__local const "
    "%s*)(localBuffer + (importSrcLocalSize - exportSrcLocalSize)), "
    "(size_t)exportSrcLocalSize, events );\n"
    // Wait for the import and export to complete, then verify by manually
    // copying to the dest
    "    wait_group_events( 1, &events );\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importDst[ get_global_id( 0 )*importCopiesPerWorkItem+i ] = "
    "localBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ];\n"
    "    }\n"
    "}\n";

static const char *export_after_import_aliased_global_kernel =
    "#pragma OPENCL EXTENSION cl_khr_async_work_group_copy_fence : enable\n"
    "%s\n" // optional pragma string
    "__kernel void test_fn( const __global %s *importSrcExportDst, __global %s "
    "*importDst,\n"
    "                       const __global %s *exportSrc,\n"
    "                       /* there isn't a dedicated __global %s buffer for "
    "export dst since import src and export dst are aliased */\n"
    "                       __local %s *importLocalBuffer, __local %s "
    "*exportLocalBuffer,\n"
    "                       int importSrcLocalSize, int "
    "importCopiesPerWorkItem,\n"
    "                       int exportSrcLocalSize, int "
    "exportCopiesPerWorkItem )\n"
    "{\n"
    "    int i;\n"
    // Zero the local storage first
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importLocalBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ] "
    "= (%s)(%s)0;\n"
    "    }\n"
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        exportLocalBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] "
    "= (%s)(%s)0;\n"
    "    }\n"
    // Do this to verify all kernels are done zeroing the local buffer before we
    // try the import and export
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    for(i=0; i<exportCopiesPerWorkItem; i++) {\n"
    "        exportLocalBuffer[ get_local_id( 0 )*exportCopiesPerWorkItem+i ] "
    "= exportSrc[ get_global_id( 0 )*exportCopiesPerWorkItem+i ];\n"
    "    }\n"
    // Do this to verify all kernels are done copying to the local buffer before
    // we try the import and export
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    event_t events;\n"
    "    events = async_work_group_copy( (__local %s*)importLocalBuffer, "
    "(__global const "
    "%s*)(importSrcExportDst+importSrcLocalSize*get_group_id(0)), "
    "(size_t)importSrcLocalSize, 0 );\n"
    "    async_work_group_copy_fence( CLK_GLOBAL_MEM_FENCE );\n"
    "    events = async_work_group_copy((__global "
    "%s*)(importSrcExportDst+importSrcLocalSize*get_group_id(0) + "
    "(importSrcLocalSize - exportSrcLocalSize)), (__local const "
    "%s*)exportLocalBuffer, (size_t)exportSrcLocalSize, events );\n"
    // Wait for the import and export to complete, then verify by manually
    // copying to the dest
    "    wait_group_events( 2, &events );\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importDst[ get_global_id( 0 )*importCopiesPerWorkItem+i ] = "
    "importLocalBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ];\n"
    "    }\n"
    "}\n";

static const char *export_after_import_aliased_global_and_local_kernel =
    "#pragma OPENCL EXTENSION cl_khr_async_work_group_copy_fence : enable\n"
    "%s\n" // optional pragma string
    "__kernel void test_fn( const __global %s *importSrcExportDst, __global %s "
    "*importDst,\n"
    "                       /* there isn't a dedicated __global %s buffer for "
    "export src since the local memory is aliased, so the export src is taken "
    "from it */\n"
    "                       /* there isn't a dedicated __global %s buffer for "
    "export dst since import src and export dst are aliased */\n"
    "                       __local %s *localBuffer, /* there isn't another "
    "__local %s local buffer since import dst and export src are aliased*/\n"
    "                       int importSrcLocalSize, int "
    "importCopiesPerWorkItem,\n"
    "                       int exportSrcLocalSize, int "
    "exportCopiesPerWorkItem )\n"
    "{\n"
    "    int i;\n"
    // Zero the local storage first
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        localBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ] = "
    "(%s)(%s)0;\n"
    "    }\n"
    "    // no need to set another local buffer values to (%s)(%s)0 since "
    "import dst and export src are aliased (use the same buffer)\n"
    // Do this to verify all kernels are done zeroing the local buffer before we
    // try the import and export
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    event_t events;\n"
    "    events = async_work_group_copy( (__local %s*)localBuffer, (__global "
    "const %s*)(importSrcExportDst+importSrcLocalSize*get_group_id(0)), "
    "(size_t)importSrcLocalSize, 0 );\n"
    "    async_work_group_copy_fence( CLK_GLOBAL_MEM_FENCE | "
    "CLK_LOCAL_MEM_FENCE );\n"
    "    events = async_work_group_copy((__global "
    "%s*)(importSrcExportDst+importSrcLocalSize*get_group_id(0) + "
    "(importSrcLocalSize - exportSrcLocalSize)), (__local const "
    "%s*)(localBuffer + (importSrcLocalSize - exportSrcLocalSize)), "
    "(size_t)exportSrcLocalSize, events );\n"
    // Wait for the import and export to complete, then verify by manually
    // copying to the dest
    "    wait_group_events( 2, &events );\n"
    "    for(i=0; i<importCopiesPerWorkItem; i++) {\n"
    "        importDst[ get_global_id( 0 )*importCopiesPerWorkItem+i ] = "
    "localBuffer[ get_local_id( 0 )*importCopiesPerWorkItem+i ];\n"
    "    }\n"
    "}\n";

int test_copy_fence(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, const char *kernelCode,
                    ExplicitType vecType, int vecSize, bool export_after_import,
                    bool aliased_local_mem, bool aliased_global_mem)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[4];
    size_t threads[1], localThreads[1];
    void *transaction1InBuffer, *transaction1OutBuffer, *transaction2InBuffer,
        *transaction2OutBuffer;
    MTdata d;
    bool transaction1DstIsTransaction2Src =
        (aliased_global_mem && !export_after_import)
        || (aliased_local_mem && export_after_import);
    bool transaction1SrcIsTransaction2Dst =
        aliased_global_mem && export_after_import;
    char vecNameString[64];
    vecNameString[0] = 0;
    if (vecSize == 1)
        sprintf(vecNameString, "%s", get_explicit_type_name(vecType));
    else
        sprintf(vecNameString, "%s%d", get_explicit_type_name(vecType),
                vecSize);

    size_t elementSize = get_explicit_type_size(vecType) * vecSize;
    log_info("Testing %s\n", vecNameString);

    cl_long max_local_mem_size;
    error =
        clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed.");

    unsigned int num_of_compute_devices;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(num_of_compute_devices),
                            &num_of_compute_devices, NULL);
    test_error(error,
               "clGetDeviceInfo for CL_DEVICE_MAX_COMPUTE_UNITS failed.");

    char programSource[4096];
    programSource[0] = 0;
    char *programPtr;

    sprintf(programSource, kernelCode,
            vecType == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                               : "",
            vecNameString, vecNameString, vecNameString, vecNameString,
            vecNameString, vecNameString, vecNameString,
            get_explicit_type_name(vecType), vecNameString,
            get_explicit_type_name(vecType), vecNameString, vecNameString,
            vecNameString, vecNameString);
    // log_info("program: %s\n", programSource);
    programPtr = programSource;

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        (const char **)&programPtr, "test_fn");
    test_error(error, "Unable to create testing kernel");

    size_t max_workgroup_size;
    error = clGetKernelWorkGroupInfo(
        kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_workgroup_size),
        &max_workgroup_size, NULL);
    test_error(
        error,
        "clGetKernelWorkGroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE.");

    size_t max_local_workgroup_size[3];
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(max_local_workgroup_size),
                            max_local_workgroup_size, NULL);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Pick the minimum of the device and the kernel
    if (max_workgroup_size > max_local_workgroup_size[0])
        max_workgroup_size = max_local_workgroup_size[0];

    size_t transaction1NumberOfCopiesPerWorkitem = 13;
    size_t transaction2NumberOfCopiesPerWorkitem = 2;
    elementSize =
        get_explicit_type_size(vecType) * ((vecSize == 3) ? 4 : vecSize);
    size_t localStorageSpacePerWorkitem =
        transaction1NumberOfCopiesPerWorkitem * elementSize
        + (aliased_local_mem
               ? 0
               : transaction2NumberOfCopiesPerWorkitem * elementSize);
    size_t maxLocalWorkgroupSize =
        (((int)max_local_mem_size / 2) / localStorageSpacePerWorkitem);

    // Calculation can return 0 on embedded devices due to 1KB local mem limit
    if (maxLocalWorkgroupSize == 0)
    {
        maxLocalWorkgroupSize = 1;
    }

    size_t localWorkgroupSize = maxLocalWorkgroupSize;
    if (maxLocalWorkgroupSize > max_workgroup_size)
        localWorkgroupSize = max_workgroup_size;

    size_t transaction1LocalBufferSize = localWorkgroupSize * elementSize
        * transaction1NumberOfCopiesPerWorkitem;
    size_t transaction2LocalBufferSize = localWorkgroupSize * elementSize
        * transaction2NumberOfCopiesPerWorkitem; // irrelevant if
                                                 // aliased_local_mem
    size_t numberOfLocalWorkgroups = 1111;
    size_t transaction1GlobalBufferSize =
        numberOfLocalWorkgroups * transaction1LocalBufferSize;
    size_t transaction2GlobalBufferSize =
        numberOfLocalWorkgroups * transaction2LocalBufferSize;
    size_t globalWorkgroupSize = numberOfLocalWorkgroups * localWorkgroupSize;

    transaction1InBuffer = (void *)malloc(transaction1GlobalBufferSize);
    transaction1OutBuffer = (void *)malloc(transaction1GlobalBufferSize);
    transaction2InBuffer = (void *)malloc(transaction2GlobalBufferSize);
    transaction2OutBuffer = (void *)malloc(transaction2GlobalBufferSize);
    memset(transaction1OutBuffer, 0, transaction1GlobalBufferSize);
    memset(transaction2OutBuffer, 0, transaction2GlobalBufferSize);

    cl_int transaction1CopiesPerWorkitemInt, transaction1CopiesPerWorkgroup,
        transaction2CopiesPerWorkitemInt, transaction2CopiesPerWorkgroup;
    transaction1CopiesPerWorkitemInt =
        (int)transaction1NumberOfCopiesPerWorkitem;
    transaction1CopiesPerWorkgroup =
        (int)(transaction1NumberOfCopiesPerWorkitem * localWorkgroupSize);
    transaction2CopiesPerWorkitemInt =
        (int)transaction2NumberOfCopiesPerWorkitem;
    transaction2CopiesPerWorkgroup =
        (int)(transaction2NumberOfCopiesPerWorkitem * localWorkgroupSize);

    log_info(
        "Global: %d, local %d. 1st Transaction: local buffer %db, global "
        "buffer %db, each work group will copy %d elements and each work "
        "item item will copy %d elements. 2nd Transaction: local buffer "
        "%db, global buffer %db, each work group will copy %d elements and "
        "each work item will copy %d elements\n",
        (int)globalWorkgroupSize, (int)localWorkgroupSize,
        (int)transaction1LocalBufferSize, (int)transaction1GlobalBufferSize,
        transaction1CopiesPerWorkgroup, transaction1CopiesPerWorkitemInt,
        (int)transaction2LocalBufferSize, (int)transaction2GlobalBufferSize,
        transaction2CopiesPerWorkgroup, transaction2CopiesPerWorkitemInt);

    threads[0] = globalWorkgroupSize;
    localThreads[0] = localWorkgroupSize;

    d = init_genrand(gRandomSeed);
    generate_random_data(
        vecType, transaction1GlobalBufferSize / get_explicit_type_size(vecType),
        d, transaction1InBuffer);
    if (!transaction1DstIsTransaction2Src)
    {
        generate_random_data(vecType,
                             transaction2GlobalBufferSize
                                 / get_explicit_type_size(vecType),
                             d, transaction2InBuffer);
    }
    free_mtdata(d);
    d = NULL;

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                transaction1GlobalBufferSize,
                                transaction1InBuffer, &error);
    test_error(error, "Unable to create input buffer");
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                transaction1GlobalBufferSize,
                                transaction1OutBuffer, &error);
    test_error(error, "Unable to create output buffer");
    if (!transaction1DstIsTransaction2Src)
    {
        streams[2] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                    transaction2GlobalBufferSize,
                                    transaction2InBuffer, &error);
        test_error(error, "Unable to create input buffer");
    }
    if (!transaction1SrcIsTransaction2Dst)
    {
        streams[3] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                    transaction2GlobalBufferSize,
                                    transaction2OutBuffer, &error);
        test_error(error, "Unable to create output buffer");
    }

    cl_uint argIndex = 0;
    error = clSetKernelArg(kernel, argIndex, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set kernel argument");
    ++argIndex;
    error = clSetKernelArg(kernel, argIndex, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set kernel argument");
    ++argIndex;
    if (!transaction1DstIsTransaction2Src)
    {
        error =
            clSetKernelArg(kernel, argIndex, sizeof(streams[2]), &streams[2]);
        test_error(error, "Unable to set kernel argument");
        ++argIndex;
    }
    if (!transaction1SrcIsTransaction2Dst)
    {
        error =
            clSetKernelArg(kernel, argIndex, sizeof(streams[3]), &streams[3]);
        test_error(error, "Unable to set kernel argument");
        ++argIndex;
    }
    error = clSetKernelArg(kernel, argIndex, transaction1LocalBufferSize, NULL);
    test_error(error, "Unable to set kernel argument");
    ++argIndex;
    if (!aliased_local_mem)
    {
        error =
            clSetKernelArg(kernel, argIndex, transaction2LocalBufferSize, NULL);
        test_error(error, "Unable to set kernel argument");
        ++argIndex;
    }
    error =
        clSetKernelArg(kernel, argIndex, sizeof(transaction1CopiesPerWorkgroup),
                       &transaction1CopiesPerWorkgroup);
    test_error(error, "Unable to set kernel argument");
    ++argIndex;
    error = clSetKernelArg(kernel, argIndex,
                           sizeof(transaction1CopiesPerWorkitemInt),
                           &transaction1CopiesPerWorkitemInt);
    test_error(error, "Unable to set kernel argument");
    ++argIndex;
    error =
        clSetKernelArg(kernel, argIndex, sizeof(transaction2CopiesPerWorkgroup),
                       &transaction2CopiesPerWorkgroup);
    test_error(error, "Unable to set kernel argument");
    ++argIndex;
    error = clSetKernelArg(kernel, argIndex,
                           sizeof(transaction2CopiesPerWorkitemInt),
                           &transaction2CopiesPerWorkitemInt);
    test_error(error, "Unable to set kernel argument");

    // Enqueue
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to queue kernel");

    // Read
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                transaction1GlobalBufferSize,
                                transaction1OutBuffer, 0, NULL, NULL);
    test_error(error, "Unable to read results");
    if (transaction1DstIsTransaction2Src)
    {
        for (size_t idx = 0; idx < numberOfLocalWorkgroups; idx++)
        {
            memcpy(
                (void *)((unsigned char *)transaction2InBuffer
                         + idx * transaction2CopiesPerWorkgroup * elementSize),
                (const void *)((unsigned char *)transaction1OutBuffer
                               + (idx * transaction1CopiesPerWorkgroup
                                  + (transaction1CopiesPerWorkgroup
                                     - transaction2CopiesPerWorkgroup))
                                   * elementSize),
                (size_t)transaction2CopiesPerWorkgroup * elementSize);
        }
    }
    if (transaction1SrcIsTransaction2Dst)
    {
        void *transaction1SrcBuffer =
            (void *)malloc(transaction1GlobalBufferSize);
        error = clEnqueueReadBuffer(queue, streams[0], CL_TRUE, 0,
                                    transaction1GlobalBufferSize,
                                    transaction1SrcBuffer, 0, NULL, NULL);
        test_error(error, "Unable to read results");
        for (size_t idx = 0; idx < numberOfLocalWorkgroups; idx++)
        {
            memcpy(
                (void *)((unsigned char *)transaction2OutBuffer
                         + idx * transaction2CopiesPerWorkgroup * elementSize),
                (const void *)((unsigned char *)transaction1SrcBuffer
                               + (idx * transaction1CopiesPerWorkgroup
                                  + (transaction1CopiesPerWorkgroup
                                     - transaction2CopiesPerWorkgroup))
                                   * elementSize),
                (size_t)transaction2CopiesPerWorkgroup * elementSize);
        }
        free(transaction1SrcBuffer);
    }
    else
    {
        error = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0,
                                    transaction2GlobalBufferSize,
                                    transaction2OutBuffer, 0, NULL, NULL);
        test_error(error, "Unable to read results");
    }

    // Verify
    int failuresPrinted = 0;
    if (memcmp(transaction1InBuffer, transaction1OutBuffer,
               transaction1GlobalBufferSize)
        != 0)
    {
        size_t typeSize = get_explicit_type_size(vecType) * vecSize;
        unsigned char *inchar = (unsigned char *)transaction1InBuffer;
        unsigned char *outchar = (unsigned char *)transaction1OutBuffer;
        for (int i = 0; i < (int)transaction1GlobalBufferSize;
             i += (int)elementSize)
        {
            if (memcmp(((char *)inchar) + i, ((char *)outchar) + i, typeSize)
                != 0)
            {
                char values[4096];
                values[0] = 0;
                if (failuresPrinted == 0)
                {
                    // Print first failure message
                    log_error("ERROR: Results of 1st transaction did not "
                              "validate!\n");
                }
                sprintf(values + strlen(values), "%d -> [", i);
                for (int j = 0; j < (int)elementSize; j++)
                    sprintf(values + strlen(values), "%2x ", inchar[i + j]);
                sprintf(values + strlen(values), "] != [");
                for (int j = 0; j < (int)elementSize; j++)
                    sprintf(values + strlen(values), "%2x ", outchar[i + j]);
                sprintf(values + strlen(values), "]");
                log_error("%s\n", values);
                failuresPrinted++;
            }

            if (failuresPrinted > 5)
            {
                log_error("Not printing further failures...\n");
                break;
            }
        }
    }
    if (memcmp(transaction2InBuffer, transaction2OutBuffer,
               transaction2GlobalBufferSize)
        != 0)
    {
        size_t typeSize = get_explicit_type_size(vecType) * vecSize;
        unsigned char *inchar = (unsigned char *)transaction2InBuffer;
        unsigned char *outchar = (unsigned char *)transaction2OutBuffer;
        for (int i = 0; i < (int)transaction2GlobalBufferSize;
             i += (int)elementSize)
        {
            if (memcmp(((char *)inchar) + i, ((char *)outchar) + i, typeSize)
                != 0)
            {
                char values[4096];
                values[0] = 0;
                if (failuresPrinted == 0)
                {
                    // Print first failure message
                    log_error("ERROR: Results of 2nd transaction did not "
                              "validate!\n");
                }
                sprintf(values + strlen(values), "%d -> [", i);
                for (int j = 0; j < (int)elementSize; j++)
                    sprintf(values + strlen(values), "%2x ", inchar[i + j]);
                sprintf(values + strlen(values), "] != [");
                for (int j = 0; j < (int)elementSize; j++)
                    sprintf(values + strlen(values), "%2x ", outchar[i + j]);
                sprintf(values + strlen(values), "]");
                log_error("%s\n", values);
                failuresPrinted++;
            }

            if (failuresPrinted > 5)
            {
                log_error("Not printing further failures...\n");
                break;
            }
        }
    }

    free(transaction1InBuffer);
    free(transaction1OutBuffer);
    free(transaction2InBuffer);
    free(transaction2OutBuffer);

    return failuresPrinted ? -1 : 0;
}

int test_copy_fence_all_types(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, const char *kernelCode,
                              bool export_after_import, bool aliased_local_mem,
                              bool aliased_global_mem)
{
    ExplicitType vecType[] = {
        kChar,  kUChar, kShort,  kUShort,          kInt, kUInt, kLong,
        kULong, kFloat, kDouble, kNumExplicitTypes
    };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int size, typeIndex;

    int errors = 0;

    if (!is_extension_available(deviceID, "cl_khr_async_work_group_copy_fence"))
    {
        log_info(
            "Device does not support extended async copies fence. Skipping "
            "test.\n");
        return 0;
    }

    for (typeIndex = 0; vecType[typeIndex] != kNumExplicitTypes; typeIndex++)
    {
        if (vecType[typeIndex] == kDouble
            && !is_extension_available(deviceID, "cl_khr_fp64"))
            continue;

        if ((vecType[typeIndex] == kLong || vecType[typeIndex] == kULong)
            && !gHasLong)
            continue;

        for (size = 0; vecSizes[size] != 0; size++)
        {
            if (test_copy_fence(deviceID, context, queue, kernelCode,
                                vecType[typeIndex], vecSizes[size],
                                export_after_import, aliased_local_mem,
                                aliased_global_mem))
            {
                errors++;
            }
        }
    }
    if (errors) return -1;
    return 0;
}

int test_async_work_group_copy_fence_import_after_export_aliased_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return test_copy_fence_all_types(deviceID, context, queue,
                                     import_after_export_aliased_local_kernel,
                                     false, true, false);
}

int test_async_work_group_copy_fence_import_after_export_aliased_global(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return test_copy_fence_all_types(deviceID, context, queue,
                                     import_after_export_aliased_global_kernel,
                                     false, false, true);
}

int test_async_work_group_copy_fence_import_after_export_aliased_global_and_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return test_copy_fence_all_types(
        deviceID, context, queue,
        import_after_export_aliased_global_and_local_kernel, false, true, true);
}

int test_async_work_group_copy_fence_export_after_import_aliased_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return test_copy_fence_all_types(deviceID, context, queue,
                                     export_after_import_aliased_local_kernel,
                                     true, true, false);
}

int test_async_work_group_copy_fence_export_after_import_aliased_global(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return test_copy_fence_all_types(deviceID, context, queue,
                                     export_after_import_aliased_global_kernel,
                                     true, false, true);
}

int test_async_work_group_copy_fence_export_after_import_aliased_global_and_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return test_copy_fence_all_types(
        deviceID, context, queue,
        export_after_import_aliased_global_and_local_kernel, true, true, true);
}
