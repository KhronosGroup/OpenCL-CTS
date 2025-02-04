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
#include "testBase.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"
#include "harness/conversions.h"

#include <vector>

typedef long long int lld;
typedef long long unsigned llu;

const char *test_kernels[] = {
"__kernel void kernelA(__global int *dst)\n"
"{\n"
"\n"
" dst[get_global_id(0)]*=3;\n"
"\n"
"}\n"
"__kernel void kernelB(__global int *dst)\n"
"{\n"
"\n"
" dst[get_global_id(0)]++;\n"
"\n"
"}\n"
};

#define TEST_SIZE 512
#define MAX_QUEUES 1000

const char *printPartition(cl_device_partition_property partition)
{
  switch (partition) {
    case (0):                                      return "<NONE>";
    case (CL_DEVICE_PARTITION_EQUALLY):            return "CL_DEVICE_PARTITION_EQUALLY";
    case (CL_DEVICE_PARTITION_BY_COUNTS):          return "CL_DEVICE_PARTITION_BY_COUNTS";
    case (CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN): return "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN";
    default:                                       return "<unknown>";
  } // switch
}

const char *printAffinity(cl_device_affinity_domain affinity)
{
  switch (affinity) {
    case (0):                                            return "<NONE>";
    case (CL_DEVICE_AFFINITY_DOMAIN_NUMA):               return "CL_DEVICE_AFFINITY_DOMAIN_NUMA";
    case (CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE):           return "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE";
    case (CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE):           return "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE";
    case (CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE):           return "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE";
    case (CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE):           return "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE";
    case (CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE): return "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE";
    default:                                             return "<unknown>";
  } // switch
}
int create_single_kernel_helper( cl_context context, cl_program *outProgram, cl_kernel *outKernel, unsigned int numKernelLines, const char **kernelProgram, const char *kernelName, const cl_device_id *parentDevice )
{
    int error = CL_SUCCESS;

    /* Create the program object from source */
    error = create_single_kernel_helper_create_program(context, outProgram, numKernelLines, kernelProgram);
    if( *outProgram == NULL || error != CL_SUCCESS)
    {
        print_error( error, "clCreateProgramWithSource failed" );
        return error;
    }

    /* Compile the program */
    int buildProgramFailed = 0;
    int printedSource = 0;
    error = clBuildProgram( *outProgram, ((parentDevice == NULL) ? 0 : 1), parentDevice, NULL, NULL, NULL );
    if (error != CL_SUCCESS)
    {
        unsigned int i;
        print_error(error, "clBuildProgram failed");
        buildProgramFailed = 1;
        printedSource = 1;
        log_error( "Original source is: ------------\n" );
        for( i = 0; i < numKernelLines; i++ )
            log_error( "%s", kernelProgram[ i ] );
    }

    // Verify the build status on all devices
    cl_uint deviceCount = 0;
    error = clGetProgramInfo( *outProgram, CL_PROGRAM_NUM_DEVICES, sizeof( deviceCount ), &deviceCount, NULL );
    if (error != CL_SUCCESS) {
        print_error(error, "clGetProgramInfo CL_PROGRAM_NUM_DEVICES failed");
        return error;
    }

    if (deviceCount == 0) {
        log_error("No devices found for program.\n");
        return -1;
    }

    cl_device_id    *devices = (cl_device_id*) malloc( deviceCount * sizeof( cl_device_id ) );
    if( NULL == devices )
        return -1;
    memset( devices, 0, deviceCount * sizeof( cl_device_id ));
    error = clGetProgramInfo( *outProgram, CL_PROGRAM_DEVICES, sizeof( cl_device_id ) * deviceCount, devices, NULL );
    if (error != CL_SUCCESS) {
        print_error(error, "clGetProgramInfo CL_PROGRAM_DEVICES failed");
        free( devices );
        return error;
    }

    cl_uint z;
    for( z = 0; z < deviceCount; z++ )
    {
        char deviceName[4096] = "";
        error = clGetDeviceInfo(devices[z], CL_DEVICE_NAME, sizeof( deviceName), deviceName, NULL);
        if (error != CL_SUCCESS || deviceName[0] == '\0') {
            log_error("Device \"%d\" failed to return a name\n", z);
            print_error(error, "clGetDeviceInfo CL_DEVICE_NAME failed");
        }

        cl_build_status buildStatus;
        error = clGetProgramBuildInfo(*outProgram, devices[z], CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, NULL);
        if (error != CL_SUCCESS) {
            print_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_STATUS failed");
            free( devices );
            return error;
        }

        if (buildStatus != CL_BUILD_SUCCESS || buildProgramFailed) {
            char log[10240] = "";
            if (buildStatus == CL_BUILD_SUCCESS && buildProgramFailed) log_error("clBuildProgram returned an error, but buildStatus is marked as CL_BUILD_SUCCESS.\n");

            char statusString[64] = "";
            if (buildStatus == (cl_build_status)CL_BUILD_SUCCESS)
                sprintf(statusString, "CL_BUILD_SUCCESS");
            else if (buildStatus == (cl_build_status)CL_BUILD_NONE)
                sprintf(statusString, "CL_BUILD_NONE");
            else if (buildStatus == (cl_build_status)CL_BUILD_ERROR)
                sprintf(statusString, "CL_BUILD_ERROR");
            else if (buildStatus == (cl_build_status)CL_BUILD_IN_PROGRESS)
                sprintf(statusString, "CL_BUILD_IN_PROGRESS");
            else
                sprintf(statusString, "UNKNOWN (%d)", buildStatus);

            if (buildStatus != CL_BUILD_SUCCESS) log_error("Build not successful for device \"%s\", status: %s\n", deviceName, statusString);
            error = clGetProgramBuildInfo( *outProgram, devices[z], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL );
            if (error != CL_SUCCESS || log[0]=='\0'){
                log_error("Device %d (%s) failed to return a build log\n", z, deviceName);
                if (error) {
                    print_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed");
                    free( devices );
                    return error;
                } else {
                    log_error("clGetProgramBuildInfo returned an empty log.\n");
                    free( devices );
                    return -1;
                }
            }
            // In this case we've already printed out the code above.
            if (!printedSource)
            {
                unsigned int i;
                log_error( "Original source is: ------------\n" );
                for( i = 0; i < numKernelLines; i++ )
                    log_error( "%s", kernelProgram[ i ] );
                printedSource = 1;
            }
            log_error( "Build log for device \"%s\" is: ------------\n", deviceName );
            log_error( "%s\n", log );
            log_error( "\n----------\n" );
            free( devices );
            return -1;
        }
    }

    /* And create a kernel from it */
    *outKernel = clCreateKernel( *outProgram, kernelName, &error );
    if( *outKernel == NULL || error != CL_SUCCESS)
    {
        print_error( error, "Unable to create kernel" );
        free( devices );
        return error;
    }

    free( devices );
    return 0;
}

template<class T>
class AutoDestructArray
{
public:
    AutoDestructArray(T* arr) : m_arr(arr) {}
    ~AutoDestructArray() { if (m_arr) delete [] m_arr; }

private:
    T* m_arr;
};

int test_device_set(size_t deviceCount, size_t queueCount, cl_device_id *devices, int num_elements, cl_device_id *parentDevice = NULL)
{
    int error;
    clContextWrapper context;
    clProgramWrapper program;
    clKernelWrapper kernels[2];
    clMemWrapper  stream;
    clCommandQueueWrapper queues[MAX_QUEUES] = {};
    size_t threads[1], localThreads[1];
    int data[TEST_SIZE];
    int outputData[TEST_SIZE];
    int expectedResults[TEST_SIZE];
    int *expectedResultsOneDeviceArray = new int[deviceCount * TEST_SIZE];
    int **expectedResultsOneDevice = (int**)alloca(sizeof(int**) * deviceCount);
    size_t i;
    AutoDestructArray<int> autoDestruct(expectedResultsOneDeviceArray);

    for (i=0; i<deviceCount; i++) {
        expectedResultsOneDevice[i] = expectedResultsOneDeviceArray + (i * TEST_SIZE);
    }

    RandomSeed seed( gRandomSeed );

    if (queueCount > MAX_QUEUES) {
        log_error("Number of queues (%zu) is greater than the number for which "
                  "the test was written (%d).",
                  queueCount, MAX_QUEUES);
        return -1;
    }

    log_info("Testing with %zu queues on %zu devices, %zu kernel executions.\n",
             queueCount, deviceCount, queueCount * num_elements / TEST_SIZE);

    for (i=0; i<deviceCount; i++) {
        size_t deviceNameSize;
        error = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
        test_error(error, "clGetDeviceInfo CL_DEVICE_NAME failed");
        char *deviceName = (char *)alloca(deviceNameSize * (sizeof(char)));
        error = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, deviceNameSize, deviceName, NULL);
        test_error(error, "clGetDeviceInfo CL_DEVICE_NAME failed");
        log_info("Device %zu is \"%s\".\n", i, deviceName);
    }

    /* Create a context */
    context = clCreateContext( NULL, (cl_uint)deviceCount, devices, notify_callback, NULL, &error );
    test_error( error, "Unable to create testing context" );

    /* Create our kernels (they all have the same arguments so we don't need multiple ones for each device) */
    if( create_single_kernel_helper( context, &program, &kernels[0], 1, test_kernels, "kernelA", parentDevice ) != 0 )
    {
        return -1;
    }

    kernels[1] = clCreateKernel(program, "kernelB", &error);
    test_error(error, "clCreateKernel failed");


    /* Now create I/O streams */
    for( i = 0; i < TEST_SIZE; i++ )
        data[i] = genrand_int32(seed);

    stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                            sizeof(cl_int) * TEST_SIZE, data, &error);
    test_error( error, "Unable to create test array" );

    // Update the expected results
    for( i = 0; i < TEST_SIZE; i++ ) {
        expectedResults[i] = data[i];
        for (size_t j=0; j<deviceCount; j++)
            expectedResultsOneDevice[j][i] = data[i];
    }


    // Set the arguments
    error = clSetKernelArg( kernels[0], 0, sizeof( stream ), &stream);
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernels[1], 0, sizeof( stream ), &stream);
    test_error( error, "Unable to set kernel arguments" );

    /* Run the test */
    threads[0] = (size_t)TEST_SIZE;

    error = get_max_common_work_group_size( context, kernels[0], threads[0], &localThreads[ 0 ] );
    test_error( error, "Unable to calc work group size" );

    /* Create work queues */
    for( i = 0; i < queueCount; i++ )
    {
        queues[i] = clCreateCommandQueueWithProperties( context, devices[ i % deviceCount ], 0, &error );
        if (error != CL_SUCCESS || queues[i] == NULL) {
            log_info("Could not create queue[%d].\n", (int)i);
            queueCount = i;
            break;
        }
    }
    log_info("Testing with %d queues.\n", (int)queueCount);

    /* Enqueue executions */
    for( int z = 0; z<num_elements/TEST_SIZE; z++) {
        for( i = 0; i < queueCount; i++ )
        {
            // Randomly choose a kernel to execute.
            int kernel_selection = (int)get_random_float(0, 2, seed);
            error = clEnqueueNDRangeKernel( queues[ i ], kernels[ kernel_selection ], 1, NULL, threads, localThreads, 0, NULL, NULL );
            test_error( error, "Kernel execution failed" );

            // Update the expected results
            for( int j = 0; j < TEST_SIZE; j++ ) {
                expectedResults[j] = (kernel_selection) ? expectedResults[j]+1 : expectedResults[j]*3;
                expectedResultsOneDevice[i % deviceCount][j] = (kernel_selection) ? expectedResultsOneDevice[i % deviceCount][j]+1 : expectedResultsOneDevice[i % deviceCount][j]*3;
            }

            // Force the queue to finish so the next one will be in sync
            error = clFinish(queues[i]);
            test_error( error, "clFinish failed");
        }
    }

    /* Read results */
    int errors = 0;
    for (int q = 0; q<(int)queueCount; q++) {
        error = clEnqueueReadBuffer( queues[ q ], stream, CL_TRUE, 0, sizeof(cl_int)*TEST_SIZE, (char *)outputData, 0, NULL, NULL );
        test_error( error, "Unable to get result data set" );

        int errorsThisTime = 0;
        /* Verify all of the data now */
        for( i = 0; i < TEST_SIZE; i++ )
        {
            if( expectedResults[ i ] != outputData[ i ] )
            {
                log_error("ERROR: Sample data did not verify for queue %d on "
                          "device %zu (sample %zu, expected %d, got %d)\n",
                          q, q % deviceCount, i, expectedResults[i],
                          outputData[i]);
                for (size_t j=0; j<deviceCount; j++) {
                    if (expectedResultsOneDevice[j][i] == outputData[i])
                        log_info("Sample consistent with only device %zu "
                                 "having modified the data.\n",
                                 j);
                }
                errorsThisTime++;
                break;
            }
        }
        if (errorsThisTime)
            errors++;
    }

    /* All done now! */
    if (errors)
        return -1;
    return 0;
}


int init_device_partition_test(cl_device_id parentDevice,
                               cl_uint &maxComputeUnits, cl_uint &maxSubDevices)
{
    int err = clGetDeviceInfo(parentDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    test_error( err, "Unable to get maximal number of compute units" );
    err = clGetDeviceInfo(parentDevice, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, sizeof(maxSubDevices), &maxSubDevices, NULL);
    test_error( err, "Unable to get maximal number of sub-devices" );

    log_info("Maximal number of sub-devices on device %p is %d.\n", parentDevice, maxSubDevices );
    return 0;
}

int test_device_partition_type_support(cl_device_id parentDevice, const cl_device_partition_property partitionType, const cl_device_affinity_domain affinityDomain)
{
    typedef std::vector< cl_device_partition_property > properties_t;
    properties_t supportedProps( 3 ); // only 3 types defined in the spec (but implementation can define more)
    size_t const propSize = sizeof( cl_device_partition_property ); // Size of one property in bytes.
    size_t size;    // size of all properties in bytes.
    cl_int err;
    size = 0;
    err = clGetDeviceInfo( parentDevice, CL_DEVICE_PARTITION_PROPERTIES, 0, NULL, & size );
    if ( err == CL_SUCCESS ) {
        if ( size % propSize != 0 ) {
            log_error( "ERROR: clGetDeviceInfo: Bad size of returned partition properties (%llu), it must me a multiply of partition property size (%llu)\n", llu( size ), llu( propSize ) );
            return -1;
        }
        supportedProps.resize( size / propSize );
        size = 0;
        err = clGetDeviceInfo( parentDevice, CL_DEVICE_PARTITION_PROPERTIES, supportedProps.size() * propSize, & supportedProps.front(), & size );
        test_error_ret( err, "Unable to get device partition properties (2)", -1 );
    } else if ( err == CL_INVALID_VALUE ) {
        log_error( "ERROR: clGetDeviceInfo: CL_DEVICE_PARTITION_PROPERTIES is not supported.\n" );
        return -1;
    } else {
        test_error_ret( err, "Unable to get device partition properties (1)", -1 );
    };
    for (size_t i = 0; i < supportedProps.size(); i++)
    {
        if (supportedProps[i] == partitionType)
        {
           if (partitionType == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN)
           {
              cl_device_affinity_domain supportedAffinityDomain;
              err = clGetDeviceInfo(parentDevice, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, sizeof(supportedAffinityDomain), &supportedAffinityDomain, NULL);
              test_error( err, "Unable to get supported affinity domains" );
              if (supportedAffinityDomain & affinityDomain)
                return 0;
           }
           else
            return 0;
        }
    }

    return -1;
}

int test_partition_of_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, cl_device_partition_property *partition_type,
                             cl_uint starting_property, cl_uint ending_property)
{
    cl_uint maxComputeUnits;
    cl_uint maxSubDevices;    // maximal number of sub-devices that can be created in one call to clCreateSubDevices
    int err = 0;

    if (init_device_partition_test(deviceID, maxComputeUnits, maxSubDevices) != 0)
        return -1;

    if (maxComputeUnits <= 1)
        return 0;
    // confirm that this devices reports how it was partitioned
    if (partition_type != NULL)
    { // if we're not the root device
      size_t psize;
      err = clGetDeviceInfo(deviceID, CL_DEVICE_PARTITION_TYPE, 0,  NULL, &psize);
      test_error( err, "Unable to get CL_DEVICE_PARTITION_TYPE" );
      cl_device_partition_property *properties_returned = (cl_device_partition_property *)alloca(psize);
      err = clGetDeviceInfo(deviceID, CL_DEVICE_PARTITION_TYPE, psize, (void *) properties_returned, NULL);
      test_error( err, "Unable to get CL_DEVICE_PARTITION_TYPE" );

      // test returned type
      for (cl_uint i = 0;i < psize / sizeof(cl_device_partition_property);i++) {
        if (properties_returned[i] != partition_type[i]) {
          if (!(partition_type[0] == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN &&
              i == 1 && partition_type[1] == CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE &&
              (properties_returned[1] == CL_DEVICE_AFFINITY_DOMAIN_NUMA     ||
               properties_returned[1] == CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ||
               properties_returned[1] == CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ||
               properties_returned[1] == CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ||
               properties_returned[1] == CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE))) {
              log_error(
                  "properties_returned[%d] 0x%x != 0x%x partition_type[%d].", i,
                  static_cast<unsigned int>(properties_returned[i]),
                  static_cast<unsigned int>(partition_type[i]), i);
              return -1;
          }
        }
      } // for
    }

#define PROPERTY_TYPES 8
    cl_device_partition_property partitionProp[PROPERTY_TYPES][5] = {
        { CL_DEVICE_PARTITION_EQUALLY, (cl_int)maxComputeUnits / 2, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_COUNTS, 1, (cl_int)maxComputeUnits - 1,
          CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_NUMA, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0, 0, 0 }
    };

    // loop thru each type, creating sub-devices for each type
    for (cl_uint i = starting_property;i < ending_property;i++) {

      if (test_device_partition_type_support(deviceID, partitionProp[i][0], partitionProp[i][1]) != 0)
      {
        if (partitionProp[i][0] == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN)
        {
          log_info( "Device partition type \"%s\" \"%s\" is not supported on device %p. Skipping test...\n",
                      printPartition(partitionProp[i][0]),
                      printAffinity(partitionProp[i][1]), deviceID);
        }
        else
        {
          log_info( "Device partition type \"%s\" is not supported on device %p. Skipping test...\n",
                      printPartition(partitionProp[i][0]), deviceID);
        }
        continue;
      }

      if (partitionProp[i][0] == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN)
      {
        log_info("Testing on device %p partition type \"%s\" \"%s\"\n", deviceID, printPartition(partitionProp[i][0]),
                  printAffinity(partitionProp[i][1]));
      }
      else
      {
          log_info("Testing on device %p partition type \"%s\" (%d,%d)\n",
                   deviceID, printPartition(partitionProp[i][0]),
                   static_cast<unsigned int>(partitionProp[i][1]),
                   static_cast<unsigned int>(partitionProp[i][2]));
      }

      cl_uint deviceCount;

      // how many sub-devices can we create?
      err = clCreateSubDevices(deviceID, partitionProp[i], 0, NULL, &deviceCount);
      if ( err == CL_DEVICE_PARTITION_FAILED ) {
          log_info( "The device %p could not be further partitioned.\n", deviceID );
          continue;
      }
      test_error( err, "Failed to get number of sub-devices" );

      // get the list of subDevices
      //  create room for 1 more device_id, so that we can put the parent device in there.
      cl_device_id *subDevices = (cl_device_id*)alloca(sizeof(cl_device_id) * (deviceCount + 1));
      err = clCreateSubDevices(deviceID, partitionProp[i], deviceCount, subDevices, &deviceCount);
      test_error( err, "Actual creation of sub-devices failed" );

      log_info("Testing on all devices in context\n");
      err = test_device_set(deviceCount, deviceCount, subDevices, num_elements);
      if (err == 0)
      {
          log_info("Testing on a parent device for context\n");

          // add the parent device
          subDevices[deviceCount] = deviceID;
          err = test_device_set(deviceCount + 1, deviceCount, subDevices, num_elements, &deviceID);
      }
      if (err != 0)
      {
          printf("error! returning %d\n",err);
          return err;
      }

      // now, recurse and test the FIRST of these sub-devices, to make sure it can be further partitioned
      err = test_partition_of_device(subDevices[0], context, queue, num_elements, partitionProp[i], starting_property, ending_property);
      if (err != 0)
      {
          printf("error! returning %d\n",err);
          return err;
      }

      for (cl_uint j=0;j < deviceCount;j++)
      {
        err = clReleaseDevice(subDevices[j]);
        test_error( err, "\n Releasing sub-device failed \n" );
      }

    } // for

    log_info("Testing on all device %p finished\n", deviceID);
    return 0;
}


int test_partition_equally(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 0, 1);
}

int test_partition_by_counts(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 1, 2);
}

int test_partition_by_affinity_domain_numa(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 2, 3);
}

int test_partition_by_affinity_domain_l4_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 3, 4);
}

int test_partition_by_affinity_domain_l3_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 4, 5);
}

int test_partition_by_affinity_domain_l2_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 5, 6);
}

int test_partition_by_affinity_domain_l1_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 6, 7);
}

int test_partition_by_affinity_domain_next_partitionable(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 7, 8);
}

int test_partition_all(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_partition_of_device(deviceID, context, queue, num_elements, NULL, 0, 8);
}
