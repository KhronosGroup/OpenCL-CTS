#include "harness/typeWrappers.h"
#include "harness/extensionHelpers.h"
#include "harness/errorHelpers.h"
#include "vulkan_wrapper/opencl_vulkan_wrapper.hpp"

#define SEMAPHORE_PARAM_TEST(param_name, param_type, expected)                 \
    do                                                                         \
    {                                                                          \
        param_type value;                                                      \
        size_t size;                                                           \
        cl_int error = clGetSemaphoreInfoKHR(sema, param_name, sizeof(value),  \
                                             &value, &size);                   \
        test_error(error, "Unable to get " #param_name " from semaphore");     \
        if (value != expected)                                                 \
        {                                                                      \
            test_fail("ERROR: Parameter %s did not validate! (expected %d, "   \
                      "got %d)\n",                                             \
                      #param_name, expected, value);                           \
        }                                                                      \
        if (size != sizeof(value))                                             \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Returned size of parameter %s does not validate! "     \
                "(expected %d, got %d)\n",                                     \
                #param_name, (int)sizeof(value), (int)size);                   \
        }                                                                      \
    } while (false)

// Confirm the semaphores can be successfully queried
int test_external_semaphores_queries(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int err = CL_SUCCESS;

    cl_platform_id platform = nullptr;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Error: Failed to get platform\n");
        return err;
    }

    init_cl_vk_ext(platform);

    VulkanDevice vkDevice;

    GET_PFN(deviceID, clGetSemaphoreInfoKHR);
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);
    GET_PFN(deviceID, clReleaseSemaphoreKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    cl_context context2 =
        clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &err);
    if (!context2)
    {
        print_error(err, "Unable to create testing context");
        return TEST_FAIL;
    }

    clExternalSemaphore sema1(vkVk2CLSemaphore, context,
                              vkExternalSemaphoreHandleType, deviceID);

    // Needed by the macro
    cl_semaphore_khr sema = sema1.getCLSemaphore();

    SEMAPHORE_PARAM_TEST(CL_DEVICE_HANDLE_LIST_KHR, cl_uint, 1);

    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR, cl_uint, 1);

    err = clReleaseSemaphoreKHR(sema);
    test_error(err, "Could not release semaphore");

    return TEST_PASS;
}


int test_external_semaphores_multi_context(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue defaultQueue,
                                           int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int err = CL_SUCCESS;

    cl_platform_id platform = nullptr;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Error: Failed to get platform\n");
        return err;
    }

    init_cl_vk_ext(platform);

    VulkanDevice vkDevice;

    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);
    GET_PFN(deviceID, clReleaseSemaphoreKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    cl_context context2 =
        clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &err);
    if (!context2)
    {
        print_error(err, "Unable to create testing context");
        return TEST_FAIL;
    }

    clExternalSemaphore sema1(vkVk2CLSemaphore, context,
                              vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema2(vkVk2CLSemaphore, context2,
                              vkExternalSemaphoreHandleType, deviceID);

    clCommandQueueWrapper queue1 =
        clCreateCommandQueue(context, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    clCommandQueueWrapper queue2 =
        clCreateCommandQueue(context2, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore 1 and 2
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue1, 1, &sema1.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 1
    clEventWrapper wait_1_event;
    err = clEnqueueWaitSemaphoresKHR(queue1, 1, &sema1.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_1_event);
    test_error(err, "Could not wait semaphore");

    err = clEnqueueSignalSemaphoresKHR(queue2, 1, &sema2.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 2
    clEventWrapper wait_2_event;
    err = clEnqueueWaitSemaphoresKHR(queue2, 1, &sema2.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_2_event);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue1);
    test_error(err, "Could not finish queue");

    err = clFinish(queue2);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_1_event);
    test_assert_event_complete(wait_2_event);

    return TEST_PASS;
}

int test_external_semaphores_in_order_queue(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue defaultQueue,
                                            int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int err = CL_SUCCESS;

    cl_platform_id platform = nullptr;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Error: Failed to get platform\n");
        return err;
    }

    init_cl_vk_ext(platform);

    VulkanDevice vkDevice;

    cl_context context2 =
        clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &err);
    if (!context2)
    {
        print_error(err, "Unable to create testing context");
        return TEST_FAIL;
    }

    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);
    GET_PFN(deviceID, clReleaseSemaphoreKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema1(vkVk2CLSemaphore, context,
                              vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema2(vkVk2CLSemaphore, context2,
                              vkExternalSemaphoreHandleType, deviceID);

    clCommandQueueWrapper queue1 =
        clCreateCommandQueue(context, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    clCommandQueueWrapper queue2 =
        clCreateCommandQueue(context2, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore 1 and 2
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue1, 1, &sema1.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 1
    clEventWrapper wait_1_event;
    err = clEnqueueWaitSemaphoresKHR(queue1, 1, &sema1.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_1_event);
    test_error(err, "Could not wait semaphore");

    err = clEnqueueSignalSemaphoresKHR(queue2, 1, &sema2.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 2
    clEventWrapper wait_2_event;
    err = clEnqueueWaitSemaphoresKHR(queue2, 1, &sema2.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_2_event);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue1);
    test_error(err, "Could not finish queue");

    err = clFinish(queue2);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_1_event);
    test_assert_event_complete(wait_2_event);

    return TEST_PASS;
}
