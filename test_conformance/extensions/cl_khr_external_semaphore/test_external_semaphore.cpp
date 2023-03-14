#include "harness/typeWrappers.h"
#include "harness/extensionHelpers.h"
#include "harness/errorHelpers.h"
#include "opencl_vulkan_wrapper.hpp"
#include <thread>
#include <chrono>

#define FLUSH_DELAY_S 5

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

#define SEMAPHORE_PARAM_TEST_ARRAY(param_name, param_type, num_params,         \
                                   expected)                                   \
    do                                                                         \
    {                                                                          \
        param_type value[num_params];                                          \
        size_t size;                                                           \
        cl_int error = clGetSemaphoreInfoKHR(sema, param_name, sizeof(value),  \
                                             &value, &size);                   \
        test_error(error, "Unable to get " #param_name " from semaphore");     \
        if (size != sizeof(value))                                             \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Returned size of parameter %s does not validate! "     \
                "(expected %d, got %d)\n",                                     \
                #param_name, (int)sizeof(value), (int)size);                   \
        }                                                                      \
        if (memcmp(value, expected, size) != 0)                                \
        {                                                                      \
            test_fail("ERROR: Parameter %s did not validate!\n", #param_name); \
        }                                                                      \
    } while (false)

static const char* source = "__kernel void empty() {}";

static int init_vuikan_device()
{
    cl_platform_id platform = nullptr;

    cl_int err = CL_SUCCESS;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Error: Failed to get platform\n");
        return err;
    }

    init_cl_vk_ext(platform);

    return CL_SUCCESS;
}

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

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    GET_PFN(deviceID, clGetSemaphoreInfoKHR);
    GET_PFN(deviceID, clReleaseSemaphoreKHR);
    GET_PFN(deviceID, clRetainSemaphoreKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    // Needed by the macro
    cl_semaphore_khr sema = sema_ext.getCLSemaphore();

    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_TYPE_KHR, cl_semaphore_type_khr,
                         CL_SEMAPHORE_TYPE_BINARY_KHR);

    SEMAPHORE_PARAM_TEST(CL_DEVICE_HANDLE_LIST_KHR, cl_uint, 1);

    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR, cl_uint, 1);

    // Confirm that querying CL_SEMAPHORE_CONTEXT_KHR returns the right context
    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_CONTEXT_KHR, cl_context, context);

    // Confirm that querying CL_SEMAPHORE_REFERENCE_COUNT_KHR returns the right
    // value
    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, 1);

    cl_int err = CL_SUCCESS;

    err = clRetainSemaphoreKHR(sema);
    test_error(err, "Could not retain semaphore");
    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, 2);

    err = clReleaseSemaphoreKHR(sema);
    test_error(err, "Could not release semaphore");
    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, 1);

    // Confirm that querying CL_SEMAPHORE_PAYLOAD_KHR returns the unsignaled
    // state
    SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_PAYLOAD_KHR, cl_semaphore_payload_khr, 0);

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

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    cl_int err = CL_SUCCESS;

    cl_context context2 =
        clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &err);
    if (!context2)
    {
        print_error(err, "Unable to create testing context");
        return TEST_FAIL;
    }

    clExternalSemaphore sema_ext_1(vkVk2CLSemaphore, context,
                                   vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema_ext_2(vkVk2CLSemaphore, context2,
                                   vkExternalSemaphoreHandleType, deviceID);

    clCommandQueueWrapper queue1 =
        clCreateCommandQueue(context, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    clCommandQueueWrapper queue2 =
        clCreateCommandQueue(context2, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore 1 and 2
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue1, 1, &sema_ext_1.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 1
    clEventWrapper wait_1_event;
    err = clEnqueueWaitSemaphoresKHR(queue1, 1, &sema_ext_1.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_1_event);
    test_error(err, "Could not wait semaphore");

    err = clEnqueueSignalSemaphoresKHR(queue2, 1, &sema_ext_2.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 2
    clEventWrapper wait_2_event;
    err = clEnqueueWaitSemaphoresKHR(queue2, 1, &sema_ext_2.getCLSemaphore(),
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

// Helper function that signals and waits on semaphore across two different
// queues.
static int semaphore_external_cross_queue_helper(cl_device_id deviceID,
                                                 cl_context context,
                                                 cl_command_queue queue_1,
                                                 cl_command_queue queue_2)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    cl_int err = CL_SUCCESS;

    // Signal semaphore on queue_1
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue_1, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore on queue_2
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue_2, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Finish queue_1 and queue_2
    err = clFinish(queue_1);
    test_error(err, "Could not finish queue");

    err = clFinish(queue_2);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that a signal followed by a wait will complete successfully
int test_external_semaphores_simple_1(cl_device_id deviceID, cl_context context,
                                      cl_command_queue defaultQueue,
                                      int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that signal a semaphore with no event dependencies will not result
// in an implicit dependency on everything previously submitted
int test_external_semaphores_simple_2(cl_device_id deviceID, cl_context context,
                                      cl_command_queue defaultQueue,
                                      int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Create user event
    clEventWrapper user_event = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    // Create Kernel
    clProgramWrapper program;
    clKernelWrapper kernel;
    err = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                      "empty");
    test_error(err, "Could not create kernel");

    // Enqueue task_1 (dependency on user_event)
    clEventWrapper task_1_event;
    err = clEnqueueTask(queue, kernel, 1, &user_event, &task_1_event);
    test_error(err, "Could not enqueue task 1");

    // Signal semaphore
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Flush and delay
    err = clFlush(queue);
    test_error(err, "Could not flush queue");
    std::this_thread::sleep_for(std::chrono::seconds(FLUSH_DELAY_S));

    // Ensure all events are completed except for task_1
    test_assert_event_inprogress(task_1_event);
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    // Complete user_event
    err = clSetUserEventStatus(user_event, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(task_1_event);
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that a semaphore can be reused multiple times
int test_external_semaphores_reuse(cl_device_id deviceID, cl_context context,
                                   cl_command_queue defaultQueue,
                                   int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Create Kernel
    clProgramWrapper program;
    clKernelWrapper kernel;
    err = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                      "empty");
    test_error(err, "Could not create kernel");

    constexpr size_t loop_count = 10;
    clEventWrapper signal_events[loop_count];
    clEventWrapper wait_events[loop_count];
    clEventWrapper task_events[loop_count];

    // Enqueue task_1
    err = clEnqueueTask(queue, kernel, 0, nullptr, &task_events[0]);
    test_error(err, "Unable to enqueue task_1");

    // Signal semaphore (dependency on task_1)
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 1, &task_events[0],
                                       &signal_events[0]);
    test_error(err, "Could not signal semaphore");

    // In a loop
    size_t loop;
    for (loop = 1; loop < loop_count; ++loop)
    {
        // Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                         nullptr, 0, nullptr,
                                         &wait_events[loop - 1]);
        test_error(err, "Could not wait semaphore");

        // Enqueue task_loop (dependency on wait)
        err = clEnqueueTask(queue, kernel, 1, &wait_events[loop - 1],
                            &task_events[loop]);
        test_error(err, "Unable to enqueue task_loop");

        // Wait for the "wait semaphore" to complete
        err = clWaitForEvents(1, &wait_events[loop - 1]);
        test_error(err, "Unable to wait for wait semaphore to complete");

        // Signal semaphore (dependency on task_loop)
        err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                           nullptr, 1, &task_events[loop],
                                           &signal_events[loop]);
        test_error(err, "Could not signal semaphore");
    }

    // Wait semaphore
    err =
        clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                   nullptr, 0, nullptr, &wait_events[loop - 1]);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    for (loop = 0; loop < loop_count; ++loop)
    {
        test_assert_event_complete(wait_events[loop]);
        test_assert_event_complete(signal_events[loop]);
        test_assert_event_complete(task_events[loop]);
    }

    return TEST_PASS;
}

// Helper function that signals and waits on semaphore across two different
// queues.
static int external_semaphore_cross_queue_helper(cl_device_id deviceID,
                                                 cl_context context,
                                                 cl_command_queue queue_1,
                                                 cl_command_queue queue_2)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Signal semaphore on queue_1
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue_1, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore on queue_2
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue_2, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Finish queue_1 and queue_2
    err = clFinish(queue_1);
    test_error(err, "Could not finish queue");

    err = clFinish(queue_2);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}


// Confirm that a semaphore works across different ooo queues
int test_external_semaphores_cross_queues_ooo(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue defaultQueue,
                                              int num_elements)
{
    cl_int err;

    // Create ooo queues
    clCommandQueueWrapper queue_1 = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    clCommandQueueWrapper queue_2 = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    return external_semaphore_cross_queue_helper(deviceID, context, queue_1,
                                                 queue_2);
}

// Confirm that a semaphore works across different in-order queues
int test_external_semaphores_cross_queues_io(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue defaultQueue,
                                             int num_elements)
{
    cl_int err;

    // Create in-order queues
    clCommandQueueWrapper queue_1 =
        clCreateCommandQueue(context, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    clCommandQueueWrapper queue_2 =
        clCreateCommandQueue(context, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    return external_semaphore_cross_queue_helper(deviceID, context, queue_1,
                                                 queue_2);
}

int test_external_semaphores_cross_queues_io2(cl_device_id deviceID,
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

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    cl_int err = CL_SUCCESS;

    cl_context context2 =
        clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &err);
    if (!context2)
    {
        print_error(err, "Unable to create testing context");
        return TEST_FAIL;
    }

    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext_1(vkVk2CLSemaphore, context,
                                   vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema_ext_2(vkVk2CLSemaphore, context2,
                                   vkExternalSemaphoreHandleType, deviceID);

    clCommandQueueWrapper queue1 =
        clCreateCommandQueue(context, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    clCommandQueueWrapper queue2 =
        clCreateCommandQueue(context2, deviceID, 0, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore 1 and 2
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue1, 1, &sema_ext_1.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 1
    clEventWrapper wait_1_event;
    err = clEnqueueWaitSemaphoresKHR(queue1, 1, &sema_ext_1.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_1_event);
    test_error(err, "Could not wait semaphore");

    err = clEnqueueSignalSemaphoresKHR(queue2, 1, &sema_ext_2.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 2
    clEventWrapper wait_2_event;
    err = clEnqueueWaitSemaphoresKHR(queue2, 1, &sema_ext_2.getCLSemaphore(),
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

// Confirm that we can signal multiple semaphores with one command
int test_external_semaphores_multi_signal(cl_device_id deviceID,
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

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore1(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkVk2CLSemaphore2(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext_1(vkVk2CLSemaphore1, context,
                                   vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema_ext_2(vkVk2CLSemaphore2, context,
                                   vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore 1 and 2
    clEventWrapper signal_event;
    cl_semaphore_khr sema_list[] = { sema_ext_1.getCLSemaphore(),
                                     sema_ext_2.getCLSemaphore() };
    err = clEnqueueSignalSemaphoresKHR(queue, 2, sema_list, nullptr, 0, nullptr,
                                       &signal_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 1
    clEventWrapper wait_1_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext_1.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_1_event);
    test_error(err, "Could not wait semaphore");

    // Wait semaphore 2
    clEventWrapper wait_2_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext_2.getCLSemaphore(),
                                     nullptr, 0, nullptr, &wait_2_event);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_1_event);
    test_assert_event_complete(wait_2_event);

    return TEST_PASS;
}

// Confirm that we can wait for multiple semaphores with one command
int test_external_semaphores_multi_wait(cl_device_id deviceID,
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

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore1(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkVk2CLSemaphore2(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext_1(vkVk2CLSemaphore1, context,
                                   vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema_ext_2(vkVk2CLSemaphore2, context,
                                   vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Signal semaphore 1
    clEventWrapper signal_1_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext_1.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_1_event);
    test_error(err, "Could not signal semaphore");

    // Signal semaphore 2
    clEventWrapper signal_2_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext_2.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_2_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore 1 and 2
    clEventWrapper wait_event;
    cl_semaphore_khr sema_list[] = { sema_ext_1.getCLSemaphore(),
                                     sema_ext_2.getCLSemaphore() };
    err = clEnqueueWaitSemaphoresKHR(queue, 2, sema_list, nullptr, 0, nullptr,
                                     &wait_event);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_1_event);
    test_assert_event_complete(signal_2_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that it is possible to enqueue a signal of wait and signal in any
// order as soon as the submission order (after deferred dependencies) is
// correct. Case: first one deferred wait, then one non deferred signal.
int test_external_semaphores_order_1(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Create user event
    clEventWrapper user_event = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    // Wait semaphore (dependency on user_event)
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 1, &user_event, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Signal semaphore
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 0, nullptr, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Flush and delay
    err = clFlush(queue);
    test_error(err, "Could not flush queue");
    std::this_thread::sleep_for(std::chrono::seconds(FLUSH_DELAY_S));

    // Ensure signal event is completed while wait event is not
    test_assert_event_complete(signal_event);
    test_assert_event_inprogress(wait_event);

    // Complete user_event
    err = clSetUserEventStatus(user_event, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that it is possible to enqueue a signal of wait and signal in any
// order as soon as the submission order (after deferred dependencies) is
// correct. Case: first two deferred signals, then one deferred wait. Unblock
// signal, then unblock wait. When wait completes, unblock the other signal.
int test_external_semaphores_order_2(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Create user events
    clEventWrapper user_event_1 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    clEventWrapper user_event_2 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    clEventWrapper user_event_3 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    // Signal semaphore (dependency on user_event_1)
    clEventWrapper signal_1_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 1, &user_event_1,
                                       &signal_1_event);
    test_error(err, "Could not signal semaphore");

    // Signal semaphore (dependency on user_event_2)
    clEventWrapper signal_2_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 1, &user_event_2,
                                       &signal_2_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore (dependency on user_event_3)
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 1, &user_event_3, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Complete user_event_1
    err = clSetUserEventStatus(user_event_1, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Complete user_event_3
    err = clSetUserEventStatus(user_event_3, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Flush and delay
    err = clFlush(queue);
    test_error(err, "Could not flush queue");
    std::this_thread::sleep_for(std::chrono::seconds(FLUSH_DELAY_S));

    // Ensure all events are completed except for second signal
    test_assert_event_complete(signal_1_event);
    test_assert_event_inprogress(signal_2_event);
    test_assert_event_complete(wait_event);

    // Complete user_event_2
    err = clSetUserEventStatus(user_event_2, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_1_event);
    test_assert_event_complete(signal_2_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that it is possible to enqueue a signal of wait and signal in any
// order as soon as the submission order (after deferred dependencies) is
// correct. Case: first two deferred signals, then two deferred waits. Unblock
// one signal and one wait (both blocked by the same user event). When wait
// completes, unblock the other signal. Then unblock the other wait.
int test_external_semaphores_order_3(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_external_semaphore"))
    {
        log_info("cl_khr_semaphore is not supported on this platoform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext(vkVk2CLSemaphore, context,
                                 vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Create user events
    clEventWrapper user_event_1 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    clEventWrapper user_event_2 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    clEventWrapper user_event_3 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    // Signal semaphore (dependency on user_event_1)
    clEventWrapper signal_1_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 1, &user_event_1,
                                       &signal_1_event);
    test_error(err, "Could not signal semaphore");

    // Signal semaphore (dependency on user_event_2)
    clEventWrapper signal_2_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                       nullptr, 1, &user_event_2,
                                       &signal_2_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore (dependency on user_event_3)
    clEventWrapper wait_1_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 1, &user_event_3, &wait_1_event);
    test_error(err, "Could not wait semaphore");

    // Wait semaphore (dependency on user_event_2)
    clEventWrapper wait_2_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_ext.getCLSemaphore(),
                                     nullptr, 1, &user_event_2, &wait_2_event);
    test_error(err, "Could not wait semaphore");

    // Complete user_event_2
    err = clSetUserEventStatus(user_event_2, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Flush and delay
    err = clFlush(queue);
    test_error(err, "Could not flush queue");
    std::this_thread::sleep_for(std::chrono::seconds(FLUSH_DELAY_S));

    // Ensure only second signal and second wait completed
    cl_event event_list[] = { signal_2_event, wait_2_event };
    err = clWaitForEvents(2, event_list);
    test_error(err, "Could not wait for events");

    test_assert_event_inprogress(signal_1_event);
    test_assert_event_inprogress(wait_1_event);

    // Complete user_event_1
    err = clSetUserEventStatus(user_event_1, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Complete user_event_3
    err = clSetUserEventStatus(user_event_3, CL_COMPLETE);
    test_error(err, "Could not set user event to CL_COMPLETE");

    // Finish
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    // Ensure all events are completed
    test_assert_event_complete(signal_1_event);
    test_assert_event_complete(signal_2_event);
    test_assert_event_complete(wait_1_event);
    test_assert_event_complete(wait_2_event);

    return TEST_PASS;
}

// Test that an invalid semaphore command results in the invalidation of the
// command's event and the dependencies' events
int test_external_semaphores_invalid_command(cl_device_id deviceID,
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

    if (init_vuikan_device())
    {
        log_info("Cannot initialise Vulkan. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    VulkanDevice vkDevice;

    // Obtain pointers to semaphore's API
    GET_PFN(deviceID, clEnqueueSignalSemaphoresKHR);
    GET_PFN(deviceID, clEnqueueWaitSemaphoresKHR);

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore1(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkVk2CLSemaphore2(vkDevice, vkExternalSemaphoreHandleType);

    clExternalSemaphore sema_ext_1(vkVk2CLSemaphore1, context,
                                   vkExternalSemaphoreHandleType, deviceID);
    clExternalSemaphore sema_ext_2(vkVk2CLSemaphore2, context,
                                   vkExternalSemaphoreHandleType, deviceID);

    cl_int err = CL_SUCCESS;

    // Create ooo queue
    clCommandQueueWrapper queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    // Create user events
    clEventWrapper user_event_1 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    clEventWrapper user_event_2 = clCreateUserEvent(context, &err);
    test_error(err, "Could not create user event");

    // Signal semaphore_1 (dependency on user_event_1)
    clEventWrapper signal_1_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext_1.getCLSemaphore(),
                                       nullptr, 1, &user_event_1,
                                       &signal_1_event);
    test_error(err, "Could not signal semaphore");

    // Wait semaphore_1 and semaphore_2 (dependency on user_event_1)
    clEventWrapper wait_event;
    cl_semaphore_khr sema_list[] = { sema_ext_1.getCLSemaphore(),
                                     sema_ext_2.getCLSemaphore() };
    err = clEnqueueWaitSemaphoresKHR(queue, 2, sema_list, nullptr, 1,
                                     &user_event_1, &wait_event);
    test_error(err, "Could not wait semaphore");

    // Signal semaphore_1 (dependency on wait_event and user_event_2)
    clEventWrapper signal_2_event;
    cl_event wait_list[] = { user_event_2, wait_event };
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_ext_1.getCLSemaphore(),
                                       nullptr, 2, wait_list, &signal_2_event);
    test_error(err, "Could not signal semaphore");

    // Flush and delay
    err = clFlush(queue);
    test_error(err, "Could not flush queue");
    std::this_thread::sleep_for(std::chrono::seconds(FLUSH_DELAY_S));

    // Ensure all events are not completed
    test_assert_event_inprogress(signal_1_event);
    test_assert_event_inprogress(signal_2_event);
    test_assert_event_inprogress(wait_event);

    // Complete user_event_1 (expect failure as waiting on semaphore_2 is not
    // allowed (unsignaled)
    err = clSetUserEventStatus(user_event_1, CL_COMPLETE);
    test_assert_error(err != CL_SUCCESS,
                      "signal_2_event completed unexpectedly");

    // Ensure signal_1 is completed while others failed (the second signal
    // should fail as it depends on wait)
    err = clFinish(queue);
    test_error(err, "Could not finish queue");

    test_assert_event_complete(signal_1_event);
    test_assert_event_terminated(wait_event);
    test_assert_event_terminated(signal_2_event);

    return TEST_PASS;
}
