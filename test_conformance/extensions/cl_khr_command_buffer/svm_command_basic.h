
#ifndef CL_KHR_SVM_COMMAND_BASIC_H
#define CL_KHR_SVM_COMMAND_BASIC_H

#include "basic_command_buffer.h"
#include "command_buffer_test_base.h"


struct BasicSVMCommandBufferTest : BasicCommandBufferTest
{
    BasicSVMCommandBufferTest(cl_device_id device, cl_context context,
                              cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    virtual bool Skip() override;
    virtual cl_int SetUpKernelArgs(void) override;

protected:
    cl_int init_extension_functions();

    clCommandSVMMemfillKHR_fn clCommandSVMMemfillKHR = nullptr;
    clCommandSVMMemcpyKHR_fn clCommandSVMMemcpyKHR = nullptr;

    clSVMWrapper svm_in_mem, svm_out_mem;
};

#endif
