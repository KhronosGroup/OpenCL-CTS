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
#ifndef __KERNELARGS_H
#define __KERNELARGS_H


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <assert.h>

#include <string>
#include <vector>
#include <iostream>

#include "harness/typeWrappers.h"

#include "exceptions.h"

class WorkSizeInfo;

/**
 Represents the single kernel argument information
 */
class KernelArgInfo
{
public:
    cl_kernel_arg_address_qualifier getAddressQualifier() const { return m_address_qualifier; }
    cl_kernel_arg_access_qualifier  getAccessQualifier() const { return m_access_qualifier; }
    cl_kernel_arg_type_qualifier    getTypeQualifier() const { return m_type_qualifier; }

    cl_kernel_arg_address_qualifier* getAddressQualifierRef() { return &m_address_qualifier; }
    cl_kernel_arg_access_qualifier*  getAccessQualifierRef() { return &m_access_qualifier; }
    cl_kernel_arg_type_qualifier*    getTypeQualifierRef() { return &m_type_qualifier; }

    void setTypeName( const char* name) { m_type.assign(name); }
    void setName( const char* name) { m_name.assign(name); }

    std::string getTypeName() const { return m_type; }
    std::string getName() const { return m_name; }

    bool operator == ( const KernelArgInfo& rhs ) const
    {
        return !m_name.compare(rhs.m_name) &&
            !m_type.compare(rhs.m_type) &&
            m_address_qualifier == rhs.m_address_qualifier &&
            m_access_qualifier == rhs.m_access_qualifier &&
            m_type_qualifier == rhs.m_type_qualifier;
    }

    bool operator != ( const KernelArgInfo& rhs ) const
    {
        return !(*this == rhs);
    }

private:
    std::string m_name;
    std::string m_type;
    cl_kernel_arg_address_qualifier m_address_qualifier;
    cl_kernel_arg_access_qualifier  m_access_qualifier;
    cl_kernel_arg_type_qualifier    m_type_qualifier;
};

/**
 Represents the single kernel's argument value.
 Responsible for livekeeping of OCL objects.
 */
class KernelArg
{
public:
    KernelArg(const KernelArgInfo& argInfo, void* buffer, size_t size):
      m_argInfo(argInfo),
      m_buffer(buffer),
      m_size(size)
    {}

    virtual ~KernelArg()
    {
        align_free(m_buffer);
    }

    virtual size_t getArgSize() const
    {
        return m_size;
    }

    virtual const void* getBuffer() const
    {
        return m_buffer;
    }

    virtual const void* getArgValue() const
    {
        return m_buffer;
    }

    virtual bool compare( const KernelArg& rhs, float ulps ) const
    {
        if( m_argInfo != rhs.m_argInfo )
        {
            return false;
        }

        if( m_size != rhs.m_size)
        {
            return false;
        }

        if( (NULL == m_buffer || NULL == rhs.m_buffer) && m_buffer != rhs.m_buffer )
        {
            return false;
        }

        //check two NULL buffers case
        if( NULL == m_buffer && NULL == rhs.m_buffer )
        {
            return true;
        }

        bool match = true;
        if( memcmp( m_buffer, rhs.m_buffer, m_size) )
        {
            std::string typeName = m_argInfo.getTypeName();
            size_t compared = 0;
            if (typeName.compare("float*") == 0)
            {
                while (compared < m_size)
                {
                    float l = *(float*)(((char*)m_buffer)+compared);
                    float r = *(float*)(((char*)rhs.m_buffer)+compared);
                    if (fabsf(Ulp_Error(l, r)) > ulps)
                    {
                        match = false;
                        break;
                    }
                    compared += sizeof(float);
                }
            }
            else if (typeName.compare("double*") == 0)
            {
                while (compared < m_size)
                {
                    double l = *(double*)(((char*)m_buffer)+compared);
                    double r = *(double*)(((char*)rhs.m_buffer)+compared);
                    if (fabsf(Ulp_Error_Double(l, r)) > ulps)
                    {
                        match = false;
                        break;
                    }
                    compared += sizeof(double);
                }
            }
            else
            {
                while (compared < m_size)
                {
                    if ( *(((char*)m_buffer)+compared) != *(((char*)rhs.m_buffer)+compared) )
                    {
                        match = false;
                        break;
                    }
                    compared++;
                }
            }
            if (!match)
            {
                std::cerr << std::endl << " difference is at offset " << compared << std::endl;
            }
        }

        return match;
    }

    virtual void readToHost(cl_command_queue queue)
    {
        return;
    }

    KernelArg* clone(cl_context context, const WorkSizeInfo& ws, const cl_kernel kernel, const cl_device_id device) const;

protected:
    KernelArgInfo m_argInfo;
    void*  m_buffer;
    size_t m_size;
};

class KernelArgSampler:public KernelArg
{
public:
    KernelArgSampler(cl_context context, cl_bool isNormalized,
                     cl_addressing_mode addressMode, cl_filter_mode filterMode):
    KernelArg(KernelArgInfo(), NULL, sizeof(cl_sampler))
    {
        m_argInfo.setTypeName("sampler_t");
        int error = CL_SUCCESS;
        m_samplerObj = clCreateSampler(context, isNormalized, addressMode,
                                       filterMode, &error);
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("clCreateSampler failed\n", error);
        }
    }

    ~KernelArgSampler()
    {
        //~clSamplerWrapper() releases the sampler object
    }

    size_t getArgSize() const
    {
        return sizeof(cl_sampler);
    }

    const void* getArgValue() const
    {
        return &m_samplerObj;
    }

    bool compare(const KernelArg& rhs, float) const
    {
        if (const KernelArgSampler *Rhs = dynamic_cast<const KernelArgSampler*>(&rhs))
        {
            return isNormalized() == Rhs->isNormalized() &&
                   getAddressingMode() == Rhs->getAddressingMode() &&
                   getFilterMode() == Rhs->getFilterMode();
        }
        return false;
    }

    cl_sampler getSampler() const
    {
      return (cl_sampler)m_samplerObj;
    }

protected:
    mutable clSamplerWrapper m_samplerObj;

    cl_bool isNormalized() const
    {
        cl_bool norm;
        cl_int err = clGetSamplerInfo(getSampler(),
                                      CL_SAMPLER_NORMALIZED_COORDS,
                                      sizeof(cl_bool),
                                      &norm,
                                      NULL);
        if (CL_SUCCESS != err)
            throw Exceptions::TestError("clGetSamplerInfo failed\n", err);
        return norm;
    }

    cl_addressing_mode getAddressingMode() const
    {
        cl_addressing_mode addressingmode;
        cl_int err = clGetSamplerInfo(getSampler(),
                                      CL_SAMPLER_ADDRESSING_MODE,
                                      sizeof(cl_addressing_mode),
                                      &addressingmode,
                                      NULL);
        if (CL_SUCCESS != err)
            throw Exceptions::TestError("clGetSamplerInfo failed\n", err);
        return addressingmode;
    }

    cl_filter_mode getFilterMode() const
    {
        cl_filter_mode filtermode;
        cl_int err = clGetSamplerInfo(getSampler(),
                                      CL_SAMPLER_FILTER_MODE,
                                      sizeof(cl_filter_mode),
                                      &filtermode,
                                      NULL);
        if (CL_SUCCESS != err)
            throw Exceptions::TestError("clGetSamplerInfo failed\n", err);
        return filtermode;
    }

};


class KernelArgMemObj:public KernelArg
{
public:
    KernelArgMemObj(const KernelArgInfo& argInfo, void* buffer, size_t size):
      KernelArg(argInfo, buffer, size)
    {
        m_memObj = NULL;
    }

    ~KernelArgMemObj()
    {
        //~clMemWrapper() releases the memory object
    }

    virtual void readToHost(cl_command_queue queue)  = 0;


    size_t getArgSize() const
    {
        if( NULL == m_buffer )
            return m_size;              // local buffer
        else
            return sizeof(cl_mem);
    }

    const void* getArgValue() const
    {
        if( NULL == m_buffer )
        {
            return NULL;                // local buffer
        }
        else {
            clMemWrapper* p = const_cast<clMemWrapper*>(&m_memObj);

            return (const void*)(&(*p));
        }
    }

protected:
    clMemWrapper m_memObj;
};

/**
 Represents the single kernel's argument value.
 Responsible for livekeeping of OCL objects.
 */
class KernelArgBuffer:public KernelArgMemObj
{
public:
    KernelArgBuffer(cl_context context, const KernelArgInfo& argInfo, void* buffer, size_t size):
        KernelArgMemObj(argInfo, buffer, size)
    {
        if( NULL != buffer )
        {
            int error = CL_SUCCESS;
            m_memObj = clCreateBuffer(context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      size, buffer, &error);
            if( error != CL_SUCCESS )
            {
                throw Exceptions::TestError("clCreateBuffer failed\n", error);
            }
        }
    }

    void readToHost(cl_command_queue queue)
    {
        if( NULL == m_buffer )
        {
            return;
        }

        int error = clEnqueueReadBuffer( queue, m_memObj, CL_TRUE, 0, m_size, m_buffer, 0, NULL, NULL);
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("clEnqueueReadBuffer failed\n", error);
        }
    }
};

class KernelArgImage:public KernelArgMemObj
{
public:
    KernelArgImage(cl_context context, const KernelArgInfo& argInfo,
                   void* buffer, size_t size, cl_mem_flags flags,
                   cl_image_format format, cl_image_desc desc):
    KernelArgMemObj(argInfo, buffer, size), m_desc(desc)
    {
        if( NULL != buffer )
        {
            int error = CL_SUCCESS;
            flags |= CL_MEM_COPY_HOST_PTR ;
            if (CL_MEM_OBJECT_IMAGE1D_BUFFER == m_desc.image_type)
            {
                m_desc.buffer = clCreateBuffer( context, flags, m_desc.image_row_pitch, buffer, &error );
                if( error != CL_SUCCESS )
                {
                    throw Exceptions::TestError("KernelArgImage clCreateBuffer failed\n", error);
                }
                buffer = NULL;
                flags &= ~CL_MEM_COPY_HOST_PTR;
                m_desc.image_row_pitch = 0;
                m_desc.image_slice_pitch = 0;
            }
            m_memObj = clCreateImage( context, flags, &format, &m_desc, buffer, &error );
            if( error != CL_SUCCESS )
            {
                throw Exceptions::TestError("KernelArgImage clCreateImage failed\n", error);
            }
        }
    }

    ~KernelArgImage()
    {
        if (CL_MEM_OBJECT_IMAGE1D_BUFFER == m_desc.image_type)
        {
             clReleaseMemObject(m_desc.buffer);
        }
    }

    void readToHost(cl_command_queue queue)
    {
        if( NULL == m_buffer )
        {
            return;
        }

        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {m_desc.image_width , m_desc.image_height , m_desc.image_depth};

        int error = clEnqueueReadImage (queue, m_memObj, CL_TRUE, origin, region, m_desc.image_row_pitch, m_desc.image_slice_pitch, m_buffer, 0, NULL, NULL);

        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("clEnqueueReadImage failed\n", error);
        }
    }

private:
    cl_image_desc m_desc;
};

/**
 Represents the container for the kernel parameters
 */
class KernelArgs
{
    typedef std::vector<KernelArg*> KernelsArgsVector;
public:
    KernelArgs(){}
    ~KernelArgs()
    {
        KernelsArgsVector::iterator i = m_args.begin();
        KernelsArgsVector::iterator e = m_args.end();

        for( ; i != e; ++i )
        {
            assert( NULL != *i );
            delete *i;
        }
    }

    void readToHost(cl_command_queue queue)
    {
        KernelsArgsVector::iterator i = m_args.begin();
        KernelsArgsVector::iterator e = m_args.end();

        for( ; i != e; ++i )
        {
            (*i)->readToHost(queue);
        }
    }

    size_t getArgCount() const { return m_args.size(); }

    KernelArg* getArg(size_t index ) { return m_args[index]; }

    const KernelArg* getArg(size_t index) const { return m_args[index]; }

    void addArg( KernelArg* arg ) { m_args.push_back(arg); }

private:
    KernelsArgsVector m_args;
};

#endif
