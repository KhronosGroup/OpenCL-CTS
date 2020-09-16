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
#ifndef __DATAGEN_H
#define __DATAGEN_H

#include "harness/compat.h"

#include <assert.h>

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>

#include "harness/mt19937.h"

#include "exceptions.h"
#include "kernelargs.h"

// ESINNS is a short name for EXPLICIT_SPECIALIZATION_IN_NON_NAMESPACE_SCOPE

#undef ESINNS

#ifdef __GNUC__

#define ESINNS
#define ESINNS_PREF() inline
#define ESINNS_POST() RandomGenerator::

#else

#define ESINNS_PREF()
#define ESINNS_POST()

#endif

#define MAX_WORK_DIM        3
#define GLOBAL_WORK_SIZE    (((CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE / sizeof(cl_double) / 16) / 2) * 2)            // max buffer size = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE / sizeof(double16)

// SPIR definitions for image channel data types (Section 2.1.3.2).
#define SPIR_CLK_SNORM_INT8         0x10D0
#define SPIR_CLK_SNORM_INT16        0x10D1
#define SPIR_CLK_UNORM_INT8         0x10D2
#define SPIR_CLK_UNORM_INT16        0x10D3
#define SPIR_CLK_UNORM_SHORT_565    0x10D4
#define SPIR_CLK_UNORM_SHORT_555    0x10D5
#define SPIR_CLK_UNORM_SHORT_101010 0x10D6
#define SPIR_CLK_SIGNED_INT8        0x10D7
#define SPIR_CLK_SIGNED_INT16       0x10D8
#define SPIR_CLK_SIGNED_INT32       0x10D9
#define SPIR_CLK_UNSIGNED_INT8      0x10DA
#define SPIR_CLK_UNSIGNED_INT16     0x10DB
#define SPIR_CLK_UNSIGNED_INT32     0x10DC
#define SPIR_CLK_HALF_FLOAT         0x10DD
#define SPIR_CLK_FLOAT              0x10DE
#define SPIR_CLK_UNORM_INT24        0x10DF

#define NUM_IMG_FORMATS 64

double get_random_double(double low, double high, MTdata d);
float get_random_float(float low, float high, MTdata d);
size_t get_random_size_t(size_t low, size_t high, MTdata d);

/**
 Simple container for the work size information
 */
class WorkSizeInfo
{
public:
    /**
      Returns the flat global size
      */
    size_t getGlobalWorkSize() const;
public:
    cl_uint work_dim;
    size_t  global_work_offset[MAX_WORK_DIM];
    size_t  global_work_size[MAX_WORK_DIM];
    size_t  local_work_size[MAX_WORK_DIM];
};

/**
 Generates various types of random numbers
 */
class RandomGenerator
{
public:
    RandomGenerator():m_d(NULL)
    {
       init(0);
    }

    ~RandomGenerator()
    {
        if( NULL != m_d )
            free_mtdata(m_d);
    }

    void init(cl_uint seed)
    {
        m_d = init_genrand( seed );
    }

    template<class T> T getNext(T low, T high)
    {
        assert(false && "Not implemented");
        return T();
    }

#ifdef ESINNS

private:
    MTdata m_d;
};

#endif

template<> ESINNS_PREF() bool ESINNS_POST()getNext(bool low, bool high)
{
    return (bool)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_char ESINNS_POST()getNext(cl_char low, cl_char high)
{
    return (cl_char)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_double ESINNS_POST()getNext(cl_double low, cl_double high)
{
    return get_random_double(low, high, m_d);
}

template<> ESINNS_PREF() cl_float ESINNS_POST()getNext(cl_float low, cl_float high)
{
    return get_random_float(low, high, m_d);
}

template<> ESINNS_PREF() cl_int ESINNS_POST()getNext(cl_int low, cl_int high)
{
    return (cl_int)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_long ESINNS_POST()getNext(cl_long low, cl_long high)
{
    return (cl_long)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_short ESINNS_POST()getNext(cl_short low, cl_short high)
{
    return (cl_short)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_uchar ESINNS_POST()getNext(cl_uchar low, cl_uchar high)
{
    return (cl_uchar)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_uint ESINNS_POST()getNext(cl_uint low, cl_uint high)
{
    return (cl_uint)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_ulong ESINNS_POST()getNext(cl_ulong low, cl_ulong high)
{
    return (cl_ulong)get_random_size_t((size_t)low, (size_t)high, m_d);
}

template<> ESINNS_PREF() cl_ushort ESINNS_POST()getNext(cl_ushort low, cl_ushort high)
{
    return (cl_ushort)get_random_size_t((size_t)low, (size_t)high, m_d);
}

#ifndef ESINNS

private:
    MTdata m_d;
};

#endif

extern RandomGenerator gRG;

/**
 Base class for kernel argument generator
 */
class KernelArgGenerator
{
protected:
    KernelArgGenerator()
    {}

public:
    virtual KernelArg* generate( cl_context context,
                                 const WorkSizeInfo& ws,
                                 const KernelArgInfo& argInfo,
                                 const KernelArg* refArg,
                                 const cl_kernel kernel,
                                 const cl_device_id device ) = 0;
    virtual ~KernelArgGenerator() {}
};

/**
 Mock: 'Not implemented' version of the kernel argument generator - used for the still unsupported types
 */
class KernelArgGeneratorNI: public KernelArgGenerator
{
public:
    KernelArgGeneratorNI( bool isBuffer, size_t vectorSize, int minValue, int maxValue )
    {}

    KernelArg* generate( cl_context context,
                         const WorkSizeInfo& ws,
                         const KernelArgInfo& argInfo,
                         const KernelArg* refArg,
                         const cl_kernel kernel,
                         const cl_device_id device )
    {
        //assert(false && "Not implemented");
        throw Exceptions::TestError("KernelArgGenerator is not implemented\n");
    }
};

/**
 Kernel argument generator for images
 */
class KernelArgGeneratorImage: public KernelArgGenerator
{
public:
    KernelArgGeneratorImage(bool isBuffer, size_t vectorSize, char minValue, char maxValue) :
        m_isBuffer(isBuffer),
        m_vectorSize(vectorSize),
        m_minValue(minValue),
        m_maxValue(maxValue)
    {
        m_format.image_channel_order = CL_RGBA;

        m_desc.image_width = 32;
        m_desc.image_height = 1;
        m_desc.image_depth = 1;
        m_desc.image_array_size = 1;
        m_desc.num_mip_levels = 0;
        m_desc.num_samples = 0;
        m_desc.buffer = NULL;
    }

    bool isValidChannelOrder(cl_context context, cl_channel_order order) const
    {
        cl_mem_flags flags = CL_MEM_COPY_HOST_PTR;
        cl_uint actualNumFormats = 0;
        cl_image_format imgFormat = m_format;
        imgFormat.image_channel_order = order;

        cl_int error = clGetSupportedImageFormats(
            context,
            flags,
            m_desc.image_type,
            0,
            NULL,
            &actualNumFormats);
        if (CL_SUCCESS != error)
            throw Exceptions::TestError("clGetSupportedImageFormats failed\n", error);

        std::vector<cl_image_format> supportedFormats(actualNumFormats);
        error = clGetSupportedImageFormats(context, flags, m_desc.image_type,
                                           actualNumFormats,
                                           supportedFormats.data(), NULL);
        if (CL_SUCCESS != error)
            throw Exceptions::TestError("clGetSupportedImageFormats failed\n", error);

        for (size_t i=0; i<actualNumFormats; ++i)
        {
            cl_image_format curFormat = supportedFormats[i];

            if(imgFormat.image_channel_order == curFormat.image_channel_order &&
               imgFormat.image_channel_data_type == curFormat.image_channel_data_type)
               return true;
        }

        return false;
    }

    void setChannelOrder(cl_channel_order order)
    {
        m_format.image_channel_order = order;
    }

    KernelArg* generate(cl_context context,
                        const WorkSizeInfo& ws,
                        const KernelArgInfo& argInfo,
                        const KernelArg* refArg,
                        const cl_kernel kernel,
                        const cl_device_id device)
    {
        void * pBuffer = NULL;
        size_t numPixels = m_desc.image_width * m_desc.image_height * m_desc.image_depth * m_desc.image_array_size;
        const int alignment = sizeof(cl_int) * 4 ; //RGBA channel size * sizeof (cl_int)
        size_t allocSize = numPixels * alignment ;

        cl_kernel_arg_access_qualifier accessQ = argInfo.getAccessQualifier();

        cl_mem_flags mem_flags = 0;

        if (accessQ == CL_KERNEL_ARG_ACCESS_READ_ONLY)
        {
            mem_flags |=  CL_MEM_READ_ONLY;
        }

        if (accessQ == CL_KERNEL_ARG_ACCESS_WRITE_ONLY)
        {
            mem_flags |=  CL_MEM_WRITE_ONLY;
        }

        if (accessQ == CL_KERNEL_ARG_ACCESS_READ_WRITE)
        {
            mem_flags |=  CL_MEM_READ_WRITE;
        }

        pBuffer = align_malloc(allocSize, alignment);
        if (NULL == pBuffer)
        {
            throw Exceptions::TestError("align_malloc failed for image\n", 1);
        }
        assert( (size_t)pBuffer % alignment == 0 );
        if (NULL == refArg)
        {
            fillBuffer((cl_char *)pBuffer, allocSize );
        }
        else {
            memcpy(pBuffer, refArg->getBuffer(), allocSize );
        }

        return new KernelArgImage(context, argInfo, pBuffer, allocSize, mem_flags, m_format, m_desc);
    }

protected:
    KernelArgGeneratorImage()
    {}

    void fillBuffer( cl_char * ptr, size_t nelem)
    {
        for( size_t i = 0; i < nelem; ++i )
        {
            ptr[i]  = gRG.getNext<cl_char>(m_minValue, m_maxValue);
        }
    }

protected:
    bool m_isBuffer;
    size_t m_vectorSize;
    cl_char m_minValue;
    cl_char m_maxValue;
    cl_image_format m_format;
    cl_image_desc m_desc;
};

/**
 Kernel argument generator for image1d_array
 */
template<cl_channel_type channel_type> class KernelArgGeneratorImage1dArray: public KernelArgGeneratorImage
{
public:
    KernelArgGeneratorImage1dArray( bool isBuffer, size_t vectorSize, char minValue, char maxValue ):
        KernelArgGeneratorImage(isBuffer, vectorSize, minValue, maxValue)
    {
        m_desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
        m_format.image_channel_data_type = channel_type;

        m_desc.image_row_pitch = m_desc.image_width*4*4;                        //RGBA channel size * sizeof (cl_int)
        m_desc.image_slice_pitch = m_desc.image_height * m_desc.image_row_pitch;

    }
};

/**
 Kernel argument generator for image1d_buffer
 */
template<cl_channel_type channel_type> class KernelArgGeneratorImage1dBuffer: public KernelArgGeneratorImage
{
public:
    KernelArgGeneratorImage1dBuffer( bool isBuffer, size_t vectorSize, char minValue, char maxValue ) :
        KernelArgGeneratorImage(isBuffer, vectorSize, minValue, maxValue)
    {
        m_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        m_format.image_channel_data_type = channel_type;

        m_desc.image_row_pitch = m_desc.image_width*4*4; //RGBA channel size * sizeof (cl_int)
        // http://www.khronos.org/registry/cl/specs/opencl-1.2.pdf 5.2.2;
        // Slice pitch of 1d images should be zero.
        m_desc.image_slice_pitch = 0;
    }
};

/**
 Kernel argument generator for image1d
 */
template<cl_channel_type channel_type> class KernelArgGeneratorImage1d: public KernelArgGeneratorImage
{
public:
    KernelArgGeneratorImage1d( bool isBuffer, size_t vectorSize, char minValue, char maxValue ) :
        KernelArgGeneratorImage(isBuffer, vectorSize, minValue, maxValue)
    {
        m_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        m_format.image_channel_data_type = channel_type;

        m_desc.image_row_pitch = m_desc.image_width*4*4;                        //RGBA channel size * sizeof (cl_int)
        // http://www.khronos.org/registry/cl/specs/opencl-1.2.pdf
        // '5.3.1.2 image descriptor': Slice pitch is not applicable for one-
        // dimensional images.
        m_desc.image_slice_pitch = 0;
    }
};

/**
 Kernel argument generator for image2d_array
 */
template<cl_channel_type channel_type> class KernelArgGeneratorImage2dArray: public KernelArgGeneratorImage
{
public:
    KernelArgGeneratorImage2dArray( bool isBuffer, size_t vectorSize, char minValue, char maxValue ) :
        KernelArgGeneratorImage(isBuffer, vectorSize, minValue, maxValue)
    {
        m_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        m_format.image_channel_data_type = channel_type;

        m_desc.image_height = 32;
        m_desc.image_array_size = 8;
        m_desc.image_row_pitch = m_desc.image_width*4*4;                        //RGBA channel size * sizeof (cl_int)
        m_desc.image_slice_pitch = m_desc.image_height * m_desc.image_row_pitch;
    }
};

/**
 Kernel argument generator for image2d
 */
template<cl_channel_type channel_type> class KernelArgGeneratorImage2d: public KernelArgGeneratorImage
{
public:
    KernelArgGeneratorImage2d( bool isBuffer, size_t vectorSize, char minValue, char maxValue ) :
        KernelArgGeneratorImage(isBuffer, vectorSize, minValue, maxValue)
    {
        m_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        m_format.image_channel_data_type = channel_type;

        m_desc.image_height = 32;
        m_desc.image_row_pitch = m_desc.image_width*4*4;                        //RGBA channel size * sizeof (cl_int)
        // http://www.khronos.org/registry/cl/specs/opencl-1.2.pdf
        // '5.3.1.2 image descriptor': Slice pitch is not applicable for two-
        // dimensional images.
        m_desc.image_slice_pitch = 0;
    }
};

/**
 Kernel argument generator for image3d
 */
template<cl_channel_type channel_type> class KernelArgGeneratorImage3d: public KernelArgGeneratorImage
{
public:
    KernelArgGeneratorImage3d( bool isBuffer, size_t vectorSize, char minValue, char maxValue ) :
        KernelArgGeneratorImage(isBuffer, vectorSize, minValue, maxValue)
    {
        m_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
        m_format.image_channel_data_type = channel_type;

        m_desc.image_height = 32;
        m_desc.image_depth = 8;
        m_desc.image_row_pitch = m_desc.image_width*4*4;                        //RGBA channel size * sizeof (cl_int)
        m_desc.image_slice_pitch = m_desc.image_height * m_desc.image_row_pitch;
    }
};

/**
 Kernel argument generator for samplers
 */
class KernelArgGeneratorSampler: public KernelArgGenerator
{
public:
    KernelArgGeneratorSampler(bool isBuffer, size_t vectorSize, int minValue, int maxValue);

    KernelArgGeneratorSampler();

    /*
     * Sampler property setters.
     */
    void setNormalized(cl_bool);
    void setAddressingMode(cl_addressing_mode);
    void setFiterMode(cl_filter_mode);

    KernelArg* generate(cl_context context,
                        const WorkSizeInfo& ws,
                        const KernelArgInfo& argInfo,
                        const KernelArg* refArg,
                        const cl_kernel kernel,
                        const cl_device_id device)
    {
        return new KernelArgSampler(context, m_normalized, m_addressingMode, m_filterMode);
    }
private:
    void initToDefaults();

    cl_bool m_normalized;
    cl_addressing_mode m_addressingMode;
    cl_filter_mode m_filterMode;
};

/*
 * Generates all the possible values for image samplers.
 */
class SamplerValuesGenerator
{
public:
    class iterator {
        friend class SamplerValuesGenerator;

        size_t m_normIndex, m_filterIndex, m_addressingModeIndex;

        iterator(size_t norm, size_t filter, size_t addressing);

        bool incrementIndex(size_t &i, const size_t limit);
    public:
        iterator();

        /*
         * Moves the iterator to the next sampler value.
         */
        iterator& operator ++();

        bool operator == (const iterator& other) const;

        bool operator != (const iterator& other) const;

        cl_bool getNormalized() const;

        cl_filter_mode getFilterMode() const;

        cl_addressing_mode getAddressingMode() const;

        /*
         * Converts the value of the sampler to a bitmask representation.
         */
        unsigned toBitmap() const;

        /*
         * Retruns a string representation of the sampler.
         */
        std::string toString() const;
    };

    iterator begin() { return iterator(); }

    iterator end();

    static cl_bool coordNormalizations[];
    static cl_filter_mode filterModes[];
    static cl_addressing_mode addressingModes[];
};

typedef struct struct_type {
    cl_float float4d[4];
    cl_int intd;
} typedef_struct_type;

typedef struct {
    cl_int width;
    cl_int channelType;
    cl_int channelOrder;
    cl_int expectedChannelType;
    cl_int expectedChannelOrder;
 } image_kernel_data;

typedef struct testStruct {
     cl_double vec[16];
 } testStruct;

typedef struct {
     cl_uint workDim;
     cl_uint globalSize[3];
     cl_uint globalID[3];
     cl_uint localSize[3];
     cl_uint localID[3];
     cl_uint numGroups[3];
     cl_uint groupID[3];
  } work_item_data;

/**
 Kernel argument generator for structure "struct_type"

 Kernel argument generator for structure "image_kernel_data"

 Kernel argument generator for structure "testStruct"
 Since there are many "testStruct", we define it to have maximum space
 Also the alignment is done following the "worst" case

 Kernel argument generator for structure "work_item_data"
 */

  template<typename T> class KernelStructTypeArgGenerator: public KernelArgGenerator
  {

  public:
      KernelStructTypeArgGenerator( bool isBuffer, size_t vectorSize, cl_int minValue, cl_int maxValue ):
          m_isBuffer(isBuffer),
          m_vectorSize(vectorSize),
          m_alignment(0),
          m_size(0)
      {}

      KernelArg* generate( cl_context context,
                                 const WorkSizeInfo& ws,
                                 const KernelArgInfo& argInfo,
                                 const KernelArg* refArg,
                                 const cl_kernel kernel,
                                 const cl_device_id device )
      {
          T *pStruct = NULL;

          calcSizeAndAlignment(pStruct);
          size_t size = m_size;

          if( m_isBuffer )
          {
              cl_kernel_arg_address_qualifier addrQ = argInfo.getAddressQualifier();

              if( CL_KERNEL_ARG_ADDRESS_CONSTANT == addrQ )
              {
                  if ( (CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE/m_size)*m_size < m_size )
                      size=(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE/m_size)*m_size;
              }

              if( CL_KERNEL_ARG_ADDRESS_GLOBAL   == addrQ ||
                  CL_KERNEL_ARG_ADDRESS_CONSTANT == addrQ )
              {
                  size_t no_e = ws.getGlobalWorkSize();
                  size = no_e * m_size;
                  pStruct = (T *)align_malloc(size, m_alignment);
                  if (NULL == pStruct)
                  {
                       throwExceptions(pStruct);
                  }
                  assert( (size_t)pStruct % m_alignment == 0 );
                   if (NULL == refArg)
                  {
                      fillBuffer(pStruct, no_e);
                  }
                  else {
                      memcpy(pStruct, refArg->getBuffer(), size);
                  }
              }
              return new KernelArgBuffer( context, argInfo, (void*)pStruct, size);
          }
          else {
              pStruct = (T *)align_malloc(m_size, m_alignment);
              if (NULL == pStruct)
              {
                   throwExceptions(pStruct);
              }
              assert( (size_t)pStruct % m_alignment == 0 );
              if (NULL == refArg)
              {
                  fillBuffer(pStruct, 1);
              }
              else {
                  memcpy(pStruct, refArg->getBuffer(), m_size);
              }

              return new KernelArg( argInfo, (void*)pStruct, m_size);
          }
      }
  private:

      std::string getTypeString(typedef_struct_type *pStruct)
      {
          return "typedef_struct_type";
      }

      std::string getTypeString(image_kernel_data *pStruct)
      {
          return "image_kernel_data";
      }

      std::string getTypeString(testStruct *pStruct)
      {
          return "testStruct";
      }

      std::string getTypeString(work_item_data *pStruct)
      {
          return "work_item_data";
      }

      void throwExceptions(T * pStruct)
      {
          std::string str = "align_malloc failed for " ;
          if (m_isBuffer)
              str += "array of " ;
          str += getTypeString(pStruct) ;
          throw Exceptions::TestError(str, 1);
      }

      void fillBuffer( typedef_struct_type *pStruct, size_t no_e )
      {
          for (size_t e = 0; e < no_e; ++e)
          {
              for( size_t i = 0; i < 4; ++i )
              {
                  pStruct[e].float4d[i] = gRG.getNext<cl_float>(-0x01000000, 0x01000000);
              }
              pStruct[e].intd = gRG.getNext<cl_int>(0, 0x7fffffff);
          }
      }

      void fillBuffer( image_kernel_data *pStruct, size_t no_e )
      {
          for (size_t e = 0; e < no_e; ++e)
          {
              pStruct[e].width = gRG.getNext<cl_int>(0, 0x7fffffff);
              pStruct[e].channelType = gRG.getNext<cl_int>(0, 0x7fffffff);
              pStruct[e].channelOrder = gRG.getNext<cl_int>(0, 0x7fffffff);
              pStruct[e].expectedChannelType = gRG.getNext<cl_int>(0, 0x7fffffff);
              pStruct[e].expectedChannelOrder = gRG.getNext<cl_int>(0, 0x7fffffff);
          }
      }

      void fillBuffer( testStruct *pStruct, size_t no_e )
      {
          for (size_t e = 0; e < no_e; ++e)
          {
              for( size_t i = 0; i < 16; ++i )
              {
                  pStruct[e].vec[i] = gRG.getNext<cl_float>(-0x01000000, 0x01000000);
               }
          }
      }

      void fillBuffer( work_item_data *pStruct, size_t no_e )
      {
          for (size_t e = 0; e < no_e; ++e)
          {
              memset(&pStruct[e], 0, sizeof(work_item_data));
          }
      }

      // structure alignment is derived from the size of the larger field in it
      // size of the structure is the size of the largest field multiple by the number of fields

      void calcSizeAndAlignment(typedef_struct_type *pStruct)
      {
          m_alignment = sizeof(cl_float) * 4;
          m_size = m_alignment * 2 ;
      }

      void calcSizeAndAlignment(image_kernel_data *pStruct)
      {
          m_alignment = sizeof(cl_int);
          m_size = sizeof(image_kernel_data) ;
      }

      void calcSizeAndAlignment(testStruct *pStruct)
      {
          m_alignment = sizeof(cl_double) * 16;
          m_size = m_alignment;
      }

      void calcSizeAndAlignment(work_item_data *pStruct)
      {
          m_alignment = sizeof(cl_uint);
          m_size = sizeof(work_item_data);
      }

  private:
      bool m_isBuffer;
      size_t m_vectorSize;
      int m_alignment;
      size_t m_size;
};

/**
 Kernel argument generator for the simple scalar and vector types
 */
template <class T> class KernelArgGeneratorT: public KernelArgGenerator
{
public:
    KernelArgGeneratorT( bool isBuffer, size_t vectorSize, T minValue, T maxValue ):
        m_isBuffer(isBuffer),
        m_vectorSize(vectorSize),
        m_minValue(minValue),
        m_maxValue(maxValue)
    {}

    KernelArg* generate( cl_context context,
                         const WorkSizeInfo& ws,
                         const KernelArgInfo& argInfo,
                         const KernelArg* refArg,
                         const cl_kernel kernel,
                         const cl_device_id device  )
    {
        T* pBuffer = NULL;
        size_t size = 0;
        int alignment, error;
        cl_ulong totalDeviceLocalMem;
        cl_ulong localMemUsedByKernel;
        cl_uint numArgs, numLocalArgs = 0;
        KernelArgInfo kernel_arg_info;

        error = CL_SUCCESS;

        // take care of 3-elements vector's alignment issue:
        // if 3-elements vector - the alignment is 4-elements
        if (m_vectorSize == 3)
            alignment = sizeof(T) * 4;
        else
            alignment = sizeof(T) * m_vectorSize;

        // gather information about the kernel and device
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(totalDeviceLocalMem), &totalDeviceLocalMem, NULL);
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(localMemUsedByKernel), &localMemUsedByKernel, NULL);
        clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, NULL);

        // Calculate the number of local memory arguments
        for (cl_uint i = 0; i < numArgs; i ++)
        {
            error = clGetKernelArgInfo( kernel, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(cl_kernel_arg_address_qualifier), kernel_arg_info.getAddressQualifierRef(), &size);
            if( error != CL_SUCCESS )
            {
                throw Exceptions::TestError("Unable to get argument address qualifier\n", error);
            }

            if(kernel_arg_info.getAddressQualifier() == CL_KERNEL_ARG_ADDRESS_LOCAL)
            {
                numLocalArgs ++;
            }
        }

        // reduce the amount of local memory by the amount the kernel + implementation uses
        totalDeviceLocalMem -= localMemUsedByKernel;

        if( m_isBuffer )
        {
            cl_kernel_arg_address_qualifier addrQ = argInfo.getAddressQualifier();

            // decide about the buffer size - take into account the alignment and padding
            size = ws.getGlobalWorkSize() * alignment;

            // reduce the size of the buffer for local memory
            if (numLocalArgs &&
                size > floor(static_cast<double>(totalDeviceLocalMem / numLocalArgs)) &&
                addrQ == CL_KERNEL_ARG_ADDRESS_LOCAL)
            {
                size = floor(static_cast<double>(totalDeviceLocalMem / numLocalArgs));
            }

            if( CL_KERNEL_ARG_ADDRESS_CONSTANT == addrQ )
            {
                if ( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE < size )
                    size = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
            }

            if( CL_KERNEL_ARG_ADDRESS_GLOBAL   == addrQ ||
                CL_KERNEL_ARG_ADDRESS_CONSTANT == addrQ )
            {
                pBuffer = (T *)align_malloc(size, alignment);
                if (NULL == pBuffer)
                {
                     throw Exceptions::TestError("align_malloc failed for array buffer\n", 1);
                }
                assert( (size_t)pBuffer % alignment == 0 );
                if (NULL == refArg)
                {
                    fillBuffer(pBuffer, size / sizeof(T));
                }
                else {
                    memcpy(pBuffer, refArg->getBuffer(), size);
                }
            }
            return new KernelArgBuffer( context, argInfo, (void*)pBuffer, size);
        }
        else
        {
            if (m_vectorSize == 3)
                size = sizeof(T) * 4;
            else
                size = sizeof(T) * m_vectorSize;

            pBuffer = (T *)align_malloc(size, alignment);
            if (NULL == pBuffer)
            {
                throw Exceptions::TestError("align_malloc failed for pBuffer\n", 1);
            }
            assert( (size_t)pBuffer % alignment == 0 );
            if (NULL == refArg)
            {
                fillBuffer(pBuffer, m_vectorSize);
            }
            else {
                memcpy(pBuffer, refArg->getBuffer(), size);
            }
            return new KernelArg( argInfo, (void*)pBuffer, size);
        }
    }
private:
    void fillBuffer( T* buffer, size_t nelem)
    {
        for( size_t i = 0; i < nelem; ++i )
        {
            buffer[i]  = gRG.getNext<T>(m_minValue, m_maxValue);
        }
    }

private:
    bool m_isBuffer;
    size_t m_vectorSize;
    T    m_minValue;
    T    m_maxValue;
};

/**
 General facade for the kernel arguments generation functionality.
 */
class DataGenerator
{
public:
     static DataGenerator* getInstance();

    ~DataGenerator();

    KernelArg* generateKernelArg(cl_context context,
                                 const KernelArgInfo& argInfo,
                                 const WorkSizeInfo& ws,
                                 const KernelArg* refArg,
                                 const cl_kernel kernel,
                                 const cl_device_id device)
    {
        KernelArgGenerator* pArgGenerator = getArgGenerator(argInfo);
        return pArgGenerator->generate(context, ws, argInfo, refArg, kernel, device);
    }

    /*
     * Gets the generator associated to the given key.
     */
    KernelArgGenerator* getArgGenerator(const KernelArgInfo& argInfo);

    /*
     * Sets the entry associated to the given key, with the given prototype
     * generator.
     */
    void setArgGenerator(const KernelArgInfo& key, KernelArgGenerator* gen);

private:
    DataGenerator();

    static DataGenerator *Instance;

    typedef std::map<std::string, KernelArgGenerator*> ArgGeneratorsMap;
    ArgGeneratorsMap m_argGenerators;
};

class ImageValuesGenerator
{
public:
    class iterator
    {
        friend class ImageValuesGenerator;
    public:
        /*
         * Iterator operators.
         */
        iterator& operator ++();
        bool operator == (const iterator&) const;
        bool operator != (const iterator&) const;
        /*
         * Returns the name of the basic image type (e.g., image2d_t).
         */
        std::string getImageTypeName() const;

        /*
         * Returns the name of the genrator that generates images of this type
         * (e.g., imaget2d_float).
         */
        std::string getImageGeneratorName() const;

        /*
         * Returns the name of the genrator that generates images of the 'base'
         * type (e.g., imaget2d_t).
         */
        std::string getBaseImageGeneratorName() const;

        /*
         * Returns the OpenCL enumeration for the channel order of the image
         * object this iterator creates.
         */
        int getOpenCLChannelOrder() const;

        /*
         * Returns the SPIR enumeration for the channel order of the image
         * object this iterator creates.
         */
        int getSPIRChannelOrder() const;

        /*
         * Returns the data type of the image object this iterator creates. (e.g.,
         * cl_float, cl_int).
         */
        int getDataType() const;

        /*
         * Returns the data type of the image object this iterator creates. (e.g.,
         * float, int), in string format.
         */
        std::string getDataTypeName() const;

        std::string toString() const;
    private:
        /*
         * Constructor for creating a 'begin' iterator.
         */
        iterator(ImageValuesGenerator*);
        /*
         * Constructor for creating an 'end' iterator.
         */
        iterator(int);
        /*
        * Increments the given argument up to the given limit.
        * In case the new value reaches the limit, the index is reset to hold zero.
        * Returns: true if the value of the index was incremented, false if it was reset
        * to zero.
        */
        bool incrementIndex(size_t& index, size_t limit);

        /*
         * Returns true is the index combination of this iterator is legal,
         * or false otherwise.
         */
        bool isLegalCombination() const;

        ImageValuesGenerator* m_parent;
        size_t m_channelIndex, m_imgTyIndex;
    }; //End iterator.

    iterator begin();
    iterator end();

    static cl_channel_order channelOrders[];
    static const char* imageTypes[];
private:
    WorkSizeInfo  m_wsInfo;
};

#endif
