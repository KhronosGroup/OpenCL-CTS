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
#ifndef __UTILS_KHR_MEDIA_H
#define __UTILS_KHR_MEDIA_H

#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include "wrappers.h"
#include "CL/cl_dx9_media_sharing.h"

#include "harness/typeWrappers.h"





extern clGetDeviceIDsFromDX9MediaAdapterKHR_fn clGetDeviceIDsFromDX9MediaAdapterKHR;
extern clCreateFromDX9MediaSurfaceKHR_fn clCreateFromDX9MediaSurfaceKHR;
extern clEnqueueAcquireDX9MediaSurfacesKHR_fn clEnqueueAcquireDX9MediaSurfacesKHR;
extern clEnqueueReleaseDX9MediaSurfacesKHR_fn clEnqueueReleaseDX9MediaSurfacesKHR;

extern cl_platform_id gPlatformIDdetected;
extern cl_device_id gDeviceIDdetected;
extern cl_device_type gDeviceTypeSelected;

#define NL "\n"
#define TEST_NOT_IMPLEMENTED -1
#define TEST_NOT_SUPPORTED -2

enum TSurfaceFormat
{
  SURFACE_FORMAT_NV12,
  SURFACE_FORMAT_YV12,
  SURFACE_FORMAT_R32F,
  SURFACE_FORMAT_R16F,
  SURFACE_FORMAT_L16,
  SURFACE_FORMAT_A8,
  SURFACE_FORMAT_L8,
  SURFACE_FORMAT_G32R32F,
  SURFACE_FORMAT_G16R16F,
  SURFACE_FORMAT_G16R16,
  SURFACE_FORMAT_A8L8,
  SURFACE_FORMAT_A32B32G32R32F,
  SURFACE_FORMAT_A16B16G16R16F,
  SURFACE_FORMAT_A16B16G16R16,
  SURFACE_FORMAT_A8B8G8R8,
  SURFACE_FORMAT_X8B8G8R8,
  SURFACE_FORMAT_A8R8G8B8,
  SURFACE_FORMAT_X8R8G8B8,
};

enum TContextFuncType
{
  CONTEXT_CREATE_DEFAULT,
  CONTEXT_CREATE_FROM_TYPE,
};

enum TSharedHandleType
{
  SHARED_HANDLE_ENABLED,
  SHARED_HANDLE_DISABLED,
};

class CResult {
public:
  enum TTestResult {
    TEST_NORESULT,
    TEST_NOTSUPPORTED,
    TEST_PASS,
    TEST_FAIL,
    TEST_ERROR,
  };

  CResult();
  ~CResult();

  void ResultSub(TTestResult result);
  TTestResult ResultLast() const;
  int Result() const;

private:
  TTestResult _result;
  TTestResult _resultLast;
};

void FunctionContextCreateToString(TContextFuncType contextCreateFunction, std::string &contextFunction);
void AdapterToString(cl_dx9_media_adapter_type_khr adapterType, std::string &adapter);
cl_context_info AdapterTypeToContextInfo(cl_dx9_media_adapter_type_khr adapterType);

//YUV utils
void YUVGenerateNV12(std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height,
                     cl_uchar valueMin, cl_uchar valueMax, double valueAdd = 0.0);
void YUVGenerateYV12(std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height,
                     cl_uchar valueMin, cl_uchar valueMax, double valueAdd = 0.0);
bool YUVGenerate(TSurfaceFormat surfaceFormat, std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height,
                 cl_uchar valueMin, cl_uchar valueMax, double valueAdd = 0.0);
bool YUVSurfaceSetNV12(std::auto_ptr<CSurfaceWrapper> &surface, const std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height);
bool YUVSurfaceSetYV12(std::auto_ptr<CSurfaceWrapper> &surface, const std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height);
bool YUVSurfaceSet(TSurfaceFormat surfaceFormat, std::auto_ptr<CSurfaceWrapper> &surface, const std::vector<cl_uchar> &yuv,
                   unsigned int width, unsigned int height);
bool YUVSurfaceGetNV12(std::auto_ptr<CSurfaceWrapper> &surface, std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height);
bool YUVSurfaceGetYV12(std::auto_ptr<CSurfaceWrapper> &surface, std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height);
bool YUVSurfaceGet(TSurfaceFormat surfaceFormat, std::auto_ptr<CSurfaceWrapper> &surface, std::vector<cl_uchar> &yuv,
                   unsigned int width, unsigned int height);
bool YUVCompareNV12(const std::vector<cl_uchar> &yuvTest, const std::vector<cl_uchar> &yuvRef,
                    unsigned int width, unsigned int height);
bool YUVCompareYV12(const std::vector<cl_uchar> &yuvTest, const std::vector<cl_uchar> &yuvRef,
                    unsigned int width, unsigned int height);
bool YUVCompare(TSurfaceFormat surfaceFormat, const std::vector<cl_uchar> &yuvTest, const std::vector<cl_uchar> &yuvRef,
                unsigned int width, unsigned int height);

//other types utils
void DataGenerate(TSurfaceFormat surfaceFormat, cl_channel_type type, std::vector<float> &data, unsigned int width, unsigned int height,
                  unsigned int channelNum, float cmin = 0.0f, float cmax = 1.0f, float add = 0.0f);
void DataGenerate(TSurfaceFormat surfaceFormat, cl_channel_type type, std::vector<cl_half> &data, unsigned int width, unsigned int height,
                  unsigned int channelNum, float cmin = 0.0f, float cmax = 1.0f, float add = 0.0f);
void DataGenerate(TSurfaceFormat surfaceFormat, cl_channel_type type, std::vector<cl_uchar> &data, unsigned int width, unsigned int height,
                  unsigned int channelNum, float cmin = 0.0f, float cmax = 1.0f, float add = 0.0f);
bool DataCompare(TSurfaceFormat surfaceFormat, cl_channel_type type, const std::vector<cl_float> &dataTest, const std::vector<cl_float> &dataExp,
                 unsigned int width, unsigned int height, unsigned int channelNum);
bool DataCompare(TSurfaceFormat surfaceFormat, cl_channel_type type, const std::vector<cl_half> &dataTest, const std::vector<cl_half> &dataExp,
                 unsigned int width, unsigned int height, unsigned int channelNum);
bool DataCompare(TSurfaceFormat surfaceFormat, cl_channel_type type, const std::vector<cl_uchar> &dataTest, const std::vector<cl_uchar> &dataExp,
                 unsigned int width, unsigned int height, unsigned int channelNum);

bool GetImageInfo(cl_mem object, cl_image_format formatExp, size_t elementSizeExp,
                  size_t rowPitchExp, size_t slicePitchExp, size_t widthExp,
                  size_t heightExp, size_t depthExp, unsigned int planeExp);
bool GetMemObjInfo(cl_mem object, cl_dx9_media_adapter_type_khr adapterType, std::auto_ptr<CSurfaceWrapper> &surface, void *shareHandleExp);
bool ImageInfoVerify(cl_dx9_media_adapter_type_khr adapterType, const std::vector<cl_mem> &memObjList, unsigned int width, unsigned int height,
                     std::auto_ptr<CSurfaceWrapper> &surface, void *sharedHandle);
bool ImageFormatCheck(cl_context context, cl_mem_object_type imageType, const cl_image_format imageFormatCheck);
unsigned int ChannelNum(TSurfaceFormat surfaceFormat);
unsigned int PlanesNum(TSurfaceFormat surfaceFormat);

#if defined(_WIN32)
D3DFORMAT SurfaceFormatToD3D(TSurfaceFormat surfaceFormat);
#endif

bool DeviceCreate(cl_dx9_media_adapter_type_khr adapterType, std::auto_ptr<CDeviceWrapper> &device);
bool SurfaceFormatCheck(cl_dx9_media_adapter_type_khr adapterType, const CDeviceWrapper &device, TSurfaceFormat surfaceFormat);
bool SurfaceFormatToOCL(TSurfaceFormat surfaceFormat, cl_image_format &format);
void SurfaceFormatToString(TSurfaceFormat surfaceFormat, std::string &str );
bool MediaSurfaceCreate(cl_dx9_media_adapter_type_khr adapterType, unsigned int width, unsigned int height, TSurfaceFormat surfaceFormat,
                      CDeviceWrapper &device, std::auto_ptr<CSurfaceWrapper> &surface, bool sharedHandle, void **objectSharedHandle);

//imported from image helpers
cl_ushort float2half_rte( float f );
cl_ushort float2half_rtz( float f );
cl_ushort convert_float_to_half( float f );
float convert_half_to_float( unsigned short halfValue );
int DetectFloatToHalfRoundingMode( cl_command_queue );

cl_int deviceExistForCLTest(cl_platform_id platform,cl_dx9_media_adapter_type_khr media_adapters_type,void *media_adapters,CResult &result,TSharedHandleType sharedHandle=SHARED_HANDLE_DISABLED);
#endif  // __UTILS_KHR_MEDIA_H
