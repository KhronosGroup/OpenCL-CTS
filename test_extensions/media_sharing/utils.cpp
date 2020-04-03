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
#include "utils.h"

#include "harness/errorHelpers.h"
#include "harness/rounding_mode.h"

#include <math.h>

static RoundingMode gFloatToHalfRoundingMode = kDefaultRoundingMode;


CResult::CResult():
_result(TEST_PASS), _resultLast(TEST_NORESULT)
{

}

CResult::~CResult()
{

}

CResult::TTestResult CResult::ResultLast() const
{
  return _resultLast;
}

int CResult::Result() const
{
  switch (_result)
  {
  case TEST_NORESULT:
  case TEST_NOTSUPPORTED:
  case TEST_PASS:
    return 0;
    break;
  case TEST_FAIL:
    return 1;
    break;
  case TEST_ERROR:
    return 2;
    break;
  default:
    return -1;
    break;
  }
}

void CResult::ResultSub( TTestResult result )
{
  _resultLast = result;
  if (static_cast<int>(result) > static_cast<int>(_result))
    _result = result;
}

void FunctionContextCreateToString(TContextFuncType contextCreateFunction, std::string &contextFunction)
{
  switch(contextCreateFunction)
  {
  case CONTEXT_CREATE_DEFAULT:
    contextFunction = "CreateContext";
    break;
  case CONTEXT_CREATE_FROM_TYPE:
    contextFunction = "CreateContextFromType";
    break;
  default:
    contextFunction = "Unknown";
    log_error("FunctionContextCreateToString(): Unknown create function enum!");
    break;
  }
}

void AdapterToString(cl_dx9_media_adapter_type_khr adapterType, std::string &adapter)
{
  switch(adapterType)
  {
  case CL_ADAPTER_D3D9_KHR:
    adapter = "D3D9";
    break;
  case CL_ADAPTER_D3D9EX_KHR:
    adapter = "D3D9EX";
    break;
  case CL_ADAPTER_DXVA_KHR:
    adapter = "DXVA";
    break;
  default:
    adapter = "Unknown";
    log_error("AdapterToString(): Unknown adapter type!");
    break;
  }
}

cl_context_info AdapterTypeToContextInfo( cl_dx9_media_adapter_type_khr adapterType )
{
  switch (adapterType)
  {
  case CL_ADAPTER_D3D9_KHR:
    return CL_CONTEXT_ADAPTER_D3D9_KHR;
    break;
  case CL_ADAPTER_D3D9EX_KHR:
    return CL_CONTEXT_ADAPTER_D3D9EX_KHR;
    break;
  case CL_ADAPTER_DXVA_KHR:
    return CL_CONTEXT_ADAPTER_DXVA_KHR;
    break;
  default:
    log_error("AdapterTypeToContextInfo(): Unknown adapter type!");
    return 0;
    break;
  }
}

void YUVGenerateNV12( std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height,
                     cl_uchar valueMin, cl_uchar valueMax, double valueAdd )
{
  yuv.clear();
  yuv.resize(width * height * 3 / 2, 0);

  double min = static_cast<double>(valueMin);
  double max = static_cast<double>(valueMax);
  double range = 255;
  double add = static_cast<double>(valueAdd * range);
  double stepX = (max - min) / static_cast<double>(width);
  double stepY = (max - min) /static_cast<double>(height);

  //generate Y plane
  for (unsigned int i = 0; i < height; ++i)
  {
    unsigned int offset = i * width;
    double valueYPlane0 = static_cast<double>(stepY * i);
    for (unsigned int j = 0; j < width; ++j)
    {
      double valueXPlane0 = static_cast<double>(stepX * j);
      yuv.at(offset + j) = static_cast<cl_uchar>(min + valueXPlane0 / 2 + valueYPlane0 / 2 + add);
    }
  }

  //generate UV planes
  for (unsigned int i = 0; i < height / 2; ++i)
  {
    unsigned int offset = width * height + i * width;
    double valueYPlane1 = static_cast<double>(stepY * i);
    double valueYPlane2 = static_cast<double>(stepY * (height / 2 + i));
    for (unsigned int j = 0; j < width / 2; ++j)
    {
      double valueXPlane1 = static_cast<double>(stepX * j);
      double valueXPlane2 = static_cast<double>(stepX * (width / 2 + j));

      yuv.at(offset + j * 2) = static_cast<cl_uchar>(min + valueXPlane1 / 2 + valueYPlane1 / 2 + add);
      yuv.at(offset + j * 2 + 1) = static_cast<cl_uchar>(min + valueXPlane2 / 2 + valueYPlane2 / 2 + add);
    }
  }
}

void YUVGenerateYV12( std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height, cl_uchar valueMin, cl_uchar valueMax, double valueAdd /*= 0.0*/ )
{
  yuv.clear();
  yuv.resize(width * height * 3 / 2, 0);

  double min = static_cast<double>(valueMin);
  double max = static_cast<double>(valueMax);
  double range = 255;
  double add = static_cast<double>(valueAdd * range);
  double stepX = (max - min) / static_cast<double>(width);
  double stepY = (max - min) /static_cast<double>(height);

  unsigned offset = 0;

  //generate Y plane
  for (unsigned int i = 0; i < height; ++i)
  {
    unsigned int plane0Offset = offset + i * width;
    double valueYPlane0 = static_cast<double>(stepY * i);
    for (unsigned int j = 0; j < width; ++j)
    {
      double valueXPlane0 = static_cast<double>(stepX * j);
      yuv.at(plane0Offset + j) = static_cast<cl_uchar>(min + valueXPlane0 / 2 + valueYPlane0 / 2 + add);
    }
  }

  //generate V plane
  offset += width * height;
  for (unsigned int i = 0; i < height / 2; ++i)
  {
    unsigned int plane1Offset = offset + i * width / 2;
    double valueYPlane1 = static_cast<double>(stepY * i);
    for (unsigned int j = 0; j < width / 2; ++j)
    {
      double valueXPlane1 = static_cast<double>(stepX * j);
      yuv.at(plane1Offset + j) = static_cast<cl_uchar>(min + valueXPlane1 / 2 + valueYPlane1 / 2 + add);
    }
  }

  //generate U plane
  offset += width * height / 4;
  for (unsigned int i = 0; i < height / 2; ++i)
  {
    unsigned int plane2Offset = offset + i * width / 2;
    double valueYPlane2 = static_cast<double>(stepY * (height / 2 + i));
    for (unsigned int j = 0; j < width / 2; ++j)
    {
      double valueXPlane2 = static_cast<double>(stepX * j);
      yuv.at(plane2Offset + j) = static_cast<cl_uchar>(min + valueXPlane2 / 2 + valueYPlane2 / 2 + add);
    }
  }
}


bool YUVGenerate( TSurfaceFormat surfaceFormat, std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height, cl_uchar valueMin, cl_uchar valueMax, double valueAdd /*= 0.0*/ )
{
  switch (surfaceFormat)
  {
  case SURFACE_FORMAT_NV12:
    YUVGenerateNV12(yuv, width, height, valueMin, valueMax, valueAdd);
    break;
  case SURFACE_FORMAT_YV12:
    YUVGenerateYV12(yuv, width, height, valueMin, valueMax, valueAdd);
    break;
  default:
    log_error("YUVGenerate(): Invalid surface type\n");
    return false;
    break;
  }

  return true;
}

bool YUVSurfaceSetNV12( std::auto_ptr<CSurfaceWrapper> &surface, const std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height )
{
#if defined(_WIN32)
  CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
  D3DLOCKED_RECT rect;
  if (FAILED((*d3dSurface)->LockRect(&rect, NULL, 0)))
  {
    log_error("YUVSurfaceSetNV12(): Surface lock failed\n");
    return false;
  }

  size_t pitch = rect.Pitch / sizeof(cl_uchar);
  size_t lineSize = width * sizeof(cl_uchar);
  cl_uchar *ptr = static_cast<cl_uchar *>(rect.pBits);
  for (size_t y = 0; y < height; ++y)
    memcpy(ptr + y * pitch, &yuv.at(y * width), lineSize);

  for (size_t y = 0; y < height / 2; ++y)
    memcpy(ptr + height * pitch + y * pitch, &yuv.at(width * height + y * width), lineSize);

  (*d3dSurface)->UnlockRect();

  return true;

#else
  return false;
#endif
}

bool YUVSurfaceSetYV12( std::auto_ptr<CSurfaceWrapper> &surface, const std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height )
{
#if defined(_WIN32)
  CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
  D3DLOCKED_RECT rect;
  if (FAILED((*d3dSurface)->LockRect(&rect, NULL, 0)))
  {
    log_error("YUVSurfaceSetYV12(): Surface lock failed!\n");
    return false;
  }

  size_t pitch = rect.Pitch / sizeof(cl_uchar);
  size_t pitchHalf = pitch / 2;
  size_t lineSize = width * sizeof(cl_uchar);
  size_t lineHalfSize = lineSize / 2;
  size_t surfaceOffset = 0;
  size_t yuvOffset = 0;
  cl_uchar *ptr = static_cast<cl_uchar *>(rect.pBits);

  for (size_t y = 0; y < height; ++y)
    memcpy(ptr + surfaceOffset + y * pitch, &yuv.at(yuvOffset + y * width), lineSize);

  surfaceOffset += height * pitch;
  yuvOffset += width * height;
  for (size_t y = 0; y < height / 2; ++y)
    memcpy(ptr + surfaceOffset + y * pitchHalf, &yuv.at(yuvOffset + y * lineHalfSize), lineHalfSize);

  surfaceOffset += pitchHalf * height / 2;
  yuvOffset += width * height / 4;
  for (size_t y = 0; y < height / 2; ++y)
    memcpy(ptr + surfaceOffset + y * pitchHalf, &yuv.at(yuvOffset + y * lineHalfSize), lineHalfSize);

  (*d3dSurface)->UnlockRect();

  return true;

#else
  return false;
#endif
}

bool YUVSurfaceSet(TSurfaceFormat surfaceFormat, std::auto_ptr<CSurfaceWrapper> &surface, const std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height )
{
  switch (surfaceFormat)
  {
  case SURFACE_FORMAT_NV12:
    if(!YUVSurfaceSetNV12(surface, yuv, width, height))
      return false;
    break;
  case SURFACE_FORMAT_YV12:
    if(!YUVSurfaceSetYV12(surface, yuv, width, height))
      return false;
    break;
  default:
    log_error("YUVSurfaceSet(): Invalid surface type!\n");
    return false;
    break;
  }

  return true;
}

bool YUVSurfaceGetNV12( std::auto_ptr<CSurfaceWrapper> &surface, std::vector<cl_uchar> &yuv,
                       unsigned int width, unsigned int height )
{
#if defined(_WIN32)
  CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
  D3DLOCKED_RECT rect;
  if (FAILED((*d3dSurface)->LockRect(&rect, NULL, 0)))
  {
    log_error("YUVSurfaceGetNV12(): Surface lock failed!\n");
    return false;
  }

  size_t pitch = rect.Pitch / sizeof(cl_uchar);
  size_t lineSize = width * sizeof(cl_uchar);
  cl_uchar *ptr = static_cast<cl_uchar *>(rect.pBits);
  size_t yuvOffset = 0;
  size_t surfaceOffset = 0;
  for (size_t y = 0; y < height; ++y)
    memcpy(&yuv.at(yuvOffset + y * width), ptr + y * pitch, lineSize);

  yuvOffset += width * height;
  surfaceOffset += pitch * height;
  for (size_t y = 0; y < height / 2; ++y)
    memcpy(&yuv.at(yuvOffset + y * width), ptr + surfaceOffset + y * pitch, lineSize);

  (*d3dSurface)->UnlockRect();

  return true;

#else
  return false;
#endif
}

bool YUVSurfaceGetYV12( std::auto_ptr<CSurfaceWrapper> &surface, std::vector<cl_uchar> &yuv, unsigned int width, unsigned int height )
{
#if defined(_WIN32)
  CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
  D3DLOCKED_RECT rect;
  if (FAILED((*d3dSurface)->LockRect(&rect, NULL, 0)))
  {
    log_error("YUVSurfaceGetYV12(): Surface lock failed!\n");
    return false;
  }

  size_t pitch = rect.Pitch / sizeof(cl_uchar);
  size_t pitchHalf = pitch / 2;
  size_t lineSize = width * sizeof(cl_uchar);
  size_t lineHalfSize = lineSize / 2;
  size_t surfaceOffset = 0;
  size_t yuvOffset = 0;
  cl_uchar *ptr = static_cast<cl_uchar *>(rect.pBits);

  for (size_t y = 0; y < height; ++y)
    memcpy(&yuv.at(yuvOffset + y * width), ptr + surfaceOffset + y * pitch, lineSize);

  surfaceOffset += pitch * height;
  yuvOffset += width * height;
  for (size_t y = 0; y < height / 2; ++y)
    memcpy(&yuv.at(yuvOffset + y * lineHalfSize), ptr + surfaceOffset + y * pitchHalf, lineHalfSize);

  surfaceOffset += pitchHalf * height / 2;
  yuvOffset += width * height / 4;
  for (size_t y = 0; y < height / 2; ++y)
    memcpy(&yuv.at(yuvOffset + y * lineHalfSize), ptr + surfaceOffset + y * pitchHalf, lineHalfSize);

  (*d3dSurface)->UnlockRect();

  return true;

#else
  return false;
#endif
}

bool YUVSurfaceGet(TSurfaceFormat surfaceFormat, std::auto_ptr<CSurfaceWrapper> &surface, std::vector<cl_uchar> &yuv,
                   unsigned int width, unsigned int height )
{
  switch (surfaceFormat)
  {
  case SURFACE_FORMAT_NV12:
    if(!YUVSurfaceGetNV12(surface, yuv, width, height))
      return false;
    break;
  case SURFACE_FORMAT_YV12:
    if(!YUVSurfaceGetYV12(surface, yuv, width, height))
      return false;
    break;
  default:
    log_error("YUVSurfaceGet(): Invalid surface type!\n");
    return false;
    break;
  }

  return true;
}

bool YUVCompareNV12( const std::vector<cl_uchar> &yuvTest, const std::vector<cl_uchar> &yuvRef,
                    unsigned int width, unsigned int height )
{
  //plane 0 verification
  size_t offset = 0;
  for (size_t y = 0; y < height; ++y)
  {
    size_t plane0Offset = offset + width * y;
    for (size_t x = 0; x < width; ++x)
    {
      if (yuvTest[plane0Offset + x] != yuvRef[plane0Offset + x])
      {
        log_error("Plane 0 (Y) is different than expected, reference value: %i, test value: %i, x: %i, y: %i\n",
          yuvRef[plane0Offset + x], yuvTest[plane0Offset + x], x, y);
        return false;
      }
    }
  }

  //plane 1 and 2 verification
  offset += width * height;
  for (size_t y = 0; y < height / 2; ++y)
  {
    size_t plane12Offset = offset + width * y;
    for (size_t x = 0; x < width / 2; ++x)
    {
      if (yuvTest.at(plane12Offset + 2 * x) != yuvRef.at(plane12Offset + 2 * x))
      {
        log_error("Plane 1 (U) is different than expected, reference value: %i, test value: %i, x: %i, y: %i\n",
          yuvRef[plane12Offset + 2 * x], yuvTest[plane12Offset + 2 * x], x, y);
        return false;
      }

      if (yuvTest.at(plane12Offset + 2 * x + 1) != yuvRef.at(plane12Offset + 2 * x + 1))
      {
        log_error("Plane 2 (V) is different than expected, reference value: %i, test value: %i, x: %i, y: %i\n",
          yuvRef[plane12Offset + 2 * x + 1], yuvTest[plane12Offset + 2 * x + 1], x, y);
        return false;
      }
    }
  }

  return true;
}

bool YUVCompareYV12( const std::vector<cl_uchar> &yuvTest, const std::vector<cl_uchar> &yuvRef,
                    unsigned int width, unsigned int height )
{
  //plane 0 verification
  size_t offset = 0;
  for (size_t y = 0; y < height; ++y)
  {
    size_t plane0Offset = width * y;
    for (size_t x = 0; x < width; ++x)
    {
      if (yuvTest.at(plane0Offset + x) != yuvRef.at(plane0Offset + x))
      {
        log_error("Plane 0 (Y) is different than expected, reference value: %i, test value: %i, x: %i, y: %i\n",
          yuvRef[plane0Offset + x], yuvTest[plane0Offset + x], x ,y);
        return false;
      }
    }
  }

  //plane 1 verification
  offset += width * height;
  for (size_t y = 0; y < height / 2; ++y)
  {
    size_t plane1Offset = offset + width * y / 2;
    for (size_t x = 0; x < width / 2; ++x)
    {
      if (yuvTest.at(plane1Offset + x) != yuvRef.at(plane1Offset + x))
      {
        log_error("Plane 1 (V) is different than expected, reference value: %i, test value: %i, x: %i, y: %i\n",
          yuvRef[plane1Offset + x], yuvTest[plane1Offset + x], x, y);
        return false;
      }
    }
  }

  //plane 2 verification
  offset += width * height / 4;
  for (size_t y = 0; y < height / 2; ++y)
  {
    size_t plane2Offset = offset + width * y / 2;
    for (size_t x = 0; x < width / 2; ++x)
    {
      if (yuvTest.at(plane2Offset + x) != yuvRef.at(plane2Offset + x))
      {
        log_error("Plane 2 (U) is different than expected, reference value: %i, test value: %i, x: %i, y: %i\n",
          yuvRef[plane2Offset + x], yuvTest[plane2Offset + x], x, y);
        return false;
      }
    }
  }

  return true;
}

bool YUVCompare( TSurfaceFormat surfaceFormat, const std::vector<cl_uchar> &yuvTest, const std::vector<cl_uchar> &yuvRef,
                unsigned int width, unsigned int height )
{
  switch (surfaceFormat)
  {
  case SURFACE_FORMAT_NV12:
    if (!YUVCompareNV12(yuvTest, yuvRef, width, height))
    {
      log_error("OCL object is different than expected!\n");
      return false;
    }
    break;
  case SURFACE_FORMAT_YV12:
    if (!YUVCompareYV12(yuvTest, yuvRef, width, height))
    {
      log_error("OCL object is different than expected!\n");
      return false;
    }
    break;
  default:
    log_error("YUVCompare(): Invalid surface type!\n");
    return false;
    break;
  }

  return true;
}

void DataGenerate( TSurfaceFormat surfaceFormat, cl_channel_type type, std::vector<float> &data, unsigned int width, unsigned int height,
                  unsigned int channelNum, float cmin /*= 0.0f*/, float cmax /*= 1.0f*/, float add /*= 0.0f*/ )
{
  data.clear();
  data.reserve(width * height * channelNum);

  double valueMin = static_cast<double>(cmin);
  double valueMax = static_cast<double>(cmax);
  double stepX = (valueMax - valueMin) / static_cast<double>(width);
  double stepY = (valueMax - valueMin) /static_cast<double>(height);
  double valueAdd = static_cast<double>(add);
  for (unsigned int i = 0; i < height; ++i)
  {
    double valueY = static_cast<double>(stepY * i);
    for (unsigned int j = 0; j < width; ++j)
    {
      double valueX = static_cast<double>(stepX * j);
      switch (channelNum)
      {
      case 1:
        data.push_back(static_cast<float>(valueMin + valueX / 2 + valueY / 2 + valueAdd));
        break;
      case 2:
        data.push_back(static_cast<float>(valueMin + valueX + valueAdd));
        data.push_back(static_cast<float>(valueMin + valueY + valueAdd));
        break;
      case 4:
        data.push_back(static_cast<float>(valueMin + valueX + valueAdd));
        data.push_back(static_cast<float>(valueMin + valueY + valueAdd));
        data.push_back(static_cast<float>(valueMin + valueX / 2 + valueAdd));
        data.push_back(static_cast<float>(valueMin + valueY / 2 + valueAdd));
        break;
      default:
        log_error("DataGenerate(): invalid channel number!");
        return;
        break;
      }
    }
  }
}

void DataGenerate( TSurfaceFormat surfaceFormat, cl_channel_type type, std::vector<cl_half> &data, unsigned int width, unsigned int height,
                  unsigned int channelNum, float cmin /*= 0.0f*/, float cmax /*= 1.0f*/, float add /*= 0.0f*/ )
{
  data.clear();
  data.reserve(width * height * channelNum);

  double valueMin = static_cast<double>(cmin);
  double valueMax = static_cast<double>(cmax);
  double stepX = (valueMax - valueMin) / static_cast<double>(width);
  double stepY = (valueMax - valueMin) /static_cast<double>(height);

  switch(type)
  {
  case CL_HALF_FLOAT:
    {
      double valueAdd = static_cast<double>(add);

      for (unsigned int i = 0; i < height; ++i)
      {
        double valueY = static_cast<double>(stepY * i);
        for (unsigned int j = 0; j < width; ++j)
        {
          double valueX = static_cast<double>(stepX * j);
          switch (channelNum)
          {
          case 1:
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueX / 2 + valueY / 2 + valueAdd)));
            break;
          case 2:
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueX + valueAdd)));
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueY + valueAdd)));
            break;
          case 4:
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueX + valueAdd)));
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueY + valueAdd)));
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueX / 2 + valueAdd)));
            data.push_back(convert_float_to_half(static_cast<float>(valueMin + valueY / 2 + valueAdd)));
            break;
          default:
            log_error("DataGenerate(): invalid channel number!");
            return;
            break;
          }
        }
      }
      break;
    }
  case CL_UNORM_INT16:
    {
      double range = 65535;
      double valueAdd = static_cast<double>(add * range);

      for (unsigned int i = 0; i < height; ++i)
      {
        double valueY = static_cast<double>(stepY * i * range);
        for (unsigned int j = 0; j < width; ++j)
        {
          double valueX = static_cast<double>(stepX * j * range);
          switch (channelNum)
          {
          case 1:
            data.push_back(static_cast<cl_ushort>(valueMin + valueX / 2 + valueY / 2 + valueAdd));
            break;
          case 2:
            data.push_back(static_cast<cl_ushort>(valueMin + valueX + valueAdd));
            data.push_back(static_cast<cl_ushort>(valueMin + valueY + valueAdd));
            break;
          case 4:
            data.push_back(static_cast<cl_ushort>(valueMin + valueX + valueAdd));
            data.push_back(static_cast<cl_ushort>(valueMin + valueY + valueAdd));
            data.push_back(static_cast<cl_ushort>(valueMin + valueX / 2 + valueAdd));
            data.push_back(static_cast<cl_ushort>(valueMin + valueY / 2 + valueAdd));
            break;
          default:
            log_error("DataGenerate(): invalid channel number!");
            return;
            break;
          }
        }
      }
    }
    break;
  default:
    log_error("DataGenerate(): unknown data type!");
    return;
    break;
  }
}

void DataGenerate( TSurfaceFormat surfaceFormat, cl_channel_type type, std::vector<cl_uchar> &data, unsigned int width, unsigned int height,
                  unsigned int channelNum, float cmin /*= 0.0f*/, float cmax /*= 1.0f*/, float add /*= 0.0f*/ )
{
  data.clear();
  data.reserve(width * height * channelNum);

  double valueMin = static_cast<double>(cmin);
  double valueMax = static_cast<double>(cmax);
  double stepX = (valueMax - valueMin) / static_cast<double>(width);
  double stepY = (valueMax - valueMin) /static_cast<double>(height);

  double range = 255;
  double valueAdd = static_cast<double>(add * range);

  for (unsigned int i = 0; i < height; ++i)
  {
    double valueY = static_cast<double>(stepY * i * range);
    for (unsigned int j = 0; j < width; ++j)
    {
      double valueX = static_cast<double>(stepX * j * range);
      switch (channelNum)
      {
      case 1:
        data.push_back(static_cast<cl_uchar>(valueMin + valueX / 2 + valueY / 2 + valueAdd));
        break;
      case 2:
        data.push_back(static_cast<cl_uchar>(valueMin + valueX + valueAdd));
        data.push_back(static_cast<cl_uchar>(valueMin + valueY + valueAdd));
        break;
      case 4:
        data.push_back(static_cast<cl_uchar>(valueMin + valueX + valueAdd));
        data.push_back(static_cast<cl_uchar>(valueMin + valueY + valueAdd));
        data.push_back(static_cast<cl_uchar>(valueMin + valueX / 2 + valueAdd));
        if (surfaceFormat == SURFACE_FORMAT_X8R8G8B8)
          data.push_back(static_cast<cl_uchar>(0xff));
        else
          data.push_back(static_cast<cl_uchar>(valueMin + valueY / 2 + valueAdd));
        break;
      default:
        log_error("DataGenerate(): invalid channel number!");
        return;
        break;
      }
    }
  }
}

bool DataCompare( TSurfaceFormat surfaceFormat, cl_channel_type type, const std::vector<float> &dataTest, const std::vector<float> &dataExp,
                 unsigned int width, unsigned int height, unsigned int channelNum)
{
  float epsilon = 0.000001f;
  for (unsigned int i = 0; i < height; ++i)
  {
    unsigned int offset = i * width * channelNum;
    for (unsigned int j = 0; j < width; ++j)
    {
      for(unsigned planeIdx = 0; planeIdx < channelNum; ++planeIdx)
      {
        if (abs(dataTest.at(offset + j * channelNum + planeIdx) - dataExp.at(offset + j * channelNum + planeIdx)) > epsilon)
        {
          log_error("Tested image is different than reference (x,y,plane) = (%i,%i,%i), test value = %f, expected value = %f\n",
            j, i, planeIdx, dataTest[offset + j * channelNum + planeIdx], dataExp[offset + j * channelNum + planeIdx]);
          return false;
        }
      }
    }
  }

  return true;
}

bool DataCompare( TSurfaceFormat surfaceFormat, cl_channel_type type, const std::vector<cl_half> &dataTest, const std::vector<cl_half> &dataExp,
                 unsigned int width, unsigned int height, unsigned int channelNum)
{
  switch(type)
  {
  case CL_HALF_FLOAT:
    {
      float epsilon = 0.001f;
      for (unsigned int i = 0; i < height; ++i)
      {
        unsigned int offset = i * width * channelNum;
        for (unsigned int j = 0; j < width; ++j)
        {
          for(unsigned planeIdx = 0; planeIdx < channelNum; ++planeIdx)
          {
            float test = convert_half_to_float(dataTest.at(offset + j * channelNum + planeIdx));
            float ref = convert_half_to_float(dataExp.at(offset + j * channelNum + planeIdx));
            if (abs(test - ref) > epsilon)
            {
              log_error("Tested image is different than reference (x,y,plane) = (%i,%i,%i), test value = %f, expected value = %f\n",
                j, i, planeIdx, test, ref);
              return false;
            }
          }
        }
      }
    }
    break;
  case CL_UNORM_INT16:
    {
      cl_ushort epsilon = 1;
      for (unsigned int i = 0; i < height; ++i)
      {
        unsigned int offset = i * width * channelNum;
        for (unsigned int j = 0; j < width; ++j)
        {
          for(unsigned planeIdx = 0; planeIdx < channelNum; ++planeIdx)
          {
            cl_ushort test = dataTest.at(offset + j * channelNum + planeIdx);
            cl_ushort ref = dataExp.at(offset + j * channelNum + planeIdx);
            if (abs(test - ref) > epsilon)
            {
              log_error("Tested image is different than reference (x,y,plane) = (%i,%i,%i), test value = %i, expected value = %i\n", j, i, planeIdx, test, ref);
              return false;
            }
          }
        }
      }
    }
    break;
  default:
    log_error("DataCompare(): Invalid data format!");
    return false;
    break;
  }

  return true;
}

bool DataCompare( TSurfaceFormat surfaceFormat, cl_channel_type type, const std::vector<cl_uchar> &dataTest, const std::vector<cl_uchar> &dataExp,
                 unsigned int width, unsigned int height, unsigned int planeNum )
{
  for (unsigned int i = 0; i < height; ++i)
  {
    unsigned int offset = i * width * planeNum;
    for (unsigned int j = 0; j < width; ++j)
    {
      for(unsigned planeIdx = 0; planeIdx < planeNum; ++planeIdx)
      {
        if (surfaceFormat == SURFACE_FORMAT_X8R8G8B8 && planeIdx == 3)
          continue;

        cl_uchar test = dataTest.at(offset + j * planeNum + planeIdx);
        cl_uchar ref = dataExp.at(offset + j * planeNum + planeIdx);
        if (test != ref)
        {
          log_error("Tested image is different than reference (x,y,plane) = (%i,%i,%i), test value = %i, expected value = %i\n",
            j, i, planeIdx, test, ref);
          return false;
        }
      }
    }
  }

  return true;
}

bool GetImageInfo( cl_mem object, cl_image_format formatExp, size_t elementSizeExp, size_t rowPitchExp,
                  size_t slicePitchExp, size_t widthExp, size_t heightExp, size_t depthExp , unsigned int planeExp)
{
  bool result = true;

  cl_image_format format;
  if (clGetImageInfo(object, CL_IMAGE_FORMAT, sizeof(cl_image_format), &format, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_FORMAT) failed\n");
    result = false;
  }

  if (formatExp.image_channel_order != format.image_channel_order || formatExp.image_channel_data_type != format.image_channel_data_type)
  {
    log_error("Value of CL_IMAGE_FORMAT is different than expected\n");
    result = false;
  }

  size_t elementSize = 0;
  if (clGetImageInfo(object, CL_IMAGE_ELEMENT_SIZE, sizeof(size_t), &elementSize, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_ELEMENT_SIZE) failed\n");
    result = false;
  }

  if (elementSizeExp != elementSize)
  {
    log_error("Value of CL_IMAGE_ELEMENT_SIZE is different than expected (size: %i, exp size: %i)\n", elementSize, elementSizeExp);
    result = false;
  }

  size_t rowPitch = 0;
  if (clGetImageInfo(object, CL_IMAGE_ROW_PITCH, sizeof(size_t), &rowPitch, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_ROW_PITCH) failed\n");
    result = false;
  }

  if ((rowPitchExp == 0 && rowPitchExp != rowPitch) || (rowPitchExp > 0 && rowPitchExp > rowPitch))
  {
    log_error("Value of CL_IMAGE_ROW_PITCH is different than expected (size: %i, exp size: %i)\n", rowPitch, rowPitchExp);
    result = false;
  }

  size_t slicePitch = 0;
  if (clGetImageInfo(object, CL_IMAGE_SLICE_PITCH, sizeof(size_t), &slicePitch, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_SLICE_PITCH) failed\n");
    result = false;
  }

  if ((slicePitchExp == 0 && slicePitchExp != slicePitch) || (slicePitchExp > 0 && slicePitchExp > slicePitch))
  {
    log_error("Value of CL_IMAGE_SLICE_PITCH is different than expected (size: %i, exp size: %i)\n", slicePitch, slicePitchExp);
    result = false;
  }

  size_t width = 0;
  if (clGetImageInfo(object, CL_IMAGE_WIDTH, sizeof(size_t), &width, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_WIDTH) failed\n");
    result = false;
  }

  if (widthExp != width)
  {
    log_error("Value of CL_IMAGE_WIDTH is different than expected (size: %i, exp size: %i)\n", width, widthExp);
    result = false;
  }

  size_t height = 0;
  if (clGetImageInfo(object, CL_IMAGE_HEIGHT, sizeof(size_t), &height, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_HEIGHT) failed\n");
    result = false;
  }

  if (heightExp != height)
  {
    log_error("Value of CL_IMAGE_HEIGHT is different than expected (size: %i, exp size: %i)\n", height, heightExp);
    result = false;
  }

  size_t depth = 0;
  if (clGetImageInfo(object, CL_IMAGE_DEPTH, sizeof(size_t), &depth, 0) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_DEPTH) failed\n");
    result = false;
  }

  if (depthExp != depth)
  {
    log_error("Value of CL_IMAGE_DEPTH is different than expected (size: %i, exp size: %i)\n", depth, depthExp);
    result = false;
  }

  unsigned int plane = 99;
  size_t paramSize = 0;
  if (clGetImageInfo(object, CL_IMAGE_DX9_MEDIA_PLANE_KHR, sizeof(unsigned int), &plane, &paramSize) != CL_SUCCESS)
  {
    log_error("clGetImageInfo(CL_IMAGE_MEDIA_SURFACE_PLANE_KHR) failed\n");
    result = false;
  }

  if (planeExp != plane)
  {
    log_error("Value of CL_IMAGE_MEDIA_SURFACE_PLANE_KHR is different than expected (plane: %i, exp plane: %i)\n", plane, planeExp);
    result = false;
  }

  return result;
}

bool GetMemObjInfo( cl_mem object, cl_dx9_media_adapter_type_khr adapterType,  std::auto_ptr<CSurfaceWrapper> &surface, void *shareHandleExp )
{
  bool result = true;
  switch(adapterType)
  {
  case CL_ADAPTER_D3D9_KHR:
  case CL_ADAPTER_D3D9EX_KHR:
  case CL_ADAPTER_DXVA_KHR:
    {
#if defined(_WIN32)
      cl_dx9_surface_info_khr surfaceInfo;
#else
      void *surfaceInfo = 0;
      return false;
#endif
      size_t paramSize = 0;
      if(clGetMemObjectInfo(object, CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR, sizeof(surfaceInfo), &surfaceInfo, &paramSize) != CL_SUCCESS)
      {
        log_error("clGetImageInfo(CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR) failed\n");
        result = false;
      }

#if defined(_WIN32)
      CD3D9SurfaceWrapper *d3d9Surface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
      if (*d3d9Surface != surfaceInfo.resource)
      {
        log_error("Invalid resource for CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR\n");
        result = false;
      }

      if (shareHandleExp != surfaceInfo.shared_handle)
      {
        log_error("Invalid shared handle for CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR\n");
        result = false;
      }
#else
      return false;
#endif

      if (paramSize != sizeof(surfaceInfo))
      {
        log_error("Invalid CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR parameter size: %i, expected: %i\n", paramSize, sizeof(surfaceInfo));
        result = false;
      }

      paramSize = 0;
      cl_dx9_media_adapter_type_khr mediaAdapterType;
      if(clGetMemObjectInfo(object, CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR, sizeof(mediaAdapterType), &mediaAdapterType, &paramSize) != CL_SUCCESS)
      {
        log_error("clGetImageInfo(CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR) failed\n");
        result = false;
      }

      if (adapterType != mediaAdapterType)
      {
        log_error("Invalid media adapter type for CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR\n");
        result = false;
      }

      if (paramSize != sizeof(mediaAdapterType))
      {
        log_error("Invalid CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR parameter size: %i, expected: %i\n", paramSize, sizeof(mediaAdapterType));
        result = false;
      }
    }
    break;
  default:
    log_error("GetMemObjInfo(): Unknown adapter type!\n");
    return false;
    break;
  }

  return result;
}

bool ImageInfoVerify( cl_dx9_media_adapter_type_khr adapterType, const std::vector<cl_mem> &memObjList, unsigned int width, unsigned int height,
                     std::auto_ptr<CSurfaceWrapper> &surface, void *sharedHandle)
{
  if (memObjList.size() != 2 && memObjList.size() != 3)
  {
    log_error("ImageInfoVerify(): Invalid object list parameter\n");
    return false;
  }

  cl_image_format formatPlane;
  formatPlane.image_channel_data_type = CL_UNORM_INT8;
  formatPlane.image_channel_order = CL_R;

  //plane 0 verification
  if (!GetImageInfo(memObjList[0], formatPlane, sizeof(cl_uchar),
    width * sizeof(cl_uchar),
    0,
    width, height, 0, 0))
  {
    log_error("clGetImageInfo failed\n");
    return false;
  }

  switch (memObjList.size())
  {
  case 2:
    {
      formatPlane.image_channel_data_type = CL_UNORM_INT8;
      formatPlane.image_channel_order = CL_RG;
      if (!GetImageInfo(memObjList[1], formatPlane, sizeof(cl_uchar) * 2,
        width * sizeof(cl_uchar),
        0,
        width / 2, height / 2, 0, 1))
      {
        log_error("clGetImageInfo failed\n");
        return false;
      }
    }
    break;
  case 3:
    {
      if (!GetImageInfo(memObjList[1], formatPlane, sizeof(cl_uchar),
        width * sizeof(cl_uchar) / 2,
        0,
        width / 2, height / 2, 0, 1))
      {
        log_error("clGetImageInfo failed\n");
        return false;
      }

      if (!GetImageInfo(memObjList[2], formatPlane, sizeof(cl_uchar),
        width * sizeof(cl_uchar) / 2,
        0,
        width / 2, height / 2, 0, 2))
      {
        log_error("clGetImageInfo failed\n");
        return false;
      }
    }
    break;
  default:
    log_error("ImageInfoVerify(): Invalid object list parameter\n");
    return false;
    break;
  }

  for (size_t i = 0; i < memObjList.size(); ++i)
  {
    if (!GetMemObjInfo(memObjList[i], adapterType, surface, sharedHandle))
    {
      log_error("clGetMemObjInfo(%i) failed\n", i);
      return false;
    }
  }

  return true;
}

bool ImageFormatCheck(cl_context context, cl_mem_object_type imageType, const cl_image_format imageFormatCheck)
{
  cl_uint imageFormatsNum = 0;
  cl_int error = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, imageType, 0, 0, &imageFormatsNum);
  if(error != CL_SUCCESS)
  {
    log_error("clGetSupportedImageFormats failed\n");
    return false;
  }

  if(imageFormatsNum < 1)
  {
    log_error("Invalid image format number returned by clGetSupportedImageFormats\n");
    return false;
  }

  std::vector<cl_image_format> imageFormats(imageFormatsNum);
  error = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, imageType, imageFormatsNum, &imageFormats[0], 0);
  if(error != CL_SUCCESS)
  {
    log_error("clGetSupportedImageFormats failed\n");
    return false;
  }

  for(cl_uint i = 0; i < imageFormatsNum; ++i)
  {
    if(imageFormats[i].image_channel_data_type == imageFormatCheck.image_channel_data_type
      && imageFormats[i].image_channel_order == imageFormatCheck.image_channel_order)
    {
      return true;
    }
  }

  return false;
}

unsigned int ChannelNum( TSurfaceFormat surfaceFormat )
{
  switch(surfaceFormat)
  {
  case SURFACE_FORMAT_R32F:
  case SURFACE_FORMAT_R16F:
  case SURFACE_FORMAT_L16:
  case SURFACE_FORMAT_A8:
  case SURFACE_FORMAT_L8:
    return 1;
    break;
  case SURFACE_FORMAT_G32R32F:
  case SURFACE_FORMAT_G16R16F:
  case SURFACE_FORMAT_G16R16:
  case SURFACE_FORMAT_A8L8:
    return 2;
    break;
  case SURFACE_FORMAT_NV12:
  case SURFACE_FORMAT_YV12:
    return 3;
    break;
  case SURFACE_FORMAT_A32B32G32R32F:
  case SURFACE_FORMAT_A16B16G16R16F:
  case SURFACE_FORMAT_A16B16G16R16:
  case SURFACE_FORMAT_A8B8G8R8:
  case SURFACE_FORMAT_X8B8G8R8:
  case SURFACE_FORMAT_A8R8G8B8:
  case SURFACE_FORMAT_X8R8G8B8:
    return 4;
    break;
  default:
    log_error("ChannelNum(): unknown surface format!\n");
    return 0;
    break;
  }
}

unsigned int PlanesNum( TSurfaceFormat surfaceFormat )
{
  switch(surfaceFormat)
  {
  case SURFACE_FORMAT_R32F:
  case SURFACE_FORMAT_R16F:
  case SURFACE_FORMAT_L16:
  case SURFACE_FORMAT_A8:
  case SURFACE_FORMAT_L8:
  case SURFACE_FORMAT_G32R32F:
  case SURFACE_FORMAT_G16R16F:
  case SURFACE_FORMAT_G16R16:
  case SURFACE_FORMAT_A8L8:
  case SURFACE_FORMAT_A32B32G32R32F:
  case SURFACE_FORMAT_A16B16G16R16F:
  case SURFACE_FORMAT_A16B16G16R16:
  case SURFACE_FORMAT_A8B8G8R8:
  case SURFACE_FORMAT_X8B8G8R8:
  case SURFACE_FORMAT_A8R8G8B8:
  case SURFACE_FORMAT_X8R8G8B8:
    return 1;
    break;
  case SURFACE_FORMAT_NV12:
    return 2;
    break;
  case SURFACE_FORMAT_YV12:
    return 3;
    break;
  default:
    log_error("PlanesNum(): unknown surface format!\n");
    return 0;
    break;
  }
}

#if defined(_WIN32)
D3DFORMAT SurfaceFormatToD3D(TSurfaceFormat surfaceFormat)
{
  switch(surfaceFormat)
  {
  case SURFACE_FORMAT_R32F:
    return D3DFMT_R32F;
    break;
  case SURFACE_FORMAT_R16F:
    return D3DFMT_R16F;
    break;
  case SURFACE_FORMAT_L16:
    return D3DFMT_L16;
    break;
  case SURFACE_FORMAT_A8:
    return D3DFMT_A8;
    break;
  case SURFACE_FORMAT_L8:
    return D3DFMT_L8;
    break;
  case SURFACE_FORMAT_G32R32F:
    return D3DFMT_G32R32F;
    break;
  case SURFACE_FORMAT_G16R16F:
    return D3DFMT_G16R16F;
    break;
  case SURFACE_FORMAT_G16R16:
    return D3DFMT_G16R16;
    break;
  case SURFACE_FORMAT_A8L8:
    return D3DFMT_A8L8;
    break;
  case SURFACE_FORMAT_A32B32G32R32F:
    return D3DFMT_A32B32G32R32F;
    break;
  case SURFACE_FORMAT_A16B16G16R16F:
    return D3DFMT_A16B16G16R16F;
    break;
  case SURFACE_FORMAT_A16B16G16R16:
    return D3DFMT_A16B16G16R16;
    break;
  case SURFACE_FORMAT_A8B8G8R8:
    return D3DFMT_A8B8G8R8;
    break;
  case SURFACE_FORMAT_X8B8G8R8:
    return D3DFMT_X8B8G8R8;
    break;
  case SURFACE_FORMAT_A8R8G8B8:
    return D3DFMT_A8R8G8B8;
    break;
  case SURFACE_FORMAT_X8R8G8B8:
    return D3DFMT_X8R8G8B8;
    break;
  case SURFACE_FORMAT_NV12:
    return static_cast<D3DFORMAT>(MAKEFOURCC('N', 'V', '1', '2'));
    break;
  case SURFACE_FORMAT_YV12:
    return static_cast<D3DFORMAT>(MAKEFOURCC('Y', 'V', '1', '2'));
    break;
  default:
    log_error("SurfaceFormatToD3D(): unknown surface format!\n");
    return D3DFMT_R32F;
    break;
  }
}
#endif

bool DeviceCreate( cl_dx9_media_adapter_type_khr adapterType, std::auto_ptr<CDeviceWrapper> &device )
{
  switch (adapterType)
  {
#if defined(_WIN32)
  case CL_ADAPTER_D3D9_KHR:
    device = std::auto_ptr<CDeviceWrapper>(new CD3D9Wrapper());
    break;
  case CL_ADAPTER_D3D9EX_KHR:
    device = std::auto_ptr<CDeviceWrapper>(new CD3D9ExWrapper());
    break;
  case CL_ADAPTER_DXVA_KHR:
    device = std::auto_ptr<CDeviceWrapper>(new CDXVAWrapper());
    break;
#endif
  default:
    log_error("DeviceCreate(): Unknown adapter type!\n");
    return false;
    break;
  }

  return device->Status();
}

bool SurfaceFormatCheck( cl_dx9_media_adapter_type_khr adapterType, const CDeviceWrapper &device, TSurfaceFormat surfaceFormat )
{
  switch (adapterType)
  {
#if defined(_WIN32)
  case CL_ADAPTER_D3D9_KHR:
  case CL_ADAPTER_D3D9EX_KHR:
  case CL_ADAPTER_DXVA_KHR:
    {
      D3DFORMAT d3dFormat = SurfaceFormatToD3D(surfaceFormat);
      LPDIRECT3D9 d3d9 = static_cast<LPDIRECT3D9>(device.D3D());
      D3DDISPLAYMODE d3ddm;
      d3d9->GetAdapterDisplayMode(device.AdapterIdx(), &d3ddm);

      if( FAILED(d3d9->CheckDeviceFormat(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, d3ddm.Format, 0, D3DRTYPE_SURFACE, d3dFormat)) )
        return false;
    }
    break;
#endif
  default:
    log_error("SurfaceFormatCheck(): Unknown adapter type!\n");
    return false;
    break;
  }

  return true;
}

bool SurfaceFormatToOCL(TSurfaceFormat surfaceFormat, cl_image_format &format)
{
  switch(surfaceFormat)
  {
  case SURFACE_FORMAT_R32F:
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_FLOAT;
    break;
  case SURFACE_FORMAT_R16F:
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_HALF_FLOAT;
    break;
  case SURFACE_FORMAT_L16:
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT16;
    break;
  case SURFACE_FORMAT_A8:
    format.image_channel_order = CL_A;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_L8:
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_G32R32F:
    format.image_channel_order = CL_RG;
    format.image_channel_data_type = CL_FLOAT;
    break;
  case SURFACE_FORMAT_G16R16F:
    format.image_channel_order = CL_RG;
    format.image_channel_data_type = CL_HALF_FLOAT;
    break;
  case SURFACE_FORMAT_G16R16:
    format.image_channel_order = CL_RG;
    format.image_channel_data_type = CL_UNORM_INT16;
    break;
  case SURFACE_FORMAT_A8L8:
    format.image_channel_order = CL_RG;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_A32B32G32R32F:
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_FLOAT;
    break;
  case SURFACE_FORMAT_A16B16G16R16F:
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_HALF_FLOAT;
    break;
  case SURFACE_FORMAT_A16B16G16R16:
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT16;
    break;
  case SURFACE_FORMAT_A8B8G8R8:
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_X8B8G8R8:
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_A8R8G8B8:
    format.image_channel_order = CL_BGRA;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_X8R8G8B8:
    format.image_channel_order = CL_BGRA;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_NV12:
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  case SURFACE_FORMAT_YV12:
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;
    break;
  default:
    log_error("SurfaceFormatToOCL(): Unknown surface format!\n");
    return false;
    break;
  }

  return true;
}

void SurfaceFormatToString( TSurfaceFormat surfaceFormat, std::string &str )
{
  switch(surfaceFormat)
  {
  case SURFACE_FORMAT_R32F:
    str = "R32F";
    break;
  case SURFACE_FORMAT_R16F:
    str = "R16F";
    break;
  case SURFACE_FORMAT_L16:
    str = "L16";
    break;
  case SURFACE_FORMAT_A8:
    str = "A8";
    break;
  case SURFACE_FORMAT_L8:
    str = "L8";
    break;
  case SURFACE_FORMAT_G32R32F:
    str = "G32R32F";
    break;
  case SURFACE_FORMAT_G16R16F:
    str = "G16R16F";
    break;
  case SURFACE_FORMAT_G16R16:
    str = "G16R16";
    break;
  case SURFACE_FORMAT_A8L8:
    str = "A8L8";
    break;
  case SURFACE_FORMAT_A32B32G32R32F:
    str = "A32B32G32R32F";
    break;
  case SURFACE_FORMAT_A16B16G16R16F:
    str = "A16B16G16R16F";
    break;
  case SURFACE_FORMAT_A16B16G16R16:
    str = "A16B16G16R16";
    break;
  case SURFACE_FORMAT_A8B8G8R8:
    str = "A8B8G8R8";
    break;
  case SURFACE_FORMAT_X8B8G8R8:
    str = "X8B8G8R8";
    break;
  case SURFACE_FORMAT_A8R8G8B8:
    str = "A8R8G8B8";
    break;
  case SURFACE_FORMAT_X8R8G8B8:
    str = "X8R8G8B8";
    break;
  case SURFACE_FORMAT_NV12:
    str = "NV12";
    break;
  case SURFACE_FORMAT_YV12:
    str = "YV12";
    break;
  default:
    log_error("SurfaceFormatToString(): unknown surface format!\n");
    str = "unknown";
    break;
  }
}

bool MediaSurfaceCreate(cl_dx9_media_adapter_type_khr adapterType, unsigned int width, unsigned int height, TSurfaceFormat surfaceFormat,
                        CDeviceWrapper &device, std::auto_ptr<CSurfaceWrapper> &surface, bool sharedHandle, void **objectSharedHandle)
{
  switch (adapterType)
  {
#if defined(_WIN32)
  case CL_ADAPTER_D3D9_KHR:
    {
      surface = std::auto_ptr<CD3D9SurfaceWrapper>(new CD3D9SurfaceWrapper);
      CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
      HRESULT hr = 0;
      D3DFORMAT d3dFormat = SurfaceFormatToD3D(surfaceFormat);
      LPDIRECT3DDEVICE9 d3d9Device = (LPDIRECT3DDEVICE9)device.Device();
      hr = d3d9Device->CreateOffscreenPlainSurface(width, height, d3dFormat, D3DPOOL_DEFAULT, &(*d3dSurface),
        sharedHandle ? objectSharedHandle: 0);

      if ( FAILED(hr))
      {
        log_error("CreateOffscreenPlainSurface failed\n");
        return false;
      }
    }
    break;
  case CL_ADAPTER_D3D9EX_KHR:
    {
      surface = std::auto_ptr<CD3D9SurfaceWrapper>(new CD3D9SurfaceWrapper);
      CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
      HRESULT hr = 0;
      D3DFORMAT d3dFormat = SurfaceFormatToD3D(surfaceFormat);
      LPDIRECT3DDEVICE9EX d3d9ExDevice = (LPDIRECT3DDEVICE9EX)device.Device();
      hr = d3d9ExDevice->CreateOffscreenPlainSurface(width, height, d3dFormat, D3DPOOL_DEFAULT, &(*d3dSurface),
        sharedHandle ? objectSharedHandle: 0);

      if ( FAILED(hr))
      {
        log_error("CreateOffscreenPlainSurface failed\n");
        return false;
      }
    }
    break;
  case CL_ADAPTER_DXVA_KHR:
    {
      surface = std::auto_ptr<CD3D9SurfaceWrapper>(new CD3D9SurfaceWrapper);
      CD3D9SurfaceWrapper *d3dSurface = static_cast<CD3D9SurfaceWrapper *>(surface.get());
      HRESULT hr = 0;
      D3DFORMAT d3dFormat = SurfaceFormatToD3D(surfaceFormat);
      IDXVAHD_Device *dxvaDevice = (IDXVAHD_Device *)device.Device();
      hr = dxvaDevice->CreateVideoSurface(width, height, d3dFormat, D3DPOOL_DEFAULT, 0,
        DXVAHD_SURFACE_TYPE_VIDEO_INPUT,  1, &(*d3dSurface), sharedHandle ? objectSharedHandle: 0);

      if ( FAILED(hr))
      {
        log_error("CreateVideoSurface failed\n");
        return false;
      }
    }
    break;
#endif
  default:
    log_error("MediaSurfaceCreate(): Unknown adapter type!\n");
    return false;
    break;
  }

  return true;
}

cl_ushort float2half_rte( float f )
{
  union{ float f; cl_uint u; } u = {f};
  cl_uint sign = (u.u >> 16) & 0x8000;
  float x = fabsf(f);

  //Nan
  if( x != x )
  {
    u.u >>= (24-11);
    u.u &= 0x7fff;
    u.u |= 0x0200;      //silence the NaN
    return u.u | sign;
  }

  // overflow
  if( x >= MAKE_HEX_FLOAT(0x1.ffep15f, 0x1ffeL, 3) )
    return 0x7c00 | sign;

  // underflow
  if( x <= MAKE_HEX_FLOAT(0x1.0p-25f, 0x1L, -25) )
    return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

  // very small
  if( x < MAKE_HEX_FLOAT(0x1.8p-24f, 0x18L, -28) )
    return sign | 1;

  // half denormal
  if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
  {
    u.f = x * MAKE_HEX_FLOAT(0x1.0p-125f, 0x1L, -125);
    return sign | u.u;
  }

  u.f *= MAKE_HEX_FLOAT(0x1.0p13f, 0x1L, 13);
  u.u &= 0x7f800000;
  x += u.f;
  u.f = x - u.f;
  u.f *= MAKE_HEX_FLOAT(0x1.0p-112f, 0x1L, -112);

  return (u.u >> (24-11)) | sign;
}

cl_ushort float2half_rtz( float f )
{
  union{ float f; cl_uint u; } u = {f};
  cl_uint sign = (u.u >> 16) & 0x8000;
  float x = fabsf(f);

  //Nan
  if( x != x )
  {
    u.u >>= (24-11);
    u.u &= 0x7fff;
    u.u |= 0x0200;      //silence the NaN
    return u.u | sign;
  }

  // overflow
  if( x >= MAKE_HEX_FLOAT(0x1.0p16f, 0x1L, 16) )
  {
    if( x == INFINITY )
      return 0x7c00 | sign;

    return 0x7bff | sign;
  }

  // underflow
  if( x < MAKE_HEX_FLOAT(0x1.0p-24f, 0x1L, -24) )
    return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

  // half denormal
  if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
  {
    x *= MAKE_HEX_FLOAT(0x1.0p24f, 0x1L, 24);
    return (cl_ushort)((int) x | sign);
  }

  u.u &= 0xFFFFE000U;
  u.u -= 0x38000000U;

  return (u.u >> (24-11)) | sign;
}

cl_int deviceExistForCLTest(cl_platform_id platform,
     cl_dx9_media_adapter_type_khr media_adapters_type,
     void *media_adapters,
     CResult &result,
     TSharedHandleType sharedHandle /*default SHARED_HANDLE_ENABLED*/
     )
{
    cl_int _error;
    cl_uint devicesAllNum = 0;
    std::string sharedHandleStr = (sharedHandle == SHARED_HANDLE_ENABLED)? "yes": "no";
    std::string adapterStr;
    AdapterToString(media_adapters_type, adapterStr);

    _error = clGetDeviceIDsFromDX9MediaAdapterKHR(platform, 1,
        &media_adapters_type, &media_adapters, CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 0, 0, &devicesAllNum);

    if (_error != CL_SUCCESS)
    {
        if(_error != CL_DEVICE_NOT_FOUND)
        {
           log_error("clGetDeviceIDsFromDX9MediaAdapterKHR failed: %s\n", IGetErrorString(_error));
           result.ResultSub(CResult::TEST_ERROR);
        }
        else
        {
          log_info("Skipping test case, device type is not supported by a device (adapter type: %s, shared handle: %s)\n", adapterStr.c_str(), sharedHandleStr.c_str());
          result.ResultSub(CResult::TEST_NOTSUPPORTED);
        }
    }

    return _error;
}
