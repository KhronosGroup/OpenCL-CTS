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
#ifndef __WRAPPERS_H
#define __WRAPPERS_H

#if defined(_WIN32)
#include <d3d9.h>
#if defined (__MINGW32__)
#include <rpcsal.h>
typedef unsigned char UINT8;
#define __out
#define __in
#define __inout
#define __out_bcount(size)
#define __out_bcount_opt(size)
#define __in_opt
#define __in_ecount(size)
#define __in_ecount_opt(size)
#define __out_opt
#define __out_ecount(size)
#define __out_ecount_opt(size)
#define __in_bcount_opt(size)
#define __inout_opt
#define __inout_bcount(size)
#define __in_bcount(size)
#define __deref_out
#endif
#include <dxvahd.h>
#include <tchar.h>
#endif

enum TDeviceStatus
{
  DEVICE_NOTSUPPORTED,
  DEVICE_PASS,
  DEVICE_FAIL,
};

class CDeviceWrapper {
public:
  enum TAccelerationType
  {
    ACCELERATION_HW,
    ACCELERATION_SW,
  };

  CDeviceWrapper();
  virtual ~CDeviceWrapper();

  virtual bool AdapterNext() = 0;
  virtual unsigned int AdapterIdx() const = 0;
  virtual void *Device() const = 0;
  virtual TDeviceStatus Status() const = 0;
  virtual void *D3D() const = 0;

#if defined(_WIN32)
  HWND WindowHandle() const;
#endif
  int WindowWidth() const;
  int WindowHeight() const;
  void WindowInit();


  static TAccelerationType AccelerationType();
  static void AccelerationType(TAccelerationType accelerationTypeNew);

private:
  static LPCTSTR WINDOW_TITLE;
  static const int WINDOW_WIDTH;
  static const int WINDOW_HEIGHT;
  static TAccelerationType accelerationType;

#if defined(_WIN32)
  HMODULE _hInstance;
  HWND _hWnd;
#endif

  void WindowDestroy();
};

class CSurfaceWrapper
{
public:
  CSurfaceWrapper();
  virtual ~CSurfaceWrapper();
};

#if defined(_WIN32)
//windows specific wrappers
class CD3D9Wrapper: public CDeviceWrapper {
public:
  CD3D9Wrapper();
  ~CD3D9Wrapper();

  virtual bool AdapterNext();
  virtual unsigned int AdapterIdx() const;
  virtual void *Device() const;
  virtual TDeviceStatus Status() const;
  virtual void *D3D() const;

private:
  LPDIRECT3D9 _d3d9;
  LPDIRECT3DDEVICE9 _d3dDevice;
  D3DDISPLAYMODE _d3ddm;
  D3DADAPTER_IDENTIFIER9 _adapter;
  TDeviceStatus _status;
  unsigned int _adapterIdx;
  bool _adapterFound;

  D3DFORMAT Format();
  D3DADAPTER_IDENTIFIER9 Adapter();
  int Init();
  void Destroy();
};

class CD3D9ExWrapper: public CDeviceWrapper {
public:
  CD3D9ExWrapper();
  ~CD3D9ExWrapper();

  virtual bool AdapterNext();
  virtual unsigned int AdapterIdx() const;
  virtual void *Device() const;
  virtual TDeviceStatus Status() const;
  virtual void *D3D() const;

private:
  LPDIRECT3D9EX _d3d9Ex;
  LPDIRECT3DDEVICE9EX _d3dDeviceEx;
  D3DDISPLAYMODEEX _d3ddmEx;
  D3DADAPTER_IDENTIFIER9 _adapter;
  TDeviceStatus _status;
  unsigned int _adapterIdx;
  bool _adapterFound;

  D3DFORMAT Format();
  D3DADAPTER_IDENTIFIER9 Adapter();
  int Init();
  void Destroy();
};

class CDXVAWrapper: public CDeviceWrapper {
public:
  CDXVAWrapper();
  ~CDXVAWrapper();

  virtual bool AdapterNext();
  virtual unsigned int AdapterIdx() const;
  virtual void *Device() const;
  virtual TDeviceStatus Status() const;
  virtual void *D3D() const;
  const CD3D9ExWrapper &D3D9() const;

private:
  CD3D9ExWrapper _d3d9;
  IDXVAHD_Device *_dxvaDevice;
  TDeviceStatus _status;
  bool _adapterFound;

  static const D3DFORMAT RENDER_TARGET_FORMAT;
  static const D3DFORMAT VIDEO_FORMAT;
  static const unsigned int VIDEO_FPS;

  TDeviceStatus DXVAHDInit();
  void DXVAHDDestroy();
};

class CD3D9SurfaceWrapper: public CSurfaceWrapper
{
public:
  CD3D9SurfaceWrapper();
  CD3D9SurfaceWrapper( IDirect3DSurface9* mem );
  ~CD3D9SurfaceWrapper();

  operator IDirect3DSurface9*() { return mMem; }
  IDirect3DSurface9* * operator&() { return &mMem; }
  IDirect3DSurface9* operator->() const { return mMem; }

private:
  IDirect3DSurface9* mMem;
};
#endif

#endif  // __D3D_WRAPPERS
