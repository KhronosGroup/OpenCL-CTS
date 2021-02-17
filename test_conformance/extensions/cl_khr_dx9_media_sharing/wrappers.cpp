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
#include "wrappers.h"
#include "harness/errorHelpers.h"

LPCTSTR CDeviceWrapper::WINDOW_TITLE = _T( "cl_khr_dx9_media_sharing" );
const int CDeviceWrapper::WINDOW_WIDTH = 256;
const int CDeviceWrapper::WINDOW_HEIGHT = 256;
CDeviceWrapper::TAccelerationType CDeviceWrapper::accelerationType = CDeviceWrapper::ACCELERATION_HW;

#if defined(_WIN32)
const D3DFORMAT CDXVAWrapper::RENDER_TARGET_FORMAT = D3DFMT_X8R8G8B8;
const D3DFORMAT CDXVAWrapper::VIDEO_FORMAT = D3DFMT_X8R8G8B8;
const unsigned int CDXVAWrapper::VIDEO_FPS = 60;
#endif

#if defined(_WIN32)
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
  switch(msg)
  {
  case WM_DESTROY:
    PostQuitMessage(0);
    return 0;
  case WM_PAINT:
    ValidateRect(hWnd, 0);
    return 0;
  default:
    break;
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
#endif

CDeviceWrapper::CDeviceWrapper()
#if defined(_WIN32)
:_hInstance(NULL),_hWnd(NULL)
#endif
{

}

void CDeviceWrapper::WindowInit()
{
#if defined(_WIN32)
  _hInstance = GetModuleHandle(NULL);
  static WNDCLASSEX wc =
  {
    sizeof(WNDCLASSEX),
    CS_CLASSDC,
    WndProc,
    0L,
    0L,
    _hInstance,
    NULL,
    NULL,
    NULL,
    NULL,
    WINDOW_TITLE,
    NULL
  };

  RegisterClassEx(&wc);

  _hWnd = CreateWindow(
    WINDOW_TITLE,
    WINDOW_TITLE,
    WS_OVERLAPPEDWINDOW,
    0, 0,
    WINDOW_WIDTH, WINDOW_HEIGHT,
    NULL,
    NULL,
    wc.hInstance,
    NULL);

  if (!_hWnd)
  {
    log_error("Failed to create window");
    return;
  }

  ShowWindow(_hWnd,SW_SHOWDEFAULT);
  UpdateWindow(_hWnd);
#endif
}

void CDeviceWrapper::WindowDestroy()
{
#if defined(_WIN32)
  if (_hWnd)
    DestroyWindow(_hWnd);
  _hWnd = NULL;
#endif
}

#if defined(_WIN32)
HWND CDeviceWrapper::WindowHandle() const
{
  return _hWnd;
}
#endif

int CDeviceWrapper::WindowWidth() const
{
  return WINDOW_WIDTH;
}

int CDeviceWrapper::WindowHeight() const
{
  return WINDOW_HEIGHT;
}

CDeviceWrapper::TAccelerationType CDeviceWrapper::AccelerationType()
{
  return accelerationType;
}

void CDeviceWrapper::AccelerationType( TAccelerationType accelerationTypeNew )
{
  accelerationType = accelerationTypeNew;
}

CDeviceWrapper::~CDeviceWrapper()
{
  WindowDestroy();
}

#if defined(_WIN32)
CD3D9Wrapper::CD3D9Wrapper():
_d3d9(NULL), _d3dDevice(NULL), _status(DEVICE_PASS), _adapterIdx(0), _adapterFound(false)
{
  WindowInit();

  _d3d9 = Direct3DCreate9(D3D_SDK_VERSION);
  if (!_d3d9)
  {
    log_error("Direct3DCreate9 failed\n");
    _status = DEVICE_FAIL;
  }
}

CD3D9Wrapper::~CD3D9Wrapper()
{
  Destroy();

  if(_d3d9)
    _d3d9->Release();
  _d3d9 = 0;
}

void CD3D9Wrapper::Destroy()
{
  if (_d3dDevice)
    _d3dDevice->Release();
  _d3dDevice = 0;
}

cl_int CD3D9Wrapper::Init()
{
  if (!WindowHandle())
  {
    log_error("D3D9: Window is not created\n");
    _status = DEVICE_FAIL;
    return DEVICE_FAIL;
  }

  if(!_d3d9 || DEVICE_PASS  != _status || !_adapterFound)
    return false;

  _d3d9->GetAdapterDisplayMode(_adapterIdx - 1, &_d3ddm);

  D3DPRESENT_PARAMETERS d3dParams;
  ZeroMemory(&d3dParams, sizeof(d3dParams));

  d3dParams.Windowed = TRUE;
  d3dParams.BackBufferCount = 1;
  d3dParams.SwapEffect = D3DSWAPEFFECT_DISCARD;
  d3dParams.hDeviceWindow = WindowHandle();
  d3dParams.BackBufferWidth = WindowWidth();
  d3dParams.BackBufferHeight = WindowHeight();
  d3dParams.BackBufferFormat = _d3ddm.Format;

  DWORD processingType = (AccelerationType() == ACCELERATION_HW)? D3DCREATE_HARDWARE_VERTEXPROCESSING:
    D3DCREATE_SOFTWARE_VERTEXPROCESSING;

  if ( FAILED( _d3d9->CreateDevice( _adapterIdx - 1, D3DDEVTYPE_HAL, WindowHandle(),
    processingType, &d3dParams, &_d3dDevice) ) )
  {
    log_error("CreateDevice failed\n");
    _status = DEVICE_FAIL;
    return DEVICE_FAIL;
  }

  _d3dDevice->BeginScene();
  _d3dDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);
  _d3dDevice->EndScene();

  return true;
}

void * CD3D9Wrapper::D3D() const
{
  return _d3d9;
}

void *CD3D9Wrapper::Device() const
{
  return _d3dDevice;
}

D3DFORMAT CD3D9Wrapper::Format()
{
  return _d3ddm.Format;
}

D3DADAPTER_IDENTIFIER9 CD3D9Wrapper::Adapter()
{
  return _adapter;
}

TDeviceStatus CD3D9Wrapper::Status() const
{
  return _status;
}

bool CD3D9Wrapper::AdapterNext()
{
  if (DEVICE_PASS != _status)
    return false;

  _adapterFound = false;
  for(; _adapterIdx < _d3d9->GetAdapterCount();)
  {
    ++_adapterIdx;
    D3DCAPS9 caps;
    if (FAILED(_d3d9->GetDeviceCaps(_adapterIdx - 1, D3DDEVTYPE_HAL, &caps)))
      continue;

    if(FAILED(_d3d9->GetAdapterIdentifier(_adapterIdx - 1, 0, &_adapter)))
    {
      log_error("D3D9: GetAdapterIdentifier failed\n");
      _status = DEVICE_FAIL;
      return false;
    }

    _adapterFound = true;

    Destroy();
    if(!Init())
    {
      _status = DEVICE_FAIL;
      _adapterFound = false;
    }
    break;
  }

  return _adapterFound;
}

unsigned int CD3D9Wrapper::AdapterIdx() const
{
  return _adapterIdx - 1;
}


CD3D9ExWrapper::CD3D9ExWrapper():
_d3d9Ex(NULL), _d3dDeviceEx(NULL), _status(DEVICE_PASS), _adapterIdx(0), _adapterFound(false)
{
  WindowInit();

  HRESULT result = Direct3DCreate9Ex(D3D_SDK_VERSION, &_d3d9Ex);
  if (FAILED(result) || !_d3d9Ex)
  {
    log_error("Direct3DCreate9Ex failed\n");
    _status = DEVICE_FAIL;
  }
}

CD3D9ExWrapper::~CD3D9ExWrapper()
{
  Destroy();

  if(_d3d9Ex)
    _d3d9Ex->Release();
  _d3d9Ex = 0;
}

void * CD3D9ExWrapper::D3D() const
{
  return _d3d9Ex;
}

void *CD3D9ExWrapper::Device() const
{
  return _d3dDeviceEx;
}

D3DFORMAT CD3D9ExWrapper::Format()
{
  return _d3ddmEx.Format;
}

D3DADAPTER_IDENTIFIER9 CD3D9ExWrapper::Adapter()
{
  return _adapter;
}

cl_int CD3D9ExWrapper::Init()
{
  if (!WindowHandle())
  {
    log_error("D3D9EX: Window is not created\n");
    _status = DEVICE_FAIL;
    return DEVICE_FAIL;
  }

  if(!_d3d9Ex || DEVICE_FAIL == _status || !_adapterFound)
    return DEVICE_FAIL;

  RECT rect;
  GetClientRect(WindowHandle(),&rect);

  D3DPRESENT_PARAMETERS d3dParams;
  ZeroMemory(&d3dParams, sizeof(d3dParams));

  d3dParams.Windowed = TRUE;
  d3dParams.SwapEffect = D3DSWAPEFFECT_FLIP;
  d3dParams.BackBufferFormat = D3DFMT_X8R8G8B8;
  d3dParams.BackBufferWidth = WindowWidth();
  d3dParams.BackBufferHeight = WindowHeight();

  d3dParams.BackBufferCount = 1;
  d3dParams.hDeviceWindow = WindowHandle();

  DWORD processingType = (AccelerationType() == ACCELERATION_HW)? D3DCREATE_HARDWARE_VERTEXPROCESSING:
    D3DCREATE_SOFTWARE_VERTEXPROCESSING;

  if ( FAILED( _d3d9Ex->CreateDeviceEx( _adapterIdx - 1, D3DDEVTYPE_HAL, WindowHandle(),
    processingType, &d3dParams, NULL, &_d3dDeviceEx) ) )
  {
    log_error("CreateDeviceEx failed\n");
    _status = DEVICE_FAIL;
    return DEVICE_FAIL;
  }

  _d3dDeviceEx->BeginScene();
  _d3dDeviceEx->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);
  _d3dDeviceEx->EndScene();

  return DEVICE_PASS;
}

void CD3D9ExWrapper::Destroy()
{
  if (_d3dDeviceEx)
    _d3dDeviceEx->Release();
  _d3dDeviceEx = 0;
}

TDeviceStatus CD3D9ExWrapper::Status() const
{
  return _status;
}

bool CD3D9ExWrapper::AdapterNext()
{
  if (DEVICE_FAIL == _status)
    return false;

  _adapterFound = false;
  for(; _adapterIdx < _d3d9Ex->GetAdapterCount();)
  {
    ++_adapterIdx;
    D3DCAPS9 caps;
    if (FAILED(_d3d9Ex->GetDeviceCaps(_adapterIdx - 1, D3DDEVTYPE_HAL, &caps)))
      continue;

    if(FAILED(_d3d9Ex->GetAdapterIdentifier(_adapterIdx - 1, 0, &_adapter)))
    {
      log_error("D3D9EX: GetAdapterIdentifier failed\n");
      _status = DEVICE_FAIL;
      return false;
    }

    _adapterFound = true;
    Destroy();
    if(!Init())
    {
      _status = DEVICE_FAIL;
      _adapterFound = _status;
    }

    break;
  }

  return _adapterFound;
}

unsigned int CD3D9ExWrapper::AdapterIdx() const
{
  return _adapterIdx - 1;
}

CDXVAWrapper::CDXVAWrapper():
_dxvaDevice(NULL), _status(DEVICE_PASS), _adapterFound(false)
{
  _status = _d3d9.Status();
}

CDXVAWrapper::~CDXVAWrapper()
{
  DXVAHDDestroy();
}

void * CDXVAWrapper::Device() const
{
  return _dxvaDevice;
}

TDeviceStatus CDXVAWrapper::Status() const
{
    if(_status == DEVICE_FAIL || _d3d9.Status() == DEVICE_FAIL)
        return DEVICE_FAIL;
    else if(_status == DEVICE_NOTSUPPORTED || _d3d9.Status() == DEVICE_NOTSUPPORTED)
        return DEVICE_NOTSUPPORTED;
    else
        return DEVICE_PASS;
}

bool CDXVAWrapper::AdapterNext()
{
  if (DEVICE_PASS != _status)
    return false;

  _adapterFound = _d3d9.AdapterNext();
  _status = _d3d9.Status();
  if (DEVICE_PASS != _status)
  {
    _adapterFound = false;
    return false;
  }

  if (!_adapterFound)
    return false;

  DXVAHDDestroy();
  _status = DXVAHDInit();
  if (DEVICE_PASS != _status)
  {
    _adapterFound = false;
    return false;
  }

  return true;
}

TDeviceStatus CDXVAWrapper::DXVAHDInit()
{
  if ((_status == DEVICE_FAIL) || (_d3d9.Status() == DEVICE_FAIL) || !_adapterFound)
    return DEVICE_FAIL;

  DXVAHD_RATIONAL fps = { VIDEO_FPS, 1 };

  DXVAHD_CONTENT_DESC desc;
  desc.InputFrameFormat= DXVAHD_FRAME_FORMAT_PROGRESSIVE;
  desc.InputFrameRate = fps;
  desc.InputWidth = WindowWidth();
  desc.InputHeight = WindowHeight();
  desc.OutputFrameRate = fps;
  desc.OutputWidth = WindowWidth();
  desc.OutputHeight = WindowHeight();

#ifdef USE_SOFTWARE_PLUGIN
  _status = DEVICE_FAIL;
  return DEVICE_FAIL;
#endif

  HRESULT hr = DXVAHD_CreateDevice(static_cast<IDirect3DDevice9Ex *>(_d3d9.Device()),
    &desc, DXVAHD_DEVICE_USAGE_PLAYBACK_NORMAL, NULL, &_dxvaDevice);
  if(FAILED(hr))
  {
    if (hr == E_NOINTERFACE)
    {
      log_error("DXVAHD_CreateDevice skipped due to no supported devices!\n");
      _status = DEVICE_NOTSUPPORTED;
    }
    else
    {
    log_error("DXVAHD_CreateDevice failed\n");
    _status = DEVICE_FAIL;
    }
  }

  return _status;
}

void CDXVAWrapper::DXVAHDDestroy()
{
  if (_dxvaDevice)
    _dxvaDevice->Release();
  _dxvaDevice = 0;
}

void * CDXVAWrapper::D3D() const
{
  return _d3d9.D3D();
}

unsigned int CDXVAWrapper::AdapterIdx() const
{
  return _d3d9.AdapterIdx();
}

const CD3D9ExWrapper & CDXVAWrapper::D3D9() const
{
  return _d3d9;
}

CD3D9SurfaceWrapper::CD3D9SurfaceWrapper():
mMem(NULL)
{

}

CD3D9SurfaceWrapper::CD3D9SurfaceWrapper( IDirect3DSurface9* mem ):
mMem(mem)
{

}

CD3D9SurfaceWrapper::~CD3D9SurfaceWrapper()
{
  if(mMem != NULL)
    mMem->Release();
  mMem = NULL;
}

#endif

CSurfaceWrapper::CSurfaceWrapper()
{

}

CSurfaceWrapper::~CSurfaceWrapper()
{

}
