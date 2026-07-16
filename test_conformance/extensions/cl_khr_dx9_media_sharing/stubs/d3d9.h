#ifndef STUBS_D3D9_H
#define STUBS_D3D9_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _T
#define _T(x) x
#endif

typedef void* HANDLE;
typedef unsigned int UINT;
typedef const char* LPCTSTR;

typedef struct IDirect3DSurface9 IDirect3DSurface9;

typedef enum _D3DFORMAT
{
    D3DFMT_UNKNOWN = 0,
    D3DFMT_R8G8B8 = 20,
    D3DFMT_A8R8G8B8 = 21,
    D3DFMT_X8R8G8B8 = 22,
} D3DFORMAT;

#ifdef __cplusplus
}
#endif

#endif // STUBS_D3D9_H
