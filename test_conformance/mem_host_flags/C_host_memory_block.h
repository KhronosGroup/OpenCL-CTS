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
#ifndef test_conformance_cHost_MemoryBlock_h
#define test_conformance_cHost_MemoryBlock_h

#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template <class T> class C_host_memory_block {
public:
    int num_elements;
    int element_size;
    T *pData;

    C_host_memory_block();
    ~C_host_memory_block();
    void Init(int num_elem, T &value);
    void Init(int num_elem);
    void Set_to(T &val);
    void Set_to_zero();
    bool Equal_to(T &val);
    size_t Count(T &val);
    bool Equal(C_host_memory_block<T> &another);
    bool Equal_rect(C_host_memory_block<T> &another, size_t *host_origin,
                    size_t *region, size_t host_row_pitch,
                    size_t host_slice_pitch);
    bool Equal(T *pData, int num_elements);

    bool Equal_rect_from_orig(C_host_memory_block<T> &another, size_t *soffset,
                              size_t *region, size_t host_row_pitch,
                              size_t host_slice_pitch);

    bool Equal_rect_from_orig(T *another_pdata, size_t *soffset, size_t *region,
                              size_t host_row_pitch, size_t host_slice_pitch);
};

template <class T> C_host_memory_block<T>::C_host_memory_block()
{
    pData = NULL;
    element_size = sizeof(T);
    num_elements = 0;
}

template <class T> C_host_memory_block<T>::~C_host_memory_block()
{
    if (pData != NULL) delete[] pData;
    num_elements = 0;
}

template <class T> void C_host_memory_block<T>::Init(int num_elem, T &value)
{
    if (pData != NULL) delete[] pData;
    pData = new T[num_elem];
    for (int i = 0; i < num_elem; i++) pData[i] = value;

    num_elements = num_elem;
}

template <class T> void C_host_memory_block<T>::Init(int num_elem)
{
    if (pData != NULL) delete[] pData;
    pData = new T[num_elem];
    for (int i = 0; i < num_elem; i++) pData[i] = (T)i;

    num_elements = num_elem;
}
template <class T> void C_host_memory_block<T>::Set_to_zero()
{
    T v = 0;
    Set_to(v);
}

template <class T> void C_host_memory_block<T>::Set_to(T &val)
{
    for (int i = 0; i < num_elements; i++) pData[i] = val;
}

template <class T> bool C_host_memory_block<T>::Equal_to(T &val)
{
    int count = 0;

    for (int i = 0; i < num_elements; i++)
    {
        if (pData[i] == val) count++;
    }

    return (count == num_elements);
}

template <class T>
bool C_host_memory_block<T>::Equal(C_host_memory_block<T> &another)
{
    int count = 0;

    for (int i = 0; i < num_elements; i++)
    {
        if (pData[i] == another.pData[i]) count++;
    }

    return (count == num_elements);
}

template <class T>
bool C_host_memory_block<T>::Equal(T *pIn_Data, int Innum_elements)
{
    if (this->num_elements != Innum_elements) return false;

    int count = 0;

    for (int i = 0; i < num_elements; i++)
    {
        if (pData[i] == pIn_Data[i]) count++;
    }

    return (count == num_elements);
}

template <class T> size_t C_host_memory_block<T>::Count(T &val)
{
    size_t count = 0;
    for (int i = 0; i < num_elements; i++)
    {
        if (pData[i] == val) count++;
    }

    return count;
}

template <class T>
bool C_host_memory_block<T>::Equal_rect(C_host_memory_block<T> &another,
                                        size_t *soffset, size_t *region,
                                        size_t host_row_pitch,
                                        size_t host_slice_pitch)
{
    size_t row_pitch = host_row_pitch ? host_row_pitch : region[0];
    size_t slice_pitch = host_slice_pitch ? host_row_pitch : region[1];

    size_t count = 0;

    size_t total = region[0] * region[1] * region[2];

    size_t x, y, z;
    size_t orig = (size_t)(soffset[0] + row_pitch * soffset[1]
                           + slice_pitch * soffset[2]);
    for (z = 0; z < region[2]; z++)
        for (y = 0; y < region[1]; y++)
            for (x = 0; x < region[0]; x++)
            {
                int p1 = (int)(x + row_pitch * y + slice_pitch * z + orig);
                if (pData[p1] == another.pData[p1]) count++;
            }

    return (count == total);
}

template <class T>
bool C_host_memory_block<T>::Equal_rect_from_orig(
    C_host_memory_block<T> &another, size_t *soffset, size_t *region,
    size_t host_row_pitch, size_t host_slice_pitch)
{
    size_t row_pitch = host_row_pitch ? host_row_pitch : region[0];
    size_t slice_pitch = host_slice_pitch ? host_row_pitch : region[1];

    size_t count = 0;

    size_t total = region[0] * region[1] * region[2];

    size_t x, y, z;
    size_t orig =
        soffset[0] + row_pitch * soffset[1] + slice_pitch * soffset[2];
    for (z = 0; z < region[2]; z++)
        for (y = 0; y < region[1]; y++)
            for (x = 0; x < region[0]; x++)
            {
                size_t p1 = x + (row_pitch * y) + (slice_pitch * z);
                size_t p2 = p1 + orig;
                if (pData[p2] == another.pData[p1]) count++;
            }

    return (count == total);
}

template <class T>
bool C_host_memory_block<T>::Equal_rect_from_orig(T *another_pdata,
                                                  size_t *soffset,
                                                  size_t *region,
                                                  size_t host_row_pitch,
                                                  size_t host_slice_pitch)
{
    size_t row_pitch = host_row_pitch ? host_row_pitch : region[0];
    size_t slice_pitch = host_slice_pitch ? host_row_pitch : region[1];

    size_t count = 0;

    size_t total = region[0] * region[1] * region[2];

    size_t x, y, z;
    size_t orig =
        soffset[0] + row_pitch * soffset[1] + slice_pitch * soffset[2];
    for (z = 0; z < region[2]; z++)
        for (y = 0; y < region[1]; y++)
            for (x = 0; x < region[0]; x++)
            {
                size_t p1 = x + (row_pitch * y) + (slice_pitch * z);
                size_t p2 = p1 + orig;
                if (pData[p2] == another_pdata[p1]) count++;
            }

    return (count == total);
}

#endif
