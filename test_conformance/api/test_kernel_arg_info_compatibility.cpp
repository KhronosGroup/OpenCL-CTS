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
#include "testBase.h"
#include <limits.h>
#include <ctype.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#define ARG_INFO_FIELD_COUNT        5

#define ARG_INFO_ADDR_OFFSET        1
#define ARG_INFO_ACCESS_OFFSET        2
#define ARG_INFO_TYPE_QUAL_OFFSET    3
#define ARG_INFO_TYPE_NAME_OFFSET    4
#define ARG_INFO_ARG_NAME_OFFSET    5

typedef char const * kernel_args_t[];

static kernel_args_t required_kernel_args = {
    "typedef float4 typedef_type;\n"
    "\n"
    "typedef struct struct_type {\n"
    "    float4 float4d;\n"
    "    int intd;\n"
    "} typedef_struct_type;\n"
    "\n"
    "typedef union union_type {\n"
    "    float4 float4d;\n"
    "    uint4 uint4d;\n"
    "} typedef_union_type;\n"
    "\n"
    "typedef enum enum_type {\n"
    "    enum_type_zero,\n"
    "    enum_type_one,\n"
    "    enum_type_two\n"
    "} typedef_enum_type;\n"
    "\n"
    "kernel void constant_scalar_p0(constant void*constantvoidp,\n"
    "                              constant char *constantcharp,\n"
    "                              constant uchar* constantucharp,\n"
    "                              constant unsigned char * constantunsignedcharp)\n"
  "{}\n",
    "kernel void constant_scalar_p1(constant short*constantshortp,\n"
    "                              constant ushort *constantushortp,\n"
    "                              constant unsigned short* constantunsignedshortp,\n"
    "                              constant int * constantintp)\n"
  "{}\n",
    "kernel void constant_scalar_p2(constant uint*constantuintp,\n"
    "                              constant unsigned int *constantunsignedintp,\n"
    "                              constant long* constantlongp,\n"
    "                              constant ulong * constantulongp)\n"
  "{}\n",
    "kernel void constant_scalar_p3(constant unsigned long*constantunsignedlongp,\n"
    "                              constant float *constantfloatp)\n"
    "{}\n",
    "\n"
    "kernel void constant_scalar_restrict_p0(constant void* restrict constantvoidrestrictp,\n"
    "                                       constant char * restrict constantcharrestrictp,\n"
    "                                       constant uchar*restrict constantucharrestrictp,\n"
    "                                       constant unsigned char *restrict constantunsignedcharrestrictp)\n"
    "{}\n",
    "kernel void constant_scalar_restrict_p1(constant short* restrict constantshortrestrictp,\n"
    "                                       constant ushort * restrict constantushortrestrictp,\n"
    "                                       constant unsigned short*restrict constantunsignedshortrestrictp,\n"
    "                                       constant int *restrict constantintrestrictp)\n"
    "{}\n",
    "kernel void constant_scalar_restrict_p2(constant uint* restrict constantuintrestrictp,\n"
    "                                       constant unsigned int * restrict constantunsignedintrestrictp,\n"
    "                                       constant long*restrict constantlongrestrictp,\n"
    "                                       constant ulong *restrict constantulongrestrictp)\n"
    "{}\n",
    "kernel void constant_scalar_restrict_p3(constant unsigned long* restrict constantunsignedlongrestrictp,\n"
    "                                       constant float * restrict constantfloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_scalar_p(global void*globalvoidp,\n"
    "                            global char *globalcharp,\n"
    "                            global uchar* globalucharp,\n"
    "                            global unsigned char * globalunsignedcharp,\n"
    "                            global short*globalshortp,\n"
    "                            global ushort *globalushortp,\n"
    "                            global unsigned short* globalunsignedshortp,\n"
    "                            global int * globalintp,\n"
    "                            global uint*globaluintp,\n"
    "                            global unsigned int *globalunsignedintp,\n"
    "                            global long* globallongp,\n"
    "                            global ulong * globalulongp,\n"
    "                            global unsigned long*globalunsignedlongp,\n"
    "                            global float *globalfloatp)\n"
    "{}\n",
    "\n"
    "kernel void global_scalar_restrict_p(global void* restrict globalvoidrestrictp,\n"
    "                                     global char * restrict globalcharrestrictp,\n"
    "                                     global uchar*restrict globalucharrestrictp,\n"
    "                                     global unsigned char *restrict globalunsignedcharrestrictp,\n"
    "                                     global short* restrict globalshortrestrictp,\n"
    "                                     global ushort * restrict globalushortrestrictp,\n"
    "                                     global unsigned short*restrict globalunsignedshortrestrictp,\n"
    "                                     global int *restrict globalintrestrictp,\n"
    "                                     global uint* restrict globaluintrestrictp,\n"
    "                                     global unsigned int * restrict globalunsignedintrestrictp,\n"
    "                                     global long*restrict globallongrestrictp,\n"
    "                                     global ulong *restrict globalulongrestrictp,\n"
    "                                     global unsigned long* restrict globalunsignedlongrestrictp,\n"
    "                                     global float * restrict globalfloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_scalar_p(global const void*globalconstvoidp,\n"
    "                                  global const char *globalconstcharp,\n"
    "                                  global const uchar* globalconstucharp,\n"
    "                                  global const unsigned char * globalconstunsignedcharp,\n"
    "                                  global const short*globalconstshortp,\n"
    "                                  global const ushort *globalconstushortp,\n"
    "                                  global const unsigned short* globalconstunsignedshortp,\n"
    "                                  global const int * globalconstintp,\n"
    "                                  global const uint*globalconstuintp,\n"
    "                                  global const unsigned int *globalconstunsignedintp,\n"
    "                                  global const long* globalconstlongp,\n"
    "                                  global const ulong * globalconstulongp,\n"
    "                                  global const unsigned long*globalconstunsignedlongp,\n"
    "                                  global const float *globalconstfloatp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_scalar_restrict_p(global const void* restrict globalconstvoidrestrictp,\n"
    "                                           global const char * restrict globalconstcharrestrictp,\n"
    "                                           global const uchar*restrict globalconstucharrestrictp,\n"
    "                                           global const unsigned char *restrict globalconstunsignedcharrestrictp,\n"
    "                                           global const short* restrict globalconstshortrestrictp,\n"
    "                                           global const ushort * restrict globalconstushortrestrictp,\n"
    "                                           global const unsigned short*restrict globalconstunsignedshortrestrictp,\n"
    "                                           global const int *restrict globalconstintrestrictp,\n"
    "                                           global const uint* restrict globalconstuintrestrictp,\n"
    "                                           global const unsigned int * restrict globalconstunsignedintrestrictp,\n"
    "                                           global const long*restrict globalconstlongrestrictp,\n"
    "                                           global const ulong *restrict globalconstulongrestrictp,\n"
    "                                           global const unsigned long* restrict globalconstunsignedlongrestrictp,\n"
    "                                           global const float * restrict globalconstfloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_scalar_p(global volatile void*globalvolatilevoidp,\n"
    "                                     global volatile char *globalvolatilecharp,\n"
    "                                     global volatile uchar* globalvolatileucharp,\n"
    "                                     global volatile unsigned char * globalvolatileunsignedcharp,\n"
    "                                     global volatile short*globalvolatileshortp,\n"
    "                                     global volatile ushort *globalvolatileushortp,\n"
    "                                     global volatile unsigned short* globalvolatileunsignedshortp,\n"
    "                                     global volatile int * globalvolatileintp,\n"
    "                                     global volatile uint*globalvolatileuintp,\n"
    "                                     global volatile unsigned int *globalvolatileunsignedintp,\n"
    "                                     global volatile long* globalvolatilelongp,\n"
    "                                     global volatile ulong * globalvolatileulongp,\n"
    "                                     global volatile unsigned long*globalvolatileunsignedlongp,\n"
    "                                     global volatile float *globalvolatilefloatp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_scalar_restrict_p(global volatile void* restrict globalvolatilevoidrestrictp,\n"
    "                                              global volatile char * restrict globalvolatilecharrestrictp,\n"
    "                                              global volatile uchar*restrict globalvolatileucharrestrictp,\n"
    "                                              global volatile unsigned char *restrict globalvolatileunsignedcharrestrictp,\n"
    "                                              global volatile short* restrict globalvolatileshortrestrictp,\n"
    "                                              global volatile ushort * restrict globalvolatileushortrestrictp,\n"
    "                                              global volatile unsigned short*restrict globalvolatileunsignedshortrestrictp,\n"
    "                                              global volatile int *restrict globalvolatileintrestrictp,\n"
    "                                              global volatile uint* restrict globalvolatileuintrestrictp,\n"
    "                                              global volatile unsigned int * restrict globalvolatileunsignedintrestrictp,\n"
    "                                              global volatile long*restrict globalvolatilelongrestrictp,\n"
    "                                              global volatile ulong *restrict globalvolatileulongrestrictp,\n"
    "                                              global volatile unsigned long* restrict globalvolatileunsignedlongrestrictp,\n"
    "                                              global volatile float * restrict globalvolatilefloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_scalar_p(global const volatile void*globalconstvolatilevoidp,\n"
    "                                           global const volatile char *globalconstvolatilecharp,\n"
    "                                           global const volatile uchar* globalconstvolatileucharp,\n"
    "                                           global const volatile unsigned char * globalconstvolatileunsignedcharp,\n"
    "                                           global const volatile short*globalconstvolatileshortp,\n"
    "                                           global const volatile ushort *globalconstvolatileushortp,\n"
    "                                           global const volatile unsigned short* globalconstvolatileunsignedshortp,\n"
    "                                           global const volatile int * globalconstvolatileintp,\n"
    "                                           global const volatile uint*globalconstvolatileuintp,\n"
    "                                           global const volatile unsigned int *globalconstvolatileunsignedintp,\n"
    "                                           global const volatile long* globalconstvolatilelongp,\n"
    "                                           global const volatile ulong * globalconstvolatileulongp,\n"
    "                                           global const volatile unsigned long*globalconstvolatileunsignedlongp,\n"
    "                                           global const volatile float *globalconstvolatilefloatp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_scalar_restrict_p(global const volatile void* restrict globalconstvolatilevoidrestrictp,\n"
    "                                                    global const volatile char * restrict globalconstvolatilecharrestrictp,\n"
    "                                                    global const volatile uchar*restrict globalconstvolatileucharrestrictp,\n"
    "                                                    global const volatile unsigned char *restrict globalconstvolatileunsignedcharrestrictp,\n"
    "                                                    global const volatile short* restrict globalconstvolatileshortrestrictp,\n"
    "                                                    global const volatile ushort * restrict globalconstvolatileushortrestrictp,\n"
    "                                                    global const volatile unsigned short*restrict globalconstvolatileunsignedshortrestrictp,\n"
    "                                                    global const volatile int *restrict globalconstvolatileintrestrictp,\n"
    "                                                    global const volatile uint* restrict globalconstvolatileuintrestrictp,\n"
    "                                                    global const volatile unsigned int * restrict globalconstvolatileunsignedintrestrictp,\n"
    "                                                    global const volatile long*restrict globalconstvolatilelongrestrictp,\n"
    "                                                    global const volatile ulong *restrict globalconstvolatileulongrestrictp,\n"
    "                                                    global const volatile unsigned long* restrict globalconstvolatileunsignedlongrestrictp,\n"
    "                                                    global const volatile float * restrict globalconstvolatilefloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_scalar_p(local void*localvoidp,\n"
    "                           local char *localcharp,\n"
    "                           local uchar* localucharp,\n"
    "                           local unsigned char * localunsignedcharp,\n"
    "                           local short*localshortp,\n"
    "                           local ushort *localushortp,\n"
    "                           local unsigned short* localunsignedshortp,\n"
    "                           local int * localintp,\n"
    "                           local uint*localuintp,\n"
    "                           local unsigned int *localunsignedintp,\n"
    "                           local long* locallongp,\n"
    "                           local ulong * localulongp,\n"
    "                           local unsigned long*localunsignedlongp,\n"
    "                           local float *localfloatp)\n"
    "{}\n",
    "\n"
    "kernel void local_scalar_restrict_p(local void* restrict localvoidrestrictp,\n"
    "                                    local char * restrict localcharrestrictp,\n"
    "                                    local uchar*restrict localucharrestrictp,\n"
    "                                    local unsigned char *restrict localunsignedcharrestrictp,\n"
    "                                    local short* restrict localshortrestrictp,\n"
    "                                    local ushort * restrict localushortrestrictp,\n"
    "                                    local unsigned short*restrict localunsignedshortrestrictp,\n"
    "                                    local int *restrict localintrestrictp,\n"
    "                                    local uint* restrict localuintrestrictp,\n"
    "                                    local unsigned int * restrict localunsignedintrestrictp,\n"
    "                                    local long*restrict locallongrestrictp,\n"
    "                                    local ulong *restrict localulongrestrictp,\n"
    "                                    local unsigned long* restrict localunsignedlongrestrictp,\n"
    "                                    local float * restrict localfloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_scalar_p(local const void*localconstvoidp,\n"
    "                                 local const char *localconstcharp,\n"
    "                                 local const uchar* localconstucharp,\n"
    "                                 local const unsigned char * localconstunsignedcharp,\n"
    "                                 local const short*localconstshortp,\n"
    "                                 local const ushort *localconstushortp,\n"
    "                                 local const unsigned short* localconstunsignedshortp,\n"
    "                                 local const int * localconstintp,\n"
    "                                 local const uint*localconstuintp,\n"
    "                                 local const unsigned int *localconstunsignedintp,\n"
    "                                 local const long* localconstlongp,\n"
    "                                 local const ulong * localconstulongp,\n"
    "                                 local const unsigned long*localconstunsignedlongp,\n"
    "                                 local const float *localconstfloatp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_scalar_restrict_p(local const void* restrict localconstvoidrestrictp,\n"
    "                                          local const char * restrict localconstcharrestrictp,\n"
    "                                          local const uchar*restrict localconstucharrestrictp,\n"
    "                                          local const unsigned char *restrict localconstunsignedcharrestrictp,\n"
    "                                          local const short* restrict localconstshortrestrictp,\n"
    "                                          local const ushort * restrict localconstushortrestrictp,\n"
    "                                          local const unsigned short*restrict localconstunsignedshortrestrictp,\n"
    "                                          local const int *restrict localconstintrestrictp,\n"
    "                                          local const uint* restrict localconstuintrestrictp,\n"
    "                                          local const unsigned int * restrict localconstunsignedintrestrictp,\n"
    "                                          local const long*restrict localconstlongrestrictp,\n"
    "                                          local const ulong *restrict localconstulongrestrictp,\n"
    "                                          local const unsigned long* restrict localconstunsignedlongrestrictp,\n"
    "                                          local const float * restrict localconstfloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_scalar_p(local volatile void*localvolatilevoidp,\n"
    "                                    local volatile char *localvolatilecharp,\n"
    "                                    local volatile uchar* localvolatileucharp,\n"
    "                                    local volatile unsigned char * localvolatileunsignedcharp,\n"
    "                                    local volatile short*localvolatileshortp,\n"
    "                                    local volatile ushort *localvolatileushortp,\n"
    "                                    local volatile unsigned short* localvolatileunsignedshortp,\n"
    "                                    local volatile int * localvolatileintp,\n"
    "                                    local volatile uint*localvolatileuintp,\n"
    "                                    local volatile unsigned int *localvolatileunsignedintp,\n"
    "                                    local volatile long* localvolatilelongp,\n"
    "                                    local volatile ulong * localvolatileulongp,\n"
    "                                    local volatile unsigned long*localvolatileunsignedlongp,\n"
    "                                    local volatile float *localvolatilefloatp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_scalar_restrict_p(local volatile void* restrict localvolatilevoidrestrictp,\n"
    "                                             local volatile char * restrict localvolatilecharrestrictp,\n"
    "                                             local volatile uchar*restrict localvolatileucharrestrictp,\n"
    "                                             local volatile unsigned char *restrict localvolatileunsignedcharrestrictp,\n"
    "                                             local volatile short* restrict localvolatileshortrestrictp,\n"
    "                                             local volatile ushort * restrict localvolatileushortrestrictp,\n"
    "                                             local volatile unsigned short*restrict localvolatileunsignedshortrestrictp,\n"
    "                                             local volatile int *restrict localvolatileintrestrictp,\n"
    "                                             local volatile uint* restrict localvolatileuintrestrictp,\n"
    "                                             local volatile unsigned int * restrict localvolatileunsignedintrestrictp,\n"
    "                                             local volatile long*restrict localvolatilelongrestrictp,\n"
    "                                             local volatile ulong *restrict localvolatileulongrestrictp,\n"
    "                                             local volatile unsigned long* restrict localvolatileunsignedlongrestrictp,\n"
    "                                             local volatile float * restrict localvolatilefloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_scalar_p(local const volatile void*localconstvolatilevoidp,\n"
    "                                          local const volatile char *localconstvolatilecharp,\n"
    "                                          local const volatile uchar* localconstvolatileucharp,\n"
    "                                          local const volatile unsigned char * localconstvolatileunsignedcharp,\n"
    "                                          local const volatile short*localconstvolatileshortp,\n"
    "                                          local const volatile ushort *localconstvolatileushortp,\n"
    "                                          local const volatile unsigned short* localconstvolatileunsignedshortp,\n"
    "                                          local const volatile int * localconstvolatileintp,\n"
    "                                          local const volatile uint*localconstvolatileuintp,\n"
    "                                          local const volatile unsigned int *localconstvolatileunsignedintp,\n"
    "                                          local const volatile long* localconstvolatilelongp,\n"
    "                                          local const volatile ulong * localconstvolatileulongp,\n"
    "                                          local const volatile unsigned long*localconstvolatileunsignedlongp,\n"
    "                                          local const volatile float *localconstvolatilefloatp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_scalar_restrict_p(local const volatile void* restrict localconstvolatilevoidrestrictp,\n"
    "                                                   local const volatile char * restrict localconstvolatilecharrestrictp,\n"
    "                                                   local const volatile uchar*restrict localconstvolatileucharrestrictp,\n"
    "                                                   local const volatile unsigned char *restrict localconstvolatileunsignedcharrestrictp,\n"
    "                                                   local const volatile short* restrict localconstvolatileshortrestrictp,\n"
    "                                                   local const volatile ushort * restrict localconstvolatileushortrestrictp,\n"
    "                                                   local const volatile unsigned short*restrict localconstvolatileunsignedshortrestrictp,\n"
    "                                                   local const volatile int *restrict localconstvolatileintrestrictp,\n"
    "                                                   local const volatile uint* restrict localconstvolatileuintrestrictp,\n"
    "                                                   local const volatile unsigned int * restrict localconstvolatileunsignedintrestrictp,\n"
    "                                                   local const volatile long*restrict localconstvolatilelongrestrictp,\n"
    "                                                   local const volatile ulong *restrict localconstvolatileulongrestrictp,\n"
    "                                                   local const volatile unsigned long* restrict localconstvolatileunsignedlongrestrictp,\n"
    "                                                   local const volatile float * restrict localconstvolatilefloatrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void scalar_d(char chard,\n"
    "                     uchar uchard,\n"
    "                     unsigned char unsignedchard,\n"
    "                     short shortd,\n"
    "                     ushort ushortd,\n"
    "                     unsigned short unsignedshortd,\n"
    "                     int intd,\n"
    "                     uint uintd,\n"
    "                     unsigned int unsignedintd,\n"
    "                     long longd,\n"
    "                     ulong ulongd,\n"
    "                     unsigned long unsignedlongd,\n"
    "                     float floatd)\n"
    "{}\n",
    "\n"
    "kernel void const_scalar_d(const char constchard,\n"
    "                           const uchar constuchard,\n"
    "                           const unsigned char constunsignedchard,\n"
    "                           const short constshortd,\n"
    "                           const ushort constushortd,\n"
    "                           const unsigned short constunsignedshortd,\n"
    "                           const int constintd,\n"
    "                           const uint constuintd,\n"
    "                           const unsigned int constunsignedintd,\n"
    "                           const long constlongd,\n"
    "                           const ulong constulongd,\n"
    "                           const unsigned long constunsignedlongd,\n"
    "                           const float constfloatd)\n"
    "{}\n",
    "\n"
    "kernel void private_scalar_d(private char privatechard,\n"
    "                             private uchar privateuchard,\n"
    "                             private unsigned char privateunsignedchard,\n"
    "                             private short privateshortd,\n"
    "                             private ushort privateushortd,\n"
    "                             private unsigned short privateunsignedshortd,\n"
    "                             private int privateintd,\n"
    "                             private uint privateuintd,\n"
    "                             private unsigned int privateunsignedintd,\n"
    "                             private long privatelongd,\n"
    "                             private ulong privateulongd,\n"
    "                             private unsigned long privateunsignedlongd,\n"
    "                             private float privatefloatd)\n"
    "{}\n",
    "\n"
    "kernel void private_const_scalar_d(private const char privateconstchard,\n"
    "                                   private const uchar privateconstuchard,\n"
    "                                   private const unsigned char privateconstunsignedchard,\n"
    "                                   private const short privateconstshortd,\n"
    "                                   private const ushort privateconstushortd,\n"
    "                                   private const unsigned short privateconstunsignedshortd,\n"
    "                                   private const int privateconstintd,\n"
    "                                   private const uint privateconstuintd,\n"
    "                                   private const unsigned int privateconstunsignedintd,\n"
    "                                   private const long privateconstlongd,\n"
    "                                   private const ulong privateconstulongd,\n"
    "                                   private const unsigned long privateconstunsignedlongd,\n"
    "                                   private const float privateconstfloatd)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector2_p0(constant char2*constantchar2p,\n"
    "                               constant uchar2 *constantuchar2p,\n"
    "                               constant short2* constantshort2p,\n"
    "                               constant ushort2 * constantushort2p)\n"
  "{}\n",
    "\n"
    "kernel void constant_vector2_p1(constant int2*constantint2p,\n"
    "                               constant uint2 *constantuint2p,\n"
    "                               constant long2* constantlong2p,\n"
    "                               constant ulong2 * constantulong2p)\n"
  "{}\n",
    "\n"
    "kernel void constant_vector2_p2(constant float2*constantfloat2p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector2_restrict_p0(constant char2 *restrict constantchar2restrictp,\n"
    "                                        constant uchar2* restrict constantuchar2restrictp,\n"
    "                                        constant short2 * restrict constantshort2restrictp,\n"
    "                                        constant ushort2*restrict constantushort2restrictp)\n"
  "{}\n",
    "\n"
    "kernel void constant_vector2_restrict_p1(constant int2 *restrict constantint2restrictp,\n"
    "                                        constant uint2* restrict constantuint2restrictp,\n"
    "                                        constant long2 * restrict constantlong2restrictp,\n"
    "                                        constant ulong2*restrict constantulong2restrictp)\n"
  "{}\n",
    "\n"
    "kernel void constant_vector2_restrict_p2(constant float2 *restrict constantfloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_vector2_p(global char2*globalchar2p,\n"
    "                             global uchar2 *globaluchar2p,\n"
    "                             global short2* globalshort2p,\n"
    "                             global ushort2 * globalushort2p,\n"
    "                             global int2*globalint2p,\n"
    "                             global uint2 *globaluint2p,\n"
    "                             global long2* globallong2p,\n"
    "                             global ulong2 * globalulong2p,\n"
    "                             global float2*globalfloat2p)\n"
    "{}\n",
    "\n"
    "kernel void global_vector2_restrict_p(global char2 *restrict globalchar2restrictp,\n"
    "                                      global uchar2* restrict globaluchar2restrictp,\n"
    "                                      global short2 * restrict globalshort2restrictp,\n"
    "                                      global ushort2*restrict globalushort2restrictp,\n"
    "                                      global int2 *restrict globalint2restrictp,\n"
    "                                      global uint2* restrict globaluint2restrictp,\n"
    "                                      global long2 * restrict globallong2restrictp,\n"
    "                                      global ulong2*restrict globalulong2restrictp,\n"
    "                                      global float2 *restrict globalfloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector2_p(global const char2* globalconstchar2p,\n"
    "                                   global const uchar2 * globalconstuchar2p,\n"
    "                                   global const short2*globalconstshort2p,\n"
    "                                   global const ushort2 *globalconstushort2p,\n"
    "                                   global const int2* globalconstint2p,\n"
    "                                   global const uint2 * globalconstuint2p,\n"
    "                                   global const long2*globalconstlong2p,\n"
    "                                   global const ulong2 *globalconstulong2p,\n"
    "                                   global const float2* globalconstfloat2p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector2_restrict_p(global const char2 * restrict globalconstchar2restrictp,\n"
    "                                            global const uchar2*restrict globalconstuchar2restrictp,\n"
    "                                            global const short2 *restrict globalconstshort2restrictp,\n"
    "                                            global const ushort2* restrict globalconstushort2restrictp,\n"
    "                                            global const int2 * restrict globalconstint2restrictp,\n"
    "                                            global const uint2*restrict globalconstuint2restrictp,\n"
    "                                            global const long2 *restrict globalconstlong2restrictp,\n"
    "                                            global const ulong2* restrict globalconstulong2restrictp,\n"
    "                                            global const float2 * restrict globalconstfloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector2_p(global volatile char2*globalvolatilechar2p,\n"
    "                                      global volatile uchar2 *globalvolatileuchar2p,\n"
    "                                      global volatile short2* globalvolatileshort2p,\n"
    "                                      global volatile ushort2 * globalvolatileushort2p,\n"
    "                                      global volatile int2*globalvolatileint2p,\n"
    "                                      global volatile uint2 *globalvolatileuint2p,\n"
    "                                      global volatile long2* globalvolatilelong2p,\n"
    "                                      global volatile ulong2 * globalvolatileulong2p,\n"
    "                                      global volatile float2*globalvolatilefloat2p)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector2_restrict_p(global volatile char2 *restrict globalvolatilechar2restrictp,\n"
    "                                               global volatile uchar2* restrict globalvolatileuchar2restrictp,\n"
    "                                               global volatile short2 * restrict globalvolatileshort2restrictp,\n"
    "                                               global volatile ushort2*restrict globalvolatileushort2restrictp,\n"
    "                                               global volatile int2 *restrict globalvolatileint2restrictp,\n"
    "                                               global volatile uint2* restrict globalvolatileuint2restrictp,\n"
    "                                               global volatile long2 * restrict globalvolatilelong2restrictp,\n"
    "                                               global volatile ulong2*restrict globalvolatileulong2restrictp,\n"
    "                                               global volatile float2 *restrict globalvolatilefloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector2_p(global const volatile char2* globalconstvolatilechar2p,\n"
    "                                            global const volatile uchar2 * globalconstvolatileuchar2p,\n"
    "                                            global const volatile short2*globalconstvolatileshort2p,\n"
    "                                            global const volatile ushort2 *globalconstvolatileushort2p,\n"
    "                                            global const volatile int2* globalconstvolatileint2p,\n"
    "                                            global const volatile uint2 * globalconstvolatileuint2p,\n"
    "                                            global const volatile long2*globalconstvolatilelong2p,\n"
    "                                            global const volatile ulong2 *globalconstvolatileulong2p,\n"
    "                                            global const volatile float2* globalconstvolatilefloat2p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector2_restrict_p(global const volatile char2 * restrict globalconstvolatilechar2restrictp,\n"
    "                                                     global const volatile uchar2*restrict globalconstvolatileuchar2restrictp,\n"
    "                                                     global const volatile short2 *restrict globalconstvolatileshort2restrictp,\n"
    "                                                     global const volatile ushort2* restrict globalconstvolatileushort2restrictp,\n"
    "                                                     global const volatile int2 * restrict globalconstvolatileint2restrictp,\n"
    "                                                     global const volatile uint2*restrict globalconstvolatileuint2restrictp,\n"
    "                                                     global const volatile long2 *restrict globalconstvolatilelong2restrictp,\n"
    "                                                     global const volatile ulong2* restrict globalconstvolatileulong2restrictp,\n"
    "                                                     global const volatile float2 * restrict globalconstvolatilefloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_vector2_p(local char2*localchar2p,\n"
    "                            local uchar2 *localuchar2p,\n"
    "                            local short2* localshort2p,\n"
    "                            local ushort2 * localushort2p,\n"
    "                            local int2*localint2p,\n"
    "                            local uint2 *localuint2p,\n"
    "                            local long2* locallong2p,\n"
    "                            local ulong2 * localulong2p,\n"
    "                            local float2*localfloat2p)\n"
    "{}\n",
    "\n"
    "kernel void local_vector2_restrict_p(local char2 *restrict localchar2restrictp,\n"
    "                                     local uchar2* restrict localuchar2restrictp,\n"
    "                                     local short2 * restrict localshort2restrictp,\n"
    "                                     local ushort2*restrict localushort2restrictp,\n"
    "                                     local int2 *restrict localint2restrictp,\n"
    "                                     local uint2* restrict localuint2restrictp,\n"
    "                                     local long2 * restrict locallong2restrictp,\n"
    "                                     local ulong2*restrict localulong2restrictp,\n"
    "                                     local float2 *restrict localfloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector2_p(local const char2* localconstchar2p,\n"
    "                                  local const uchar2 * localconstuchar2p,\n"
    "                                  local const short2*localconstshort2p,\n"
    "                                  local const ushort2 *localconstushort2p,\n"
    "                                  local const int2* localconstint2p,\n"
    "                                  local const uint2 * localconstuint2p,\n"
    "                                  local const long2*localconstlong2p,\n"
    "                                  local const ulong2 *localconstulong2p,\n"
    "                                  local const float2* localconstfloat2p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector2_restrict_p(local const char2 * restrict localconstchar2restrictp,\n"
    "                                           local const uchar2*restrict localconstuchar2restrictp,\n"
    "                                           local const short2 *restrict localconstshort2restrictp,\n"
    "                                           local const ushort2* restrict localconstushort2restrictp,\n"
    "                                           local const int2 * restrict localconstint2restrictp,\n"
    "                                           local const uint2*restrict localconstuint2restrictp,\n"
    "                                           local const long2 *restrict localconstlong2restrictp,\n"
    "                                           local const ulong2* restrict localconstulong2restrictp,\n"
    "                                           local const float2 * restrict localconstfloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector2_p(local volatile char2*localvolatilechar2p,\n"
    "                                     local volatile uchar2 *localvolatileuchar2p,\n"
    "                                     local volatile short2* localvolatileshort2p,\n"
    "                                     local volatile ushort2 * localvolatileushort2p,\n"
    "                                     local volatile int2*localvolatileint2p,\n"
    "                                     local volatile uint2 *localvolatileuint2p,\n"
    "                                     local volatile long2* localvolatilelong2p,\n"
    "                                     local volatile ulong2 * localvolatileulong2p,\n"
    "                                     local volatile float2*localvolatilefloat2p)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector2_restrict_p(local volatile char2 *restrict localvolatilechar2restrictp,\n"
    "                                              local volatile uchar2* restrict localvolatileuchar2restrictp,\n"
    "                                              local volatile short2 * restrict localvolatileshort2restrictp,\n"
    "                                              local volatile ushort2*restrict localvolatileushort2restrictp,\n"
    "                                              local volatile int2 *restrict localvolatileint2restrictp,\n"
    "                                              local volatile uint2* restrict localvolatileuint2restrictp,\n"
    "                                              local volatile long2 * restrict localvolatilelong2restrictp,\n"
    "                                              local volatile ulong2*restrict localvolatileulong2restrictp,\n"
    "                                              local volatile float2 *restrict localvolatilefloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector2_p(local const volatile char2* localconstvolatilechar2p,\n"
    "                                           local const volatile uchar2 * localconstvolatileuchar2p,\n"
    "                                           local const volatile short2*localconstvolatileshort2p,\n"
    "                                           local const volatile ushort2 *localconstvolatileushort2p,\n"
    "                                           local const volatile int2* localconstvolatileint2p,\n"
    "                                           local const volatile uint2 * localconstvolatileuint2p,\n"
    "                                           local const volatile long2*localconstvolatilelong2p,\n"
    "                                           local const volatile ulong2 *localconstvolatileulong2p,\n"
    "                                           local const volatile float2* localconstvolatilefloat2p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector2_restrict_p(local const volatile char2 * restrict localconstvolatilechar2restrictp,\n"
    "                                                    local const volatile uchar2*restrict localconstvolatileuchar2restrictp,\n"
    "                                                    local const volatile short2 *restrict localconstvolatileshort2restrictp,\n"
    "                                                    local const volatile ushort2* restrict localconstvolatileushort2restrictp,\n"
    "                                                    local const volatile int2 * restrict localconstvolatileint2restrictp,\n"
    "                                                    local const volatile uint2*restrict localconstvolatileuint2restrictp,\n"
    "                                                    local const volatile long2 *restrict localconstvolatilelong2restrictp,\n"
    "                                                    local const volatile ulong2* restrict localconstvolatileulong2restrictp,\n"
    "                                                    local const volatile float2 * restrict localconstvolatilefloat2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void vector2_d(char2 char2d,\n"
    "                      uchar2 uchar2d,\n"
    "                      short2 short2d,\n"
    "                      ushort2 ushort2d,\n"
    "                      int2 int2d,\n"
    "                      uint2 uint2d,\n"
    "                      long2 long2d,\n"
    "                      ulong2 ulong2d,\n"
    "                      float2 float2d)\n"
    "{}\n",
    "\n"
    "kernel void const_vector2_d(const char2 constchar2d,\n"
    "                            const uchar2 constuchar2d,\n"
    "                            const short2 constshort2d,\n"
    "                            const ushort2 constushort2d,\n"
    "                            const int2 constint2d,\n"
    "                            const uint2 constuint2d,\n"
    "                            const long2 constlong2d,\n"
    "                            const ulong2 constulong2d,\n"
    "                            const float2 constfloat2d)\n"
    "{}\n",
    "\n"
    "kernel void private_vector2_d(private char2 privatechar2d,\n"
    "                              private uchar2 privateuchar2d,\n"
    "                              private short2 privateshort2d,\n"
    "                              private ushort2 privateushort2d,\n"
    "                              private int2 privateint2d,\n"
    "                              private uint2 privateuint2d,\n"
    "                              private long2 privatelong2d,\n"
    "                              private ulong2 privateulong2d,\n"
    "                              private float2 privatefloat2d)\n"
    "{}\n",
    "\n"
    "kernel void private_const_vector2_d(private const char2 privateconstchar2d,\n"
    "                                    private const uchar2 privateconstuchar2d,\n"
    "                                    private const short2 privateconstshort2d,\n"
    "                                    private const ushort2 privateconstushort2d,\n"
    "                                    private const int2 privateconstint2d,\n"
    "                                    private const uint2 privateconstuint2d,\n"
    "                                    private const long2 privateconstlong2d,\n"
    "                                    private const ulong2 privateconstulong2d,\n"
    "                                    private const float2 privateconstfloat2d)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector3_p0(constant char3*constantchar3p,\n"
    "                               constant uchar3 *constantuchar3p,\n"
    "                               constant short3* constantshort3p,\n"
    "                               constant ushort3 * constantushort3p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector3_p1(constant int3*constantint3p,\n"
    "                               constant uint3 *constantuint3p,\n"
    "                               constant long3* constantlong3p,\n"
    "                               constant ulong3 * constantulong3p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector3_p2(constant float3*constantfloat3p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector3_restrict_p0(constant char3 *restrict constantchar3restrictp,\n"
    "                                        constant uchar3* restrict constantuchar3restrictp,\n"
    "                                        constant short3 * restrict constantshort3restrictp,\n"
    "                                        constant ushort3*restrict constantushort3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector3_restrict_p1(constant int3 *restrict constantint3restrictp,\n"
    "                                        constant uint3* restrict constantuint3restrictp,\n"
    "                                        constant long3 * restrict constantlong3restrictp,\n"
    "                                        constant ulong3*restrict constantulong3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector3_restrict_p2(constant float3 *restrict constantfloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_vector3_p(global char3*globalchar3p,\n"
    "                             global uchar3 *globaluchar3p,\n"
    "                             global short3* globalshort3p,\n"
    "                             global ushort3 * globalushort3p,\n"
    "                             global int3*globalint3p,\n"
    "                             global uint3 *globaluint3p,\n"
    "                             global long3* globallong3p,\n"
    "                             global ulong3 * globalulong3p,\n"
    "                             global float3*globalfloat3p)\n"
    "{}\n",
    "\n"
    "kernel void global_vector3_restrict_p(global char3 *restrict globalchar3restrictp,\n"
    "                                      global uchar3* restrict globaluchar3restrictp,\n"
    "                                      global short3 * restrict globalshort3restrictp,\n"
    "                                      global ushort3*restrict globalushort3restrictp,\n"
    "                                      global int3 *restrict globalint3restrictp,\n"
    "                                      global uint3* restrict globaluint3restrictp,\n"
    "                                      global long3 * restrict globallong3restrictp,\n"
    "                                      global ulong3*restrict globalulong3restrictp,\n"
    "                                      global float3 *restrict globalfloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector3_p(global const char3* globalconstchar3p,\n"
    "                                   global const uchar3 * globalconstuchar3p,\n"
    "                                   global const short3*globalconstshort3p,\n"
    "                                   global const ushort3 *globalconstushort3p,\n"
    "                                   global const int3* globalconstint3p,\n"
    "                                   global const uint3 * globalconstuint3p,\n"
    "                                   global const long3*globalconstlong3p,\n"
    "                                   global const ulong3 *globalconstulong3p,\n"
    "                                   global const float3* globalconstfloat3p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector3_restrict_p(global const char3 * restrict globalconstchar3restrictp,\n"
    "                                            global const uchar3*restrict globalconstuchar3restrictp,\n"
    "                                            global const short3 *restrict globalconstshort3restrictp,\n"
    "                                            global const ushort3* restrict globalconstushort3restrictp,\n"
    "                                            global const int3 * restrict globalconstint3restrictp,\n"
    "                                            global const uint3*restrict globalconstuint3restrictp,\n"
    "                                            global const long3 *restrict globalconstlong3restrictp,\n"
    "                                            global const ulong3* restrict globalconstulong3restrictp,\n"
    "                                            global const float3 * restrict globalconstfloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector3_p(global volatile char3*globalvolatilechar3p,\n"
    "                                      global volatile uchar3 *globalvolatileuchar3p,\n"
    "                                      global volatile short3* globalvolatileshort3p,\n"
    "                                      global volatile ushort3 * globalvolatileushort3p,\n"
    "                                      global volatile int3*globalvolatileint3p,\n"
    "                                      global volatile uint3 *globalvolatileuint3p,\n"
    "                                      global volatile long3* globalvolatilelong3p,\n"
    "                                      global volatile ulong3 * globalvolatileulong3p,\n"
    "                                      global volatile float3*globalvolatilefloat3p)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector3_restrict_p(global volatile char3 *restrict globalvolatilechar3restrictp,\n"
    "                                               global volatile uchar3* restrict globalvolatileuchar3restrictp,\n"
    "                                               global volatile short3 * restrict globalvolatileshort3restrictp,\n"
    "                                               global volatile ushort3*restrict globalvolatileushort3restrictp,\n"
    "                                               global volatile int3 *restrict globalvolatileint3restrictp,\n"
    "                                               global volatile uint3* restrict globalvolatileuint3restrictp,\n"
    "                                               global volatile long3 * restrict globalvolatilelong3restrictp,\n"
    "                                               global volatile ulong3*restrict globalvolatileulong3restrictp,\n"
    "                                               global volatile float3 *restrict globalvolatilefloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector3_p(global const volatile char3* globalconstvolatilechar3p,\n"
    "                                            global const volatile uchar3 * globalconstvolatileuchar3p,\n"
    "                                            global const volatile short3*globalconstvolatileshort3p,\n"
    "                                            global const volatile ushort3 *globalconstvolatileushort3p,\n"
    "                                            global const volatile int3* globalconstvolatileint3p,\n"
    "                                            global const volatile uint3 * globalconstvolatileuint3p,\n"
    "                                            global const volatile long3*globalconstvolatilelong3p,\n"
    "                                            global const volatile ulong3 *globalconstvolatileulong3p,\n"
    "                                            global const volatile float3* globalconstvolatilefloat3p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector3_restrict_p(global const volatile char3 * restrict globalconstvolatilechar3restrictp,\n"
    "                                                     global const volatile uchar3*restrict globalconstvolatileuchar3restrictp,\n"
    "                                                     global const volatile short3 *restrict globalconstvolatileshort3restrictp,\n"
    "                                                     global const volatile ushort3* restrict globalconstvolatileushort3restrictp,\n"
    "                                                     global const volatile int3 * restrict globalconstvolatileint3restrictp,\n"
    "                                                     global const volatile uint3*restrict globalconstvolatileuint3restrictp,\n"
    "                                                     global const volatile long3 *restrict globalconstvolatilelong3restrictp,\n"
    "                                                     global const volatile ulong3* restrict globalconstvolatileulong3restrictp,\n"
    "                                                     global const volatile float3 * restrict globalconstvolatilefloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_vector3_p(local char3*localchar3p,\n"
    "                            local uchar3 *localuchar3p,\n"
    "                            local short3* localshort3p,\n"
    "                            local ushort3 * localushort3p,\n"
    "                            local int3*localint3p,\n"
    "                            local uint3 *localuint3p,\n"
    "                            local long3* locallong3p,\n"
    "                            local ulong3 * localulong3p,\n"
    "                            local float3*localfloat3p)\n"
    "{}\n",
    "\n"
    "kernel void local_vector3_restrict_p(local char3 *restrict localchar3restrictp,\n"
    "                                     local uchar3* restrict localuchar3restrictp,\n"
    "                                     local short3 * restrict localshort3restrictp,\n"
    "                                     local ushort3*restrict localushort3restrictp,\n"
    "                                     local int3 *restrict localint3restrictp,\n"
    "                                     local uint3* restrict localuint3restrictp,\n"
    "                                     local long3 * restrict locallong3restrictp,\n"
    "                                     local ulong3*restrict localulong3restrictp,\n"
    "                                     local float3 *restrict localfloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector3_p(local const char3* localconstchar3p,\n"
    "                                  local const uchar3 * localconstuchar3p,\n"
    "                                  local const short3*localconstshort3p,\n"
    "                                  local const ushort3 *localconstushort3p,\n"
    "                                  local const int3* localconstint3p,\n"
    "                                  local const uint3 * localconstuint3p,\n"
    "                                  local const long3*localconstlong3p,\n"
    "                                  local const ulong3 *localconstulong3p,\n"
    "                                  local const float3* localconstfloat3p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector3_restrict_p(local const char3 * restrict localconstchar3restrictp,\n"
    "                                           local const uchar3*restrict localconstuchar3restrictp,\n"
    "                                           local const short3 *restrict localconstshort3restrictp,\n"
    "                                           local const ushort3* restrict localconstushort3restrictp,\n"
    "                                           local const int3 * restrict localconstint3restrictp,\n"
    "                                           local const uint3*restrict localconstuint3restrictp,\n"
    "                                           local const long3 *restrict localconstlong3restrictp,\n"
    "                                           local const ulong3* restrict localconstulong3restrictp,\n"
    "                                           local const float3 * restrict localconstfloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector3_p(local volatile char3*localvolatilechar3p,\n"
    "                                     local volatile uchar3 *localvolatileuchar3p,\n"
    "                                     local volatile short3* localvolatileshort3p,\n"
    "                                     local volatile ushort3 * localvolatileushort3p,\n"
    "                                     local volatile int3*localvolatileint3p,\n"
    "                                     local volatile uint3 *localvolatileuint3p,\n"
    "                                     local volatile long3* localvolatilelong3p,\n"
    "                                     local volatile ulong3 * localvolatileulong3p,\n"
    "                                     local volatile float3*localvolatilefloat3p)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector3_restrict_p(local volatile char3 *restrict localvolatilechar3restrictp,\n"
    "                                              local volatile uchar3* restrict localvolatileuchar3restrictp,\n"
    "                                              local volatile short3 * restrict localvolatileshort3restrictp,\n"
    "                                              local volatile ushort3*restrict localvolatileushort3restrictp,\n"
    "                                              local volatile int3 *restrict localvolatileint3restrictp,\n"
    "                                              local volatile uint3* restrict localvolatileuint3restrictp,\n"
    "                                              local volatile long3 * restrict localvolatilelong3restrictp,\n"
    "                                              local volatile ulong3*restrict localvolatileulong3restrictp,\n"
    "                                              local volatile float3 *restrict localvolatilefloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector3_p(local const volatile char3* localconstvolatilechar3p,\n"
    "                                           local const volatile uchar3 * localconstvolatileuchar3p,\n"
    "                                           local const volatile short3*localconstvolatileshort3p,\n"
    "                                           local const volatile ushort3 *localconstvolatileushort3p,\n"
    "                                           local const volatile int3* localconstvolatileint3p,\n"
    "                                           local const volatile uint3 * localconstvolatileuint3p,\n"
    "                                           local const volatile long3*localconstvolatilelong3p,\n"
    "                                           local const volatile ulong3 *localconstvolatileulong3p,\n"
    "                                           local const volatile float3* localconstvolatilefloat3p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector3_restrict_p(local const volatile char3 * restrict localconstvolatilechar3restrictp,\n"
    "                                                    local const volatile uchar3*restrict localconstvolatileuchar3restrictp,\n"
    "                                                    local const volatile short3 *restrict localconstvolatileshort3restrictp,\n"
    "                                                    local const volatile ushort3* restrict localconstvolatileushort3restrictp,\n"
    "                                                    local const volatile int3 * restrict localconstvolatileint3restrictp,\n"
    "                                                    local const volatile uint3*restrict localconstvolatileuint3restrictp,\n"
    "                                                    local const volatile long3 *restrict localconstvolatilelong3restrictp,\n"
    "                                                    local const volatile ulong3* restrict localconstvolatileulong3restrictp,\n"
    "                                                    local const volatile float3 * restrict localconstvolatilefloat3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void vector3_d(char3 char3d,\n"
    "                      uchar3 uchar3d,\n"
    "                      short3 short3d,\n"
    "                      ushort3 ushort3d,\n"
    "                      int3 int3d,\n"
    "                      uint3 uint3d,\n"
    "                      long3 long3d,\n"
    "                      ulong3 ulong3d,\n"
    "                      float3 float3d)\n"
    "{}\n",
    "\n"
    "kernel void const_vector3_d(const char3 constchar3d,\n"
    "                            const uchar3 constuchar3d,\n"
    "                            const short3 constshort3d,\n"
    "                            const ushort3 constushort3d,\n"
    "                            const int3 constint3d,\n"
    "                            const uint3 constuint3d,\n"
    "                            const long3 constlong3d,\n"
    "                            const ulong3 constulong3d,\n"
    "                            const float3 constfloat3d)\n"
    "{}\n",
    "\n"
    "kernel void private_vector3_d(private char3 privatechar3d,\n"
    "                              private uchar3 privateuchar3d,\n"
    "                              private short3 privateshort3d,\n"
    "                              private ushort3 privateushort3d,\n"
    "                              private int3 privateint3d,\n"
    "                              private uint3 privateuint3d,\n"
    "                              private long3 privatelong3d,\n"
    "                              private ulong3 privateulong3d,\n"
    "                              private float3 privatefloat3d)\n"
    "{}\n",
    "\n"
    "kernel void private_const_vector3_d(private const char3 privateconstchar3d,\n"
    "                                    private const uchar3 privateconstuchar3d,\n"
    "                                    private const short3 privateconstshort3d,\n"
    "                                    private const ushort3 privateconstushort3d,\n"
    "                                    private const int3 privateconstint3d,\n"
    "                                    private const uint3 privateconstuint3d,\n"
    "                                    private const long3 privateconstlong3d,\n"
    "                                    private const ulong3 privateconstulong3d,\n"
    "                                    private const float3 privateconstfloat3d)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector4_p0(constant char4*constantchar4p,\n"
    "                               constant uchar4 *constantuchar4p,\n"
    "                               constant short4* constantshort4p,\n"
    "                               constant ushort4 * constantushort4p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector4_p1(constant int4*constantint4p,\n"
    "                               constant uint4 *constantuint4p,\n"
    "                               constant long4* constantlong4p,\n"
    "                               constant ulong4 * constantulong4p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector4_p2(constant float4*constantfloat4p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector4_restrict_p0(constant char4 *restrict constantchar4restrictp,\n"
    "                                        constant uchar4* restrict constantuchar4restrictp,\n"
    "                                        constant short4 * restrict constantshort4restrictp,\n"
    "                                        constant ushort4*restrict constantushort4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector4_restrict_p1(constant int4 *restrict constantint4restrictp,\n"
    "                                        constant uint4* restrict constantuint4restrictp,\n"
    "                                        constant long4 * restrict constantlong4restrictp,\n"
    "                                        constant ulong4*restrict constantulong4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector4_restrict_p2(constant float4 *restrict constantfloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_vector4_p(global char4*globalchar4p,\n"
    "                             global uchar4 *globaluchar4p,\n"
    "                             global short4* globalshort4p,\n"
    "                             global ushort4 * globalushort4p,\n"
    "                             global int4*globalint4p,\n"
    "                             global uint4 *globaluint4p,\n"
    "                             global long4* globallong4p,\n"
    "                             global ulong4 * globalulong4p,\n"
    "                             global float4*globalfloat4p)\n"
    "{}\n",
    "\n"
    "kernel void global_vector4_restrict_p(global char4 *restrict globalchar4restrictp,\n"
    "                                      global uchar4* restrict globaluchar4restrictp,\n"
    "                                      global short4 * restrict globalshort4restrictp,\n"
    "                                      global ushort4*restrict globalushort4restrictp,\n"
    "                                      global int4 *restrict globalint4restrictp,\n"
    "                                      global uint4* restrict globaluint4restrictp,\n"
    "                                      global long4 * restrict globallong4restrictp,\n"
    "                                      global ulong4*restrict globalulong4restrictp,\n"
    "                                      global float4 *restrict globalfloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector4_p(global const char4* globalconstchar4p,\n"
    "                                   global const uchar4 * globalconstuchar4p,\n"
    "                                   global const short4*globalconstshort4p,\n"
    "                                   global const ushort4 *globalconstushort4p,\n"
    "                                   global const int4* globalconstint4p,\n"
    "                                   global const uint4 * globalconstuint4p,\n"
    "                                   global const long4*globalconstlong4p,\n"
    "                                   global const ulong4 *globalconstulong4p,\n"
    "                                   global const float4* globalconstfloat4p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector4_restrict_p(global const char4 * restrict globalconstchar4restrictp,\n"
    "                                            global const uchar4*restrict globalconstuchar4restrictp,\n"
    "                                            global const short4 *restrict globalconstshort4restrictp,\n"
    "                                            global const ushort4* restrict globalconstushort4restrictp,\n"
    "                                            global const int4 * restrict globalconstint4restrictp,\n"
    "                                            global const uint4*restrict globalconstuint4restrictp,\n"
    "                                            global const long4 *restrict globalconstlong4restrictp,\n"
    "                                            global const ulong4* restrict globalconstulong4restrictp,\n"
    "                                            global const float4 * restrict globalconstfloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector4_p(global volatile char4*globalvolatilechar4p,\n"
    "                                      global volatile uchar4 *globalvolatileuchar4p,\n"
    "                                      global volatile short4* globalvolatileshort4p,\n"
    "                                      global volatile ushort4 * globalvolatileushort4p,\n"
    "                                      global volatile int4*globalvolatileint4p,\n"
    "                                      global volatile uint4 *globalvolatileuint4p,\n"
    "                                      global volatile long4* globalvolatilelong4p,\n"
    "                                      global volatile ulong4 * globalvolatileulong4p,\n"
    "                                      global volatile float4*globalvolatilefloat4p)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector4_restrict_p(global volatile char4 *restrict globalvolatilechar4restrictp,\n"
    "                                               global volatile uchar4* restrict globalvolatileuchar4restrictp,\n"
    "                                               global volatile short4 * restrict globalvolatileshort4restrictp,\n"
    "                                               global volatile ushort4*restrict globalvolatileushort4restrictp,\n"
    "                                               global volatile int4 *restrict globalvolatileint4restrictp,\n"
    "                                               global volatile uint4* restrict globalvolatileuint4restrictp,\n"
    "                                               global volatile long4 * restrict globalvolatilelong4restrictp,\n"
    "                                               global volatile ulong4*restrict globalvolatileulong4restrictp,\n"
    "                                               global volatile float4 *restrict globalvolatilefloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector4_p(global const volatile char4* globalconstvolatilechar4p,\n"
    "                                            global const volatile uchar4 * globalconstvolatileuchar4p,\n"
    "                                            global const volatile short4*globalconstvolatileshort4p,\n"
    "                                            global const volatile ushort4 *globalconstvolatileushort4p,\n"
    "                                            global const volatile int4* globalconstvolatileint4p,\n"
    "                                            global const volatile uint4 * globalconstvolatileuint4p,\n"
    "                                            global const volatile long4*globalconstvolatilelong4p,\n"
    "                                            global const volatile ulong4 *globalconstvolatileulong4p,\n"
    "                                            global const volatile float4* globalconstvolatilefloat4p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector4_restrict_p(global const volatile char4 * restrict globalconstvolatilechar4restrictp,\n"
    "                                                     global const volatile uchar4*restrict globalconstvolatileuchar4restrictp,\n"
    "                                                     global const volatile short4 *restrict globalconstvolatileshort4restrictp,\n"
    "                                                     global const volatile ushort4* restrict globalconstvolatileushort4restrictp,\n"
    "                                                     global const volatile int4 * restrict globalconstvolatileint4restrictp,\n"
    "                                                     global const volatile uint4*restrict globalconstvolatileuint4restrictp,\n"
    "                                                     global const volatile long4 *restrict globalconstvolatilelong4restrictp,\n"
    "                                                     global const volatile ulong4* restrict globalconstvolatileulong4restrictp,\n"
    "                                                     global const volatile float4 * restrict globalconstvolatilefloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_vector4_p(local char4*localchar4p,\n"
    "                            local uchar4 *localuchar4p,\n"
    "                            local short4* localshort4p,\n"
    "                            local ushort4 * localushort4p,\n"
    "                            local int4*localint4p,\n"
    "                            local uint4 *localuint4p,\n"
    "                            local long4* locallong4p,\n"
    "                            local ulong4 * localulong4p,\n"
    "                            local float4*localfloat4p)\n"
    "{}\n",
    "\n"
    "kernel void local_vector4_restrict_p(local char4 *restrict localchar4restrictp,\n"
    "                                     local uchar4* restrict localuchar4restrictp,\n"
    "                                     local short4 * restrict localshort4restrictp,\n"
    "                                     local ushort4*restrict localushort4restrictp,\n"
    "                                     local int4 *restrict localint4restrictp,\n"
    "                                     local uint4* restrict localuint4restrictp,\n"
    "                                     local long4 * restrict locallong4restrictp,\n"
    "                                     local ulong4*restrict localulong4restrictp,\n"
    "                                     local float4 *restrict localfloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector4_p(local const char4* localconstchar4p,\n"
    "                                  local const uchar4 * localconstuchar4p,\n"
    "                                  local const short4*localconstshort4p,\n"
    "                                  local const ushort4 *localconstushort4p,\n"
    "                                  local const int4* localconstint4p,\n"
    "                                  local const uint4 * localconstuint4p,\n"
    "                                  local const long4*localconstlong4p,\n"
    "                                  local const ulong4 *localconstulong4p,\n"
    "                                  local const float4* localconstfloat4p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector4_restrict_p(local const char4 * restrict localconstchar4restrictp,\n"
    "                                           local const uchar4*restrict localconstuchar4restrictp,\n"
    "                                           local const short4 *restrict localconstshort4restrictp,\n"
    "                                           local const ushort4* restrict localconstushort4restrictp,\n"
    "                                           local const int4 * restrict localconstint4restrictp,\n"
    "                                           local const uint4*restrict localconstuint4restrictp,\n"
    "                                           local const long4 *restrict localconstlong4restrictp,\n"
    "                                           local const ulong4* restrict localconstulong4restrictp,\n"
    "                                           local const float4 * restrict localconstfloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector4_p(local volatile char4*localvolatilechar4p,\n"
    "                                     local volatile uchar4 *localvolatileuchar4p,\n"
    "                                     local volatile short4* localvolatileshort4p,\n"
    "                                     local volatile ushort4 * localvolatileushort4p,\n"
    "                                     local volatile int4*localvolatileint4p,\n"
    "                                     local volatile uint4 *localvolatileuint4p,\n"
    "                                     local volatile long4* localvolatilelong4p,\n"
    "                                     local volatile ulong4 * localvolatileulong4p,\n"
    "                                     local volatile float4*localvolatilefloat4p)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector4_restrict_p(local volatile char4 *restrict localvolatilechar4restrictp,\n"
    "                                              local volatile uchar4* restrict localvolatileuchar4restrictp,\n"
    "                                              local volatile short4 * restrict localvolatileshort4restrictp,\n"
    "                                              local volatile ushort4*restrict localvolatileushort4restrictp,\n"
    "                                              local volatile int4 *restrict localvolatileint4restrictp,\n"
    "                                              local volatile uint4* restrict localvolatileuint4restrictp,\n"
    "                                              local volatile long4 * restrict localvolatilelong4restrictp,\n"
    "                                              local volatile ulong4*restrict localvolatileulong4restrictp,\n"
    "                                              local volatile float4 *restrict localvolatilefloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector4_p(local const volatile char4* localconstvolatilechar4p,\n"
    "                                           local const volatile uchar4 * localconstvolatileuchar4p,\n"
    "                                           local const volatile short4*localconstvolatileshort4p,\n"
    "                                           local const volatile ushort4 *localconstvolatileushort4p,\n"
    "                                           local const volatile int4* localconstvolatileint4p,\n"
    "                                           local const volatile uint4 * localconstvolatileuint4p,\n"
    "                                           local const volatile long4*localconstvolatilelong4p,\n"
    "                                           local const volatile ulong4 *localconstvolatileulong4p,\n"
    "                                           local const volatile float4* localconstvolatilefloat4p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector4_restrict_p(local const volatile char4 * restrict localconstvolatilechar4restrictp,\n"
    "                                                    local const volatile uchar4*restrict localconstvolatileuchar4restrictp,\n"
    "                                                    local const volatile short4 *restrict localconstvolatileshort4restrictp,\n"
    "                                                    local const volatile ushort4* restrict localconstvolatileushort4restrictp,\n"
    "                                                    local const volatile int4 * restrict localconstvolatileint4restrictp,\n"
    "                                                    local const volatile uint4*restrict localconstvolatileuint4restrictp,\n"
    "                                                    local const volatile long4 *restrict localconstvolatilelong4restrictp,\n"
    "                                                    local const volatile ulong4* restrict localconstvolatileulong4restrictp,\n"
    "                                                    local const volatile float4 * restrict localconstvolatilefloat4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void vector4_d(char4 char4d,\n"
    "                      uchar4 uchar4d,\n"
    "                      short4 short4d,\n"
    "                      ushort4 ushort4d,\n"
    "                      int4 int4d,\n"
    "                      uint4 uint4d,\n"
    "                      long4 long4d,\n"
    "                      ulong4 ulong4d,\n"
    "                      float4 float4d)\n"
    "{}\n",
    "\n"
    "kernel void const_vector4_d(const char4 constchar4d,\n"
    "                            const uchar4 constuchar4d,\n"
    "                            const short4 constshort4d,\n"
    "                            const ushort4 constushort4d,\n"
    "                            const int4 constint4d,\n"
    "                            const uint4 constuint4d,\n"
    "                            const long4 constlong4d,\n"
    "                            const ulong4 constulong4d,\n"
    "                            const float4 constfloat4d)\n"
    "{}\n",
    "\n"
    "kernel void private_vector4_d(private char4 privatechar4d,\n"
    "                              private uchar4 privateuchar4d,\n"
    "                              private short4 privateshort4d,\n"
    "                              private ushort4 privateushort4d,\n"
    "                              private int4 privateint4d,\n"
    "                              private uint4 privateuint4d,\n"
    "                              private long4 privatelong4d,\n"
    "                              private ulong4 privateulong4d,\n"
    "                              private float4 privatefloat4d)\n"
    "{}\n",
    "\n"
    "kernel void private_const_vector4_d(private const char4 privateconstchar4d,\n"
    "                                    private const uchar4 privateconstuchar4d,\n"
    "                                    private const short4 privateconstshort4d,\n"
    "                                    private const ushort4 privateconstushort4d,\n"
    "                                    private const int4 privateconstint4d,\n"
    "                                    private const uint4 privateconstuint4d,\n"
    "                                    private const long4 privateconstlong4d,\n"
    "                                    private const ulong4 privateconstulong4d,\n"
    "                                    private const float4 privateconstfloat4d)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector8_p0(constant char8*constantchar8p,\n"
    "                               constant uchar8 *constantuchar8p,\n"
    "                               constant short8* constantshort8p,\n"
    "                               constant ushort8 * constantushort8p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector8_p1(constant int8*constantint8p,\n"
    "                               constant uint8 *constantuint8p,\n"
    "                               constant long8* constantlong8p,\n"
    "                               constant ulong8 * constantulong8p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector8_p2(constant float8*constantfloat8p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector8_restrict_p0(constant char8 *restrict constantchar8restrictp,\n"
    "                                        constant uchar8* restrict constantuchar8restrictp,\n"
    "                                        constant short8 * restrict constantshort8restrictp,\n"
    "                                        constant ushort8*restrict constantushort8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector8_restrict_p1(constant int8 *restrict constantint8restrictp,\n"
    "                                        constant uint8* restrict constantuint8restrictp,\n"
    "                                        constant long8 * restrict constantlong8restrictp,\n"
    "                                        constant ulong8*restrict constantulong8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector8_restrict_p2(constant float8 *restrict constantfloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_vector8_p(global char8*globalchar8p,\n"
    "                             global uchar8 *globaluchar8p,\n"
    "                             global short8* globalshort8p,\n"
    "                             global ushort8 * globalushort8p,\n"
    "                             global int8*globalint8p,\n"
    "                             global uint8 *globaluint8p,\n"
    "                             global long8* globallong8p,\n"
    "                             global ulong8 * globalulong8p,\n"
    "                             global float8*globalfloat8p)\n"
    "{}\n",
    "\n"
    "kernel void global_vector8_restrict_p(global char8 *restrict globalchar8restrictp,\n"
    "                                      global uchar8* restrict globaluchar8restrictp,\n"
    "                                      global short8 * restrict globalshort8restrictp,\n"
    "                                      global ushort8*restrict globalushort8restrictp,\n"
    "                                      global int8 *restrict globalint8restrictp,\n"
    "                                      global uint8* restrict globaluint8restrictp,\n"
    "                                      global long8 * restrict globallong8restrictp,\n"
    "                                      global ulong8*restrict globalulong8restrictp,\n"
    "                                      global float8 *restrict globalfloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector8_p(global const char8* globalconstchar8p,\n"
    "                                   global const uchar8 * globalconstuchar8p,\n"
    "                                   global const short8*globalconstshort8p,\n"
    "                                   global const ushort8 *globalconstushort8p,\n"
    "                                   global const int8* globalconstint8p,\n"
    "                                   global const uint8 * globalconstuint8p,\n"
    "                                   global const long8*globalconstlong8p,\n"
    "                                   global const ulong8 *globalconstulong8p,\n"
    "                                   global const float8* globalconstfloat8p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector8_restrict_p(global const char8 * restrict globalconstchar8restrictp,\n"
    "                                            global const uchar8*restrict globalconstuchar8restrictp,\n"
    "                                            global const short8 *restrict globalconstshort8restrictp,\n"
    "                                            global const ushort8* restrict globalconstushort8restrictp,\n"
    "                                            global const int8 * restrict globalconstint8restrictp,\n"
    "                                            global const uint8*restrict globalconstuint8restrictp,\n"
    "                                            global const long8 *restrict globalconstlong8restrictp,\n"
    "                                            global const ulong8* restrict globalconstulong8restrictp,\n"
    "                                            global const float8 * restrict globalconstfloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector8_p(global volatile char8*globalvolatilechar8p,\n"
    "                                      global volatile uchar8 *globalvolatileuchar8p,\n"
    "                                      global volatile short8* globalvolatileshort8p,\n"
    "                                      global volatile ushort8 * globalvolatileushort8p,\n"
    "                                      global volatile int8*globalvolatileint8p,\n"
    "                                      global volatile uint8 *globalvolatileuint8p,\n"
    "                                      global volatile long8* globalvolatilelong8p,\n"
    "                                      global volatile ulong8 * globalvolatileulong8p,\n"
    "                                      global volatile float8*globalvolatilefloat8p)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector8_restrict_p(global volatile char8 *restrict globalvolatilechar8restrictp,\n"
    "                                               global volatile uchar8* restrict globalvolatileuchar8restrictp,\n"
    "                                               global volatile short8 * restrict globalvolatileshort8restrictp,\n"
    "                                               global volatile ushort8*restrict globalvolatileushort8restrictp,\n"
    "                                               global volatile int8 *restrict globalvolatileint8restrictp,\n"
    "                                               global volatile uint8* restrict globalvolatileuint8restrictp,\n"
    "                                               global volatile long8 * restrict globalvolatilelong8restrictp,\n"
    "                                               global volatile ulong8*restrict globalvolatileulong8restrictp,\n"
    "                                               global volatile float8 *restrict globalvolatilefloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector8_p(global const volatile char8* globalconstvolatilechar8p,\n"
    "                                            global const volatile uchar8 * globalconstvolatileuchar8p,\n"
    "                                            global const volatile short8*globalconstvolatileshort8p,\n"
    "                                            global const volatile ushort8 *globalconstvolatileushort8p,\n"
    "                                            global const volatile int8* globalconstvolatileint8p,\n"
    "                                            global const volatile uint8 * globalconstvolatileuint8p,\n"
    "                                            global const volatile long8*globalconstvolatilelong8p,\n"
    "                                            global const volatile ulong8 *globalconstvolatileulong8p,\n"
    "                                            global const volatile float8* globalconstvolatilefloat8p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector8_restrict_p(global const volatile char8 * restrict globalconstvolatilechar8restrictp,\n"
    "                                                     global const volatile uchar8*restrict globalconstvolatileuchar8restrictp,\n"
    "                                                     global const volatile short8 *restrict globalconstvolatileshort8restrictp,\n"
    "                                                     global const volatile ushort8* restrict globalconstvolatileushort8restrictp,\n"
    "                                                     global const volatile int8 * restrict globalconstvolatileint8restrictp,\n"
    "                                                     global const volatile uint8*restrict globalconstvolatileuint8restrictp,\n"
    "                                                     global const volatile long8 *restrict globalconstvolatilelong8restrictp,\n"
    "                                                     global const volatile ulong8* restrict globalconstvolatileulong8restrictp,\n"
    "                                                     global const volatile float8 * restrict globalconstvolatilefloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_vector8_p(local char8*localchar8p,\n"
    "                            local uchar8 *localuchar8p,\n"
    "                            local short8* localshort8p,\n"
    "                            local ushort8 * localushort8p,\n"
    "                            local int8*localint8p,\n"
    "                            local uint8 *localuint8p,\n"
    "                            local long8* locallong8p,\n"
    "                            local ulong8 * localulong8p,\n"
    "                            local float8*localfloat8p)\n"
    "{}\n",
    "\n"
    "kernel void local_vector8_restrict_p(local char8 *restrict localchar8restrictp,\n"
    "                                     local uchar8* restrict localuchar8restrictp,\n"
    "                                     local short8 * restrict localshort8restrictp,\n"
    "                                     local ushort8*restrict localushort8restrictp,\n"
    "                                     local int8 *restrict localint8restrictp,\n"
    "                                     local uint8* restrict localuint8restrictp,\n"
    "                                     local long8 * restrict locallong8restrictp,\n"
    "                                     local ulong8*restrict localulong8restrictp,\n"
    "                                     local float8 *restrict localfloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector8_p(local const char8* localconstchar8p,\n"
    "                                  local const uchar8 * localconstuchar8p,\n"
    "                                  local const short8*localconstshort8p,\n"
    "                                  local const ushort8 *localconstushort8p,\n"
    "                                  local const int8* localconstint8p,\n"
    "                                  local const uint8 * localconstuint8p,\n"
    "                                  local const long8*localconstlong8p,\n"
    "                                  local const ulong8 *localconstulong8p,\n"
    "                                  local const float8* localconstfloat8p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector8_restrict_p(local const char8 * restrict localconstchar8restrictp,\n"
    "                                           local const uchar8*restrict localconstuchar8restrictp,\n"
    "                                           local const short8 *restrict localconstshort8restrictp,\n"
    "                                           local const ushort8* restrict localconstushort8restrictp,\n"
    "                                           local const int8 * restrict localconstint8restrictp,\n"
    "                                           local const uint8*restrict localconstuint8restrictp,\n"
    "                                           local const long8 *restrict localconstlong8restrictp,\n"
    "                                           local const ulong8* restrict localconstulong8restrictp,\n"
    "                                           local const float8 * restrict localconstfloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector8_p(local volatile char8*localvolatilechar8p,\n"
    "                                     local volatile uchar8 *localvolatileuchar8p,\n"
    "                                     local volatile short8* localvolatileshort8p,\n"
    "                                     local volatile ushort8 * localvolatileushort8p,\n"
    "                                     local volatile int8*localvolatileint8p,\n"
    "                                     local volatile uint8 *localvolatileuint8p,\n"
    "                                     local volatile long8* localvolatilelong8p,\n"
    "                                     local volatile ulong8 * localvolatileulong8p,\n"
    "                                     local volatile float8*localvolatilefloat8p)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector8_restrict_p(local volatile char8 *restrict localvolatilechar8restrictp,\n"
    "                                              local volatile uchar8* restrict localvolatileuchar8restrictp,\n"
    "                                              local volatile short8 * restrict localvolatileshort8restrictp,\n"
    "                                              local volatile ushort8*restrict localvolatileushort8restrictp,\n"
    "                                              local volatile int8 *restrict localvolatileint8restrictp,\n"
    "                                              local volatile uint8* restrict localvolatileuint8restrictp,\n"
    "                                              local volatile long8 * restrict localvolatilelong8restrictp,\n"
    "                                              local volatile ulong8*restrict localvolatileulong8restrictp,\n"
    "                                              local volatile float8 *restrict localvolatilefloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector8_p(local const volatile char8* localconstvolatilechar8p,\n"
    "                                           local const volatile uchar8 * localconstvolatileuchar8p,\n"
    "                                           local const volatile short8*localconstvolatileshort8p,\n"
    "                                           local const volatile ushort8 *localconstvolatileushort8p,\n"
    "                                           local const volatile int8* localconstvolatileint8p,\n"
    "                                           local const volatile uint8 * localconstvolatileuint8p,\n"
    "                                           local const volatile long8*localconstvolatilelong8p,\n"
    "                                           local const volatile ulong8 *localconstvolatileulong8p,\n"
    "                                           local const volatile float8* localconstvolatilefloat8p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector8_restrict_p(local const volatile char8 * restrict localconstvolatilechar8restrictp,\n"
    "                                                    local const volatile uchar8*restrict localconstvolatileuchar8restrictp,\n"
    "                                                    local const volatile short8 *restrict localconstvolatileshort8restrictp,\n"
    "                                                    local const volatile ushort8* restrict localconstvolatileushort8restrictp,\n"
    "                                                    local const volatile int8 * restrict localconstvolatileint8restrictp,\n"
    "                                                    local const volatile uint8*restrict localconstvolatileuint8restrictp,\n"
    "                                                    local const volatile long8 *restrict localconstvolatilelong8restrictp,\n"
    "                                                    local const volatile ulong8* restrict localconstvolatileulong8restrictp,\n"
    "                                                    local const volatile float8 * restrict localconstvolatilefloat8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void vector8_d(char8 char8d,\n"
    "                      uchar8 uchar8d,\n"
    "                      short8 short8d,\n"
    "                      ushort8 ushort8d,\n"
    "                      int8 int8d,\n"
    "                      uint8 uint8d,\n"
    "                      long8 long8d,\n"
    "                      ulong8 ulong8d,\n"
    "                      float8 float8d)\n"
    "{}\n",
    "\n"
    "kernel void const_vector8_d(const char8 constchar8d,\n"
    "                            const uchar8 constuchar8d,\n"
    "                            const short8 constshort8d,\n"
    "                            const ushort8 constushort8d,\n"
    "                            const int8 constint8d,\n"
    "                            const uint8 constuint8d,\n"
    "                            const long8 constlong8d,\n"
    "                            const ulong8 constulong8d,\n"
    "                            const float8 constfloat8d)\n"
    "{}\n",
    "\n"
    "kernel void private_vector8_d(private char8 privatechar8d,\n"
    "                              private uchar8 privateuchar8d,\n"
    "                              private short8 privateshort8d,\n"
    "                              private ushort8 privateushort8d,\n"
    "                              private int8 privateint8d,\n"
    "                              private uint8 privateuint8d,\n"
    "                              private long8 privatelong8d,\n"
    "                              private ulong8 privateulong8d,\n"
    "                              private float8 privatefloat8d)\n"
    "{}\n",
    "\n"
    "kernel void private_const_vector8_d(private const char8 privateconstchar8d,\n"
    "                                    private const uchar8 privateconstuchar8d,\n"
    "                                    private const short8 privateconstshort8d,\n"
    "                                    private const ushort8 privateconstushort8d,\n"
    "                                    private const int8 privateconstint8d,\n"
    "                                    private const uint8 privateconstuint8d,\n"
    "                                    private const long8 privateconstlong8d,\n"
    "                                    private const ulong8 privateconstulong8d,\n"
    "                                    private const float8 privateconstfloat8d)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector16_p0(constant char16*constantchar16p,\n"
    "                                constant uchar16 *constantuchar16p,\n"
    "                                constant short16* constantshort16p,\n"
    "                                constant ushort16 * constantushort16p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector16_p1(constant int16*constantint16p,\n"
    "                                constant uint16 *constantuint16p,\n"
    "                                constant long16* constantlong16p,\n"
    "                                constant ulong16 * constantulong16p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector16_p2(constant float16*constantfloat16p)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector16_restrict_p0(constant char16 *restrict constantchar16restrictp,\n"
    "                                         constant uchar16* restrict constantuchar16restrictp,\n"
    "                                         constant short16 * restrict constantshort16restrictp,\n"
    "                                         constant ushort16*restrict constantushort16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector16_restrict_p1(constant int16 *restrict constantint16restrictp,\n"
    "                                         constant uint16* restrict constantuint16restrictp,\n"
    "                                         constant long16 * restrict constantlong16restrictp,\n"
    "                                         constant ulong16*restrict constantulong16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_vector16_restrict_p2(constant float16 *restrict constantfloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_vector16_p(global char16*globalchar16p,\n"
    "                              global uchar16 *globaluchar16p,\n"
    "                              global short16* globalshort16p,\n"
    "                              global ushort16 * globalushort16p,\n"
    "                              global int16*globalint16p,\n"
    "                              global uint16 *globaluint16p,\n"
    "                              global long16* globallong16p,\n"
    "                              global ulong16 * globalulong16p,\n"
    "                              global float16*globalfloat16p)\n"
    "{}\n",
    "\n"
    "kernel void global_vector16_restrict_p(global char16 *restrict globalchar16restrictp,\n"
    "                                       global uchar16* restrict globaluchar16restrictp,\n"
    "                                       global short16 * restrict globalshort16restrictp,\n"
    "                                       global ushort16*restrict globalushort16restrictp,\n"
    "                                       global int16 *restrict globalint16restrictp,\n"
    "                                       global uint16* restrict globaluint16restrictp,\n"
    "                                       global long16 * restrict globallong16restrictp,\n"
    "                                       global ulong16*restrict globalulong16restrictp,\n"
    "                                       global float16 *restrict globalfloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector16_p(global const char16* globalconstchar16p,\n"
    "                                    global const uchar16 * globalconstuchar16p,\n"
    "                                    global const short16*globalconstshort16p,\n"
    "                                    global const ushort16 *globalconstushort16p,\n"
    "                                    global const int16* globalconstint16p,\n"
    "                                    global const uint16 * globalconstuint16p,\n"
    "                                    global const long16*globalconstlong16p,\n"
    "                                    global const ulong16 *globalconstulong16p,\n"
    "                                    global const float16* globalconstfloat16p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_vector16_restrict_p(global const char16 * restrict globalconstchar16restrictp,\n"
    "                                             global const uchar16*restrict globalconstuchar16restrictp,\n"
    "                                             global const short16 *restrict globalconstshort16restrictp,\n"
    "                                             global const ushort16* restrict globalconstushort16restrictp,\n"
    "                                             global const int16 * restrict globalconstint16restrictp,\n"
    "                                             global const uint16*restrict globalconstuint16restrictp,\n"
    "                                             global const long16 *restrict globalconstlong16restrictp,\n"
    "                                             global const ulong16* restrict globalconstulong16restrictp,\n"
    "                                             global const float16 * restrict globalconstfloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector16_p(global volatile char16*globalvolatilechar16p,\n"
    "                                       global volatile uchar16 *globalvolatileuchar16p,\n"
    "                                       global volatile short16* globalvolatileshort16p,\n"
    "                                       global volatile ushort16 * globalvolatileushort16p,\n"
    "                                       global volatile int16*globalvolatileint16p,\n"
    "                                       global volatile uint16 *globalvolatileuint16p,\n"
    "                                       global volatile long16* globalvolatilelong16p,\n"
    "                                       global volatile ulong16 * globalvolatileulong16p,\n"
    "                                       global volatile float16*globalvolatilefloat16p)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_vector16_restrict_p(global volatile char16 *restrict globalvolatilechar16restrictp,\n"
    "                                                global volatile uchar16* restrict globalvolatileuchar16restrictp,\n"
    "                                                global volatile short16 * restrict globalvolatileshort16restrictp,\n"
    "                                                global volatile ushort16*restrict globalvolatileushort16restrictp,\n"
    "                                                global volatile int16 *restrict globalvolatileint16restrictp,\n"
    "                                                global volatile uint16* restrict globalvolatileuint16restrictp,\n"
    "                                                global volatile long16 * restrict globalvolatilelong16restrictp,\n"
    "                                                global volatile ulong16*restrict globalvolatileulong16restrictp,\n"
    "                                                global volatile float16 *restrict globalvolatilefloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector16_p(global const volatile char16* globalconstvolatilechar16p,\n"
    "                                             global const volatile uchar16 * globalconstvolatileuchar16p,\n"
    "                                             global const volatile short16*globalconstvolatileshort16p,\n"
    "                                             global const volatile ushort16 *globalconstvolatileushort16p,\n"
    "                                             global const volatile int16* globalconstvolatileint16p,\n"
    "                                             global const volatile uint16 * globalconstvolatileuint16p,\n"
    "                                             global const volatile long16*globalconstvolatilelong16p,\n"
    "                                             global const volatile ulong16 *globalconstvolatileulong16p,\n"
    "                                             global const volatile float16* globalconstvolatilefloat16p)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_vector16_restrict_p(global const volatile char16 * restrict globalconstvolatilechar16restrictp,\n"
    "                                                      global const volatile uchar16*restrict globalconstvolatileuchar16restrictp,\n"
    "                                                      global const volatile short16 *restrict globalconstvolatileshort16restrictp,\n"
    "                                                      global const volatile ushort16* restrict globalconstvolatileushort16restrictp,\n"
    "                                                      global const volatile int16 * restrict globalconstvolatileint16restrictp,\n"
    "                                                      global const volatile uint16*restrict globalconstvolatileuint16restrictp,\n"
    "                                                      global const volatile long16 *restrict globalconstvolatilelong16restrictp,\n"
    "                                                      global const volatile ulong16* restrict globalconstvolatileulong16restrictp,\n"
    "                                                      global const volatile float16 * restrict globalconstvolatilefloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_vector16_p(local char16*localchar16p,\n"
    "                             local uchar16 *localuchar16p,\n"
    "                             local short16* localshort16p,\n"
    "                             local ushort16 * localushort16p,\n"
    "                             local int16*localint16p,\n"
    "                             local uint16 *localuint16p,\n"
    "                             local long16* locallong16p,\n"
    "                             local ulong16 * localulong16p,\n"
    "                             local float16*localfloat16p)\n"
    "{}\n",
    "\n"
    "kernel void local_vector16_restrict_p(local char16 *restrict localchar16restrictp,\n"
    "                                      local uchar16* restrict localuchar16restrictp,\n"
    "                                      local short16 * restrict localshort16restrictp,\n"
    "                                      local ushort16*restrict localushort16restrictp,\n"
    "                                      local int16 *restrict localint16restrictp,\n"
    "                                      local uint16* restrict localuint16restrictp,\n"
    "                                      local long16 * restrict locallong16restrictp,\n"
    "                                      local ulong16*restrict localulong16restrictp,\n"
    "                                      local float16 *restrict localfloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector16_p(local const char16* localconstchar16p,\n"
    "                                   local const uchar16 * localconstuchar16p,\n"
    "                                   local const short16*localconstshort16p,\n"
    "                                   local const ushort16 *localconstushort16p,\n"
    "                                   local const int16* localconstint16p,\n"
    "                                   local const uint16 * localconstuint16p,\n"
    "                                   local const long16*localconstlong16p,\n"
    "                                   local const ulong16 *localconstulong16p,\n"
    "                                   local const float16* localconstfloat16p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_vector16_restrict_p(local const char16 * restrict localconstchar16restrictp,\n"
    "                                            local const uchar16*restrict localconstuchar16restrictp,\n"
    "                                            local const short16 *restrict localconstshort16restrictp,\n"
    "                                            local const ushort16* restrict localconstushort16restrictp,\n"
    "                                            local const int16 * restrict localconstint16restrictp,\n"
    "                                            local const uint16*restrict localconstuint16restrictp,\n"
    "                                            local const long16 *restrict localconstlong16restrictp,\n"
    "                                            local const ulong16* restrict localconstulong16restrictp,\n"
    "                                            local const float16 * restrict localconstfloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector16_p(local volatile char16*localvolatilechar16p,\n"
    "                                      local volatile uchar16 *localvolatileuchar16p,\n"
    "                                      local volatile short16* localvolatileshort16p,\n"
    "                                      local volatile ushort16 * localvolatileushort16p,\n"
    "                                      local volatile int16*localvolatileint16p,\n"
    "                                      local volatile uint16 *localvolatileuint16p,\n"
    "                                      local volatile long16* localvolatilelong16p,\n"
    "                                      local volatile ulong16 * localvolatileulong16p,\n"
    "                                      local volatile float16*localvolatilefloat16p)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_vector16_restrict_p(local volatile char16 *restrict localvolatilechar16restrictp,\n"
    "                                               local volatile uchar16* restrict localvolatileuchar16restrictp,\n"
    "                                               local volatile short16 * restrict localvolatileshort16restrictp,\n"
    "                                               local volatile ushort16*restrict localvolatileushort16restrictp,\n"
    "                                               local volatile int16 *restrict localvolatileint16restrictp,\n"
    "                                               local volatile uint16* restrict localvolatileuint16restrictp,\n"
    "                                               local volatile long16 * restrict localvolatilelong16restrictp,\n"
    "                                               local volatile ulong16*restrict localvolatileulong16restrictp,\n"
    "                                               local volatile float16 *restrict localvolatilefloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector16_p(local const volatile char16* localconstvolatilechar16p,\n"
    "                                            local const volatile uchar16 * localconstvolatileuchar16p,\n"
    "                                            local const volatile short16*localconstvolatileshort16p,\n"
    "                                            local const volatile ushort16 *localconstvolatileushort16p,\n"
    "                                            local const volatile int16* localconstvolatileint16p,\n"
    "                                            local const volatile uint16 * localconstvolatileuint16p,\n"
    "                                            local const volatile long16*localconstvolatilelong16p,\n"
    "                                            local const volatile ulong16 *localconstvolatileulong16p,\n"
    "                                            local const volatile float16* localconstvolatilefloat16p)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_vector16_restrict_p(local const volatile char16 * restrict localconstvolatilechar16restrictp,\n"
    "                                                     local const volatile uchar16*restrict localconstvolatileuchar16restrictp,\n"
    "                                                     local const volatile short16 *restrict localconstvolatileshort16restrictp,\n"
    "                                                     local const volatile ushort16* restrict localconstvolatileushort16restrictp,\n"
    "                                                     local const volatile int16 * restrict localconstvolatileint16restrictp,\n"
    "                                                     local const volatile uint16*restrict localconstvolatileuint16restrictp,\n"
    "                                                     local const volatile long16 *restrict localconstvolatilelong16restrictp,\n"
    "                                                     local const volatile ulong16* restrict localconstvolatileulong16restrictp,\n"
    "                                                     local const volatile float16 * restrict localconstvolatilefloat16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void vector16_d(char16 char16d,\n"
    "                       uchar16 uchar16d,\n"
    "                       short16 short16d,\n"
    "                       ushort16 ushort16d,\n"
    "                       int16 int16d,\n"
    "                       uint16 uint16d,\n"
    "                       long16 long16d,\n"
    "                       ulong16 ulong16d,\n"
    "                       float16 float16d)\n"
    "{}\n",
    "\n"
    "kernel void const_vector16_d(const char16 constchar16d,\n"
    "                             const uchar16 constuchar16d,\n"
    "                             const short16 constshort16d,\n"
    "                             const ushort16 constushort16d,\n"
    "                             const int16 constint16d,\n"
    "                             const uint16 constuint16d,\n"
    "                             const long16 constlong16d,\n"
    "                             const ulong16 constulong16d,\n"
    "                             const float16 constfloat16d)\n"
    "{}\n",
    "\n"
    "kernel void private_vector16_d(private char16 privatechar16d,\n"
    "                               private uchar16 privateuchar16d,\n"
    "                               private short16 privateshort16d,\n"
    "                               private ushort16 privateushort16d,\n"
    "                               private int16 privateint16d,\n"
    "                               private uint16 privateuint16d,\n"
    "                               private long16 privatelong16d,\n"
    "                               private ulong16 privateulong16d,\n"
    "                               private float16 privatefloat16d)\n"
    "{}\n",
    "\n"
    "kernel void private_const_vector16_d(private const char16 privateconstchar16d,\n"
    "                                     private const uchar16 privateconstuchar16d,\n"
    "                                     private const short16 privateconstshort16d,\n"
    "                                     private const ushort16 privateconstushort16d,\n"
    "                                     private const int16 privateconstint16d,\n"
    "                                     private const uint16 privateconstuint16d,\n"
    "                                     private const long16 privateconstlong16d,\n"
    "                                     private const ulong16 privateconstulong16d,\n"
    "                                     private const float16 privateconstfloat16d)\n"
    "{}\n",
    "\n"
    "kernel void constant_derived_p0(constant typedef_type*constanttypedef_typep,\n"
    "                               constant struct struct_type *constantstructstruct_typep,\n"
    "                               constant typedef_struct_type* constanttypedef_struct_typep,\n"
    "                               constant union union_type * constantunionunion_typep)\n"
    "{}\n",
    "\n"
    "kernel void constant_derived_p1(constant typedef_union_type*constanttypedef_union_typep,\n"
    "                               constant enum enum_type *constantenumenum_typep,\n"
    "                               constant typedef_enum_type* constanttypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void constant_derived_restrict_p0(constant typedef_type * restrict constanttypedef_typerestrictp,\n"
    "                                        constant struct struct_type*restrict constantstructstruct_typerestrictp,\n"
    "                                        constant typedef_struct_type *restrict constanttypedef_struct_typerestrictp,\n"
    "                                        constant union union_type* restrict constantunionunion_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void constant_derived_restrict_p1(constant typedef_union_type * restrict constanttypedef_union_typerestrictp,\n"
    "                                        constant enum enum_type*restrict constantenumenum_typerestrictp,\n"
    "                                        constant typedef_enum_type *restrict constanttypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_derived_p(global typedef_type*globaltypedef_typep,\n"
    "                             global struct struct_type *globalstructstruct_typep,\n"
    "                             global typedef_struct_type* globaltypedef_struct_typep,\n"
    "                             global union union_type * globalunionunion_typep,\n"
    "                             global typedef_union_type*globaltypedef_union_typep,\n"
    "                             global enum enum_type *globalenumenum_typep,\n"
    "                             global typedef_enum_type* globaltypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void global_derived_restrict_p(global typedef_type * restrict globaltypedef_typerestrictp,\n"
    "                                      global struct struct_type*restrict globalstructstruct_typerestrictp,\n"
    "                                      global typedef_struct_type *restrict globaltypedef_struct_typerestrictp,\n"
    "                                      global union union_type* restrict globalunionunion_typerestrictp,\n"
    "                                      global typedef_union_type * restrict globaltypedef_union_typerestrictp,\n"
    "                                      global enum enum_type*restrict globalenumenum_typerestrictp,\n"
    "                                      global typedef_enum_type *restrict globaltypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_derived_p(global const typedef_type* globalconsttypedef_typep,\n"
    "                                   global const struct struct_type * globalconststructstruct_typep,\n"
    "                                   global const typedef_struct_type*globalconsttypedef_struct_typep,\n"
    "                                   global const union union_type *globalconstunionunion_typep,\n"
    "                                   global const typedef_union_type* globalconsttypedef_union_typep,\n"
    "                                   global const enum enum_type * globalconstenumenum_typep,\n"
    "                                   global const typedef_enum_type*globalconsttypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void global_const_derived_restrict_p(global const typedef_type *restrict globalconsttypedef_typerestrictp,\n"
    "                                            global const struct struct_type* restrict globalconststructstruct_typerestrictp,\n"
    "                                            global const typedef_struct_type * restrict globalconsttypedef_struct_typerestrictp,\n"
    "                                            global const union union_type*restrict globalconstunionunion_typerestrictp,\n"
    "                                            global const typedef_union_type *restrict globalconsttypedef_union_typerestrictp,\n"
    "                                            global const enum enum_type* restrict globalconstenumenum_typerestrictp,\n"
    "                                            global const typedef_enum_type * restrict globalconsttypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_derived_p(global volatile typedef_type*globalvolatiletypedef_typep,\n"
    "                                      global volatile struct struct_type *globalvolatilestructstruct_typep,\n"
    "                                      global volatile typedef_struct_type* globalvolatiletypedef_struct_typep,\n"
    "                                      global volatile union union_type * globalvolatileunionunion_typep,\n"
    "                                      global volatile typedef_union_type*globalvolatiletypedef_union_typep,\n"
    "                                      global volatile enum enum_type *globalvolatileenumenum_typep,\n"
    "                                      global volatile typedef_enum_type* globalvolatiletypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void global_volatile_derived_restrict_p(global volatile typedef_type * restrict globalvolatiletypedef_typerestrictp,\n"
    "                                               global volatile struct struct_type*restrict globalvolatilestructstruct_typerestrictp,\n"
    "                                               global volatile typedef_struct_type *restrict globalvolatiletypedef_struct_typerestrictp,\n"
    "                                               global volatile union union_type* restrict globalvolatileunionunion_typerestrictp,\n"
    "                                               global volatile typedef_union_type * restrict globalvolatiletypedef_union_typerestrictp,\n"
    "                                               global volatile enum enum_type*restrict globalvolatileenumenum_typerestrictp,\n"
    "                                               global volatile typedef_enum_type *restrict globalvolatiletypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_derived_p(global const volatile typedef_type* globalconstvolatiletypedef_typep,\n"
    "                                            global const volatile struct struct_type * globalconstvolatilestructstruct_typep,\n"
    "                                            global const volatile typedef_struct_type*globalconstvolatiletypedef_struct_typep,\n"
    "                                            global const volatile union union_type *globalconstvolatileunionunion_typep,\n"
    "                                            global const volatile typedef_union_type* globalconstvolatiletypedef_union_typep,\n"
    "                                            global const volatile enum enum_type * globalconstvolatileenumenum_typep,\n"
    "                                            global const volatile typedef_enum_type*globalconstvolatiletypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void global_const_volatile_derived_restrict_p(global const volatile typedef_type *restrict globalconstvolatiletypedef_typerestrictp,\n"
    "                                                     global const volatile struct struct_type* restrict globalconstvolatilestructstruct_typerestrictp,\n"
    "                                                     global const volatile typedef_struct_type * restrict globalconstvolatiletypedef_struct_typerestrictp,\n"
    "                                                     global const volatile union union_type*restrict globalconstvolatileunionunion_typerestrictp,\n"
    "                                                     global const volatile typedef_union_type *restrict globalconstvolatiletypedef_union_typerestrictp,\n"
    "                                                     global const volatile enum enum_type* restrict globalconstvolatileenumenum_typerestrictp,\n"
    "                                                     global const volatile typedef_enum_type * restrict globalconstvolatiletypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_derived_p(local typedef_type*localtypedef_typep,\n"
    "                            local struct struct_type *localstructstruct_typep,\n"
    "                            local typedef_struct_type* localtypedef_struct_typep,\n"
    "                            local union union_type * localunionunion_typep,\n"
    "                            local typedef_union_type*localtypedef_union_typep,\n"
    "                            local enum enum_type *localenumenum_typep,\n"
    "                            local typedef_enum_type* localtypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void local_derived_restrict_p(local typedef_type * restrict localtypedef_typerestrictp,\n"
    "                                     local struct struct_type*restrict localstructstruct_typerestrictp,\n"
    "                                     local typedef_struct_type *restrict localtypedef_struct_typerestrictp,\n"
    "                                     local union union_type* restrict localunionunion_typerestrictp,\n"
    "                                     local typedef_union_type * restrict localtypedef_union_typerestrictp,\n"
    "                                     local enum enum_type*restrict localenumenum_typerestrictp,\n"
    "                                     local typedef_enum_type *restrict localtypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_derived_p(local const typedef_type* localconsttypedef_typep,\n"
    "                                  local const struct struct_type * localconststructstruct_typep,\n"
    "                                  local const typedef_struct_type*localconsttypedef_struct_typep,\n"
    "                                  local const union union_type *localconstunionunion_typep,\n"
    "                                  local const typedef_union_type* localconsttypedef_union_typep,\n"
    "                                  local const enum enum_type * localconstenumenum_typep,\n"
    "                                  local const typedef_enum_type*localconsttypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void local_const_derived_restrict_p(local const typedef_type *restrict localconsttypedef_typerestrictp,\n"
    "                                           local const struct struct_type* restrict localconststructstruct_typerestrictp,\n"
    "                                           local const typedef_struct_type * restrict localconsttypedef_struct_typerestrictp,\n"
    "                                           local const union union_type*restrict localconstunionunion_typerestrictp,\n"
    "                                           local const typedef_union_type *restrict localconsttypedef_union_typerestrictp,\n"
    "                                           local const enum enum_type* restrict localconstenumenum_typerestrictp,\n"
    "                                           local const typedef_enum_type * restrict localconsttypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_derived_p(local volatile typedef_type*localvolatiletypedef_typep,\n"
    "                                     local volatile struct struct_type *localvolatilestructstruct_typep,\n"
    "                                     local volatile typedef_struct_type* localvolatiletypedef_struct_typep,\n"
    "                                     local volatile union union_type * localvolatileunionunion_typep,\n"
    "                                     local volatile typedef_union_type*localvolatiletypedef_union_typep,\n"
    "                                     local volatile enum enum_type *localvolatileenumenum_typep,\n"
    "                                     local volatile typedef_enum_type* localvolatiletypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void local_volatile_derived_restrict_p(local volatile typedef_type * restrict localvolatiletypedef_typerestrictp,\n"
    "                                              local volatile struct struct_type*restrict localvolatilestructstruct_typerestrictp,\n"
    "                                              local volatile typedef_struct_type *restrict localvolatiletypedef_struct_typerestrictp,\n"
    "                                              local volatile union union_type* restrict localvolatileunionunion_typerestrictp,\n"
    "                                              local volatile typedef_union_type * restrict localvolatiletypedef_union_typerestrictp,\n"
    "                                              local volatile enum enum_type*restrict localvolatileenumenum_typerestrictp,\n"
    "                                              local volatile typedef_enum_type *restrict localvolatiletypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_derived_p(local const volatile typedef_type* localconstvolatiletypedef_typep,\n"
    "                                           local const volatile struct struct_type * localconstvolatilestructstruct_typep,\n"
    "                                           local const volatile typedef_struct_type*localconstvolatiletypedef_struct_typep,\n"
    "                                           local const volatile union union_type *localconstvolatileunionunion_typep,\n"
    "                                           local const volatile typedef_union_type* localconstvolatiletypedef_union_typep,\n"
    "                                           local const volatile enum enum_type * localconstvolatileenumenum_typep,\n"
    "                                           local const volatile typedef_enum_type*localconstvolatiletypedef_enum_typep)\n"
    "{}\n",
    "\n"
    "kernel void local_const_volatile_derived_restrict_p(local const volatile typedef_type *restrict localconstvolatiletypedef_typerestrictp,\n"
    "                                                    local const volatile struct struct_type* restrict localconstvolatilestructstruct_typerestrictp,\n"
    "                                                    local const volatile typedef_struct_type * restrict localconstvolatiletypedef_struct_typerestrictp,\n"
    "                                                    local const volatile union union_type*restrict localconstvolatileunionunion_typerestrictp,\n"
    "                                                    local const volatile typedef_union_type *restrict localconstvolatiletypedef_union_typerestrictp,\n"
    "                                                    local const volatile enum enum_type* restrict localconstvolatileenumenum_typerestrictp,\n"
    "                                                    local const volatile typedef_enum_type * restrict localconstvolatiletypedef_enum_typerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void derived_d(typedef_type typedef_typed,\n"
    "                      struct struct_type structstruct_typed,\n"
    "                      typedef_struct_type typedef_struct_typed,\n"
    "                      union union_type unionunion_typed,\n"
    "                      typedef_union_type typedef_union_typed,\n"
    "                      enum enum_type enumenum_typed,\n"
    "                      typedef_enum_type typedef_enum_typed)\n"
    "{}\n",
    "\n"
    "kernel void const_derived_d(const typedef_type consttypedef_typed,\n"
    "                            const struct struct_type conststructstruct_typed,\n"
    "                            const typedef_struct_type consttypedef_struct_typed,\n"
    "                            const union union_type constunionunion_typed,\n"
    "                            const typedef_union_type consttypedef_union_typed,\n"
    "                            const enum enum_type constenumenum_typed,\n"
    "                            const typedef_enum_type consttypedef_enum_typed)\n"
    "{}\n",
    "\n"
    "kernel void private_derived_d(private typedef_type privatetypedef_typed,\n"
    "                              private struct struct_type privatestructstruct_typed,\n"
    "                              private typedef_struct_type privatetypedef_struct_typed,\n"
    "                              private union union_type privateunionunion_typed,\n"
    "                              private typedef_union_type privatetypedef_union_typed,\n"
    "                              private enum enum_type privateenumenum_typed,\n"
    "                              private typedef_enum_type privatetypedef_enum_typed)\n"
    "{}\n",
    "\n"
    "kernel void private_const_derived_d(private const typedef_type privateconsttypedef_typed,\n"
    "                                    private const struct struct_type privateconststructstruct_typed,\n"
    "                                    private const typedef_struct_type privateconsttypedef_struct_typed,\n"
    "                                    private const union union_type privateconstunionunion_typed,\n"
    "                                    private const typedef_union_type privateconsttypedef_union_typed,\n"
    "                                    private const enum enum_type privateconstenumenum_typed,\n"
    "                                    private const typedef_enum_type privateconsttypedef_enum_typed)\n"
    "{}\n",
    "\n"
};

static const char * required_arg_info[][72] = {
  // The minimum value of CL_DEVICE_MAX_CONSTANT_ARGS is 4
    {
        "constant_scalar_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "void*", "constantvoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char*", "constantcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar*", "constantucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar*", "constantunsignedcharp",
    NULL
  },
  {
    "constant_scalar_p1",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short*", "constantshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort*", "constantushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort*", "constantunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int*", "constantintp",
    NULL
  },
  {
    "constant_scalar_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint*", "constantuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint*", "constantunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long*", "constantlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong*", "constantulongp",
    NULL
  },
  {
    "constant_scalar_p3",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong*", "constantunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float*", "constantfloatp",
        NULL
    },
    {
        "constant_scalar_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "constantvoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "constantcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "constantucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "constantunsignedcharrestrictp",
    NULL
  },
  {
    "constant_scalar_restrict_p1",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "constantshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "constantushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "constantunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "constantintrestrictp",
    NULL
  },
  {
    "constant_scalar_restrict_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "constantuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "constantunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "constantlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "constantulongrestrictp",
    NULL
  },
  {
    "constant_scalar_restrict_p3",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "constantunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "constantfloatrestrictp",
        NULL
    },
    {
        "global_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "void*", "globalvoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char*", "globalcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar*", "globalucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar*", "globalunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short*", "globalshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort*", "globalushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort*", "globalunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int*", "globalintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint*", "globaluintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint*", "globalunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long*", "globallongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong*", "globalulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong*", "globalunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float*", "globalfloatp",
        NULL
    },
    {
        "global_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "globalvoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "globalcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "globalshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "globalintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globaluintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "globallongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "globalfloatrestrictp",
        NULL
    },
    {
        "global_const_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "void*", "globalconstvoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char*", "globalconstcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar*", "globalconstucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar*", "globalconstunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short*", "globalconstshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort*", "globalconstushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort*", "globalconstunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int*", "globalconstintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint*", "globalconstuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint*", "globalconstunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long*", "globalconstlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong*", "globalconstulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong*", "globalconstunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float*", "globalconstfloatp",
        NULL
    },
    {
        "global_const_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "globalconstvoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "globalconstcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalconstucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalconstunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "globalconstshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalconstushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalconstunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "globalconstintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalconstuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalconstunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "globalconstlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalconstulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalconstunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "globalconstfloatrestrictp",
        NULL
    },
    {
        "global_volatile_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "void*", "globalvolatilevoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char*", "globalvolatilecharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "globalvolatileucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "globalvolatileunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short*", "globalvolatileshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "globalvolatileushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "globalvolatileunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int*", "globalvolatileintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "globalvolatileuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "globalvolatileunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long*", "globalvolatilelongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "globalvolatileulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "globalvolatileunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float*", "globalvolatilefloatp",
        NULL
    },
    {
        "global_volatile_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "globalvolatilevoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "globalvolatilecharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalvolatileucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalvolatileunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "globalvolatileshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalvolatileushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalvolatileunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "globalvolatileintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalvolatileuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalvolatileunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "globalvolatilelongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalvolatileulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalvolatileunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "globalvolatilefloatrestrictp",
        NULL
    },
    {
        "global_const_volatile_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "void*", "globalconstvolatilevoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char*", "globalconstvolatilecharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "globalconstvolatileucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "globalconstvolatileunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short*", "globalconstvolatileshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "globalconstvolatileushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "globalconstvolatileunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int*", "globalconstvolatileintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "globalconstvolatileuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "globalconstvolatileunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long*", "globalconstvolatilelongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "globalconstvolatileulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "globalconstvolatileunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float*", "globalconstvolatilefloatp",
        NULL
    },
    {
        "global_const_volatile_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "globalconstvolatilevoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "globalconstvolatilecharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalconstvolatileucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "globalconstvolatileunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "globalconstvolatileshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalconstvolatileushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "globalconstvolatileunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "globalconstvolatileintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalconstvolatileuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "globalconstvolatileunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "globalconstvolatilelongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalconstvolatileulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "globalconstvolatileunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "globalconstvolatilefloatrestrictp",
        NULL
    },
    {
        "local_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "void*", "localvoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char*", "localcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar*", "localucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar*", "localunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short*", "localshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort*", "localushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort*", "localunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int*", "localintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint*", "localuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint*", "localunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long*", "locallongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong*", "localulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong*", "localunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float*", "localfloatp",
        NULL
    },
    {
        "local_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "localvoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "localcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "localshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "localintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "locallongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "localfloatrestrictp",
        NULL
    },
    {
        "local_const_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "void*", "localconstvoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char*", "localconstcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar*", "localconstucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar*", "localconstunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short*", "localconstshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort*", "localconstushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort*", "localconstunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int*", "localconstintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint*", "localconstuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint*", "localconstunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long*", "localconstlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong*", "localconstulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong*", "localconstunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float*", "localconstfloatp",
        NULL
    },
    {
        "local_const_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "localconstvoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "localconstcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localconstucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localconstunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "localconstshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localconstushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localconstunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "localconstintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localconstuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localconstunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "localconstlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localconstulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localconstunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "localconstfloatrestrictp",
        NULL
    },
    {
        "local_volatile_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "void*", "localvolatilevoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char*", "localvolatilecharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "localvolatileucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "localvolatileunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short*", "localvolatileshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "localvolatileushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "localvolatileunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int*", "localvolatileintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "localvolatileuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "localvolatileunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long*", "localvolatilelongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "localvolatileulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "localvolatileunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float*", "localvolatilefloatp",
        NULL
    },
    {
        "local_volatile_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "localvolatilevoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "localvolatilecharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localvolatileucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localvolatileunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "localvolatileshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localvolatileushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localvolatileunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "localvolatileintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localvolatileuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localvolatileunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "localvolatilelongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localvolatileulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localvolatileunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "localvolatilefloatrestrictp",
        NULL
    },
    {
        "local_const_volatile_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "void*", "localconstvolatilevoidp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char*", "localconstvolatilecharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "localconstvolatileucharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar*", "localconstvolatileunsignedcharp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short*", "localconstvolatileshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "localconstvolatileushortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort*", "localconstvolatileunsignedshortp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int*", "localconstvolatileintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "localconstvolatileuintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint*", "localconstvolatileunsignedintp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long*", "localconstvolatilelongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "localconstvolatileulongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong*", "localconstvolatileunsignedlongp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float*", "localconstvolatilefloatp",
        NULL
    },
    {
        "local_const_volatile_scalar_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "void*", "localconstvolatilevoidrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char*", "localconstvolatilecharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localconstvolatileucharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar*", "localconstvolatileunsignedcharrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short*", "localconstvolatileshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localconstvolatileushortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort*", "localconstvolatileunsignedshortrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int*", "localconstvolatileintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localconstvolatileuintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint*", "localconstvolatileunsignedintrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long*", "localconstvolatilelongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localconstvolatileulongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong*", "localconstvolatileunsignedlongrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float*", "localconstvolatilefloatrestrictp",
        NULL
    },
    {
        "scalar_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char", "chard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "uchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "unsignedchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short", "shortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "ushortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "unsignedshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int", "intd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "uintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "unsignedintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long", "longd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "ulongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "unsignedlongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float", "floatd",
        NULL
    },
    {
        "const_scalar_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char", "constchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "constuchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "constunsignedchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short", "constshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "constushortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "constunsignedshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int", "constintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "constuintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "constunsignedintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long", "constlongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "constulongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "constunsignedlongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float", "constfloatd",
        NULL
    },
    {
        "private_scalar_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char", "privatechard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "privateuchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "privateunsignedchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short", "privateshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "privateushortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "privateunsignedshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int", "privateintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "privateuintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "privateunsignedintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long", "privatelongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "privateulongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "privateunsignedlongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float", "privatefloatd",
        NULL
    },
    {
        "private_const_scalar_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char", "privateconstchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "privateconstuchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar", "privateconstunsignedchard",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short", "privateconstshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "privateconstushortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort", "privateconstunsignedshortd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int", "privateconstintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "privateconstuintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint", "privateconstunsignedintd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long", "privateconstlongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "privateconstulongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong", "privateconstunsignedlongd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float", "privateconstfloatd",
        NULL
    },
    {
        "constant_vector2_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char2*", "constantchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar2*", "constantuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short2*", "constantshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort2*", "constantushort2p",
    NULL
    },
    {
        "constant_vector2_p1",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int2*", "constantint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint2*", "constantuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long2*", "constantlong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong2*", "constantulong2p",
    NULL
    },
    {
        "constant_vector2_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float2*", "constantfloat2p",
        NULL
    },
    {
        "constant_vector2_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "constantchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "constantuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "constantshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "constantushort2restrictp",
    NULL
    },
    {
        "constant_vector2_restrict_p1",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "constantint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "constantuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "constantlong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "constantulong2restrictp",
    NULL
    },
    {
        "constant_vector2_restrict_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "constantfloat2restrictp",
        NULL
    },
    {
        "global_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char2*", "globalchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar2*", "globaluchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short2*", "globalshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort2*", "globalushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int2*", "globalint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint2*", "globaluint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long2*", "globallong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong2*", "globalulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float2*", "globalfloat2p",
        NULL
    },
    {
        "global_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "globalchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "globaluchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "globalshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "globalushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "globalint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "globaluint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "globallong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "globalulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "globalfloat2restrictp",
        NULL
    },
    {
        "global_const_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char2*", "globalconstchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar2*", "globalconstuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short2*", "globalconstshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort2*", "globalconstushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int2*", "globalconstint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint2*", "globalconstuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long2*", "globalconstlong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong2*", "globalconstulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float2*", "globalconstfloat2p",
        NULL
    },
    {
        "global_const_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "globalconstchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "globalconstuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "globalconstshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "globalconstushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "globalconstint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "globalconstuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "globalconstlong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "globalconstulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "globalconstfloat2restrictp",
        NULL
    },
    {
        "global_volatile_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char2*", "globalvolatilechar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar2*", "globalvolatileuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short2*", "globalvolatileshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort2*", "globalvolatileushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int2*", "globalvolatileint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint2*", "globalvolatileuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long2*", "globalvolatilelong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong2*", "globalvolatileulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float2*", "globalvolatilefloat2p",
        NULL
    },
    {
        "global_volatile_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "globalvolatilechar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "globalvolatileuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "globalvolatileshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "globalvolatileushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "globalvolatileint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "globalvolatileuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "globalvolatilelong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "globalvolatileulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "globalvolatilefloat2restrictp",
        NULL
    },
    {
        "global_const_volatile_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char2*", "globalconstvolatilechar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar2*", "globalconstvolatileuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short2*", "globalconstvolatileshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort2*", "globalconstvolatileushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int2*", "globalconstvolatileint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint2*", "globalconstvolatileuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long2*", "globalconstvolatilelong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong2*", "globalconstvolatileulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float2*", "globalconstvolatilefloat2p",
        NULL
    },
    {
        "global_const_volatile_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "globalconstvolatilechar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "globalconstvolatileuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "globalconstvolatileshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "globalconstvolatileushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "globalconstvolatileint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "globalconstvolatileuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "globalconstvolatilelong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "globalconstvolatileulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "globalconstvolatilefloat2restrictp",
        NULL
    },
    {
        "local_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char2*", "localchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar2*", "localuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short2*", "localshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort2*", "localushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int2*", "localint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint2*", "localuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long2*", "locallong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong2*", "localulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float2*", "localfloat2p",
        NULL
    },
    {
        "local_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "localchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "localuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "localshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "localushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "localint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "localuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "locallong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "localulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "localfloat2restrictp",
        NULL
    },
    {
        "local_const_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char2*", "localconstchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar2*", "localconstuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short2*", "localconstshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort2*", "localconstushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int2*", "localconstint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint2*", "localconstuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long2*", "localconstlong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong2*", "localconstulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float2*", "localconstfloat2p",
        NULL
    },
    {
        "local_const_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "localconstchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "localconstuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "localconstshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "localconstushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "localconstint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "localconstuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "localconstlong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "localconstulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "localconstfloat2restrictp",
        NULL
    },
    {
        "local_volatile_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char2*", "localvolatilechar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar2*", "localvolatileuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short2*", "localvolatileshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort2*", "localvolatileushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int2*", "localvolatileint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint2*", "localvolatileuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long2*", "localvolatilelong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong2*", "localvolatileulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float2*", "localvolatilefloat2p",
        NULL
    },
    {
        "local_volatile_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "localvolatilechar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "localvolatileuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "localvolatileshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "localvolatileushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "localvolatileint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "localvolatileuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "localvolatilelong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "localvolatileulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "localvolatilefloat2restrictp",
        NULL
    },
    {
        "local_const_volatile_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char2*", "localconstvolatilechar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar2*", "localconstvolatileuchar2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short2*", "localconstvolatileshort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort2*", "localconstvolatileushort2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int2*", "localconstvolatileint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint2*", "localconstvolatileuint2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long2*", "localconstvolatilelong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong2*", "localconstvolatileulong2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float2*", "localconstvolatilefloat2p",
        NULL
    },
    {
        "local_const_volatile_vector2_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char2*", "localconstvolatilechar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar2*", "localconstvolatileuchar2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short2*", "localconstvolatileshort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort2*", "localconstvolatileushort2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int2*", "localconstvolatileint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint2*", "localconstvolatileuint2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long2*", "localconstvolatilelong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong2*", "localconstvolatileulong2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float2*", "localconstvolatilefloat2restrictp",
        NULL
    },
    {
        "vector2_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char2", "char2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar2", "uchar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short2", "short2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort2", "ushort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int2", "int2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint2", "uint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long2", "long2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong2", "ulong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float2", "float2d",
        NULL
    },
    {
        "const_vector2_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char2", "constchar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar2", "constuchar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short2", "constshort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort2", "constushort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int2", "constint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint2", "constuint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long2", "constlong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong2", "constulong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float2", "constfloat2d",
        NULL
    },
    {
        "private_vector2_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char2", "privatechar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar2", "privateuchar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short2", "privateshort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort2", "privateushort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int2", "privateint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint2", "privateuint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long2", "privatelong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong2", "privateulong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float2", "privatefloat2d",
        NULL
    },
    {
        "private_const_vector2_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char2", "privateconstchar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar2", "privateconstuchar2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short2", "privateconstshort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort2", "privateconstushort2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int2", "privateconstint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint2", "privateconstuint2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long2", "privateconstlong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong2", "privateconstulong2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float2", "privateconstfloat2d",
        NULL
    },
    {
        "constant_vector3_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char3*", "constantchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar3*", "constantuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short3*", "constantshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort3*", "constantushort3p",
        NULL
    },
    {
        "constant_vector3_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int3*", "constantint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint3*", "constantuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long3*", "constantlong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong3*", "constantulong3p",
    NULL
    },
    {
        "constant_vector3_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float3*", "constantfloat3p",
        NULL
    },
    {
        "constant_vector3_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "constantchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "constantuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "constantshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "constantushort3restrictp",
        NULL
    },
    {
        "constant_vector3_restrict_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "constantint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "constantuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "constantlong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "constantulong3restrictp",
    NULL
    },
    {
        "constant_vector3_restrict_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "constantfloat3restrictp",
        NULL
    },
    {
        "global_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char3*", "globalchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar3*", "globaluchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short3*", "globalshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort3*", "globalushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int3*", "globalint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint3*", "globaluint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long3*", "globallong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong3*", "globalulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float3*", "globalfloat3p",
        NULL
    },
    {
        "global_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "globalchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "globaluchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "globalshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "globalushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "globalint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "globaluint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "globallong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "globalulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "globalfloat3restrictp",
        NULL
    },
    {
        "global_const_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char3*", "globalconstchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar3*", "globalconstuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short3*", "globalconstshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort3*", "globalconstushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int3*", "globalconstint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint3*", "globalconstuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long3*", "globalconstlong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong3*", "globalconstulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float3*", "globalconstfloat3p",
        NULL
    },
    {
        "global_const_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "globalconstchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "globalconstuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "globalconstshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "globalconstushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "globalconstint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "globalconstuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "globalconstlong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "globalconstulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "globalconstfloat3restrictp",
        NULL
    },
    {
        "global_volatile_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char3*", "globalvolatilechar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar3*", "globalvolatileuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short3*", "globalvolatileshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort3*", "globalvolatileushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int3*", "globalvolatileint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint3*", "globalvolatileuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long3*", "globalvolatilelong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong3*", "globalvolatileulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float3*", "globalvolatilefloat3p",
        NULL
    },
    {
        "global_volatile_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "globalvolatilechar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "globalvolatileuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "globalvolatileshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "globalvolatileushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "globalvolatileint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "globalvolatileuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "globalvolatilelong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "globalvolatileulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "globalvolatilefloat3restrictp",
        NULL
    },
    {
        "global_const_volatile_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char3*", "globalconstvolatilechar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar3*", "globalconstvolatileuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short3*", "globalconstvolatileshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort3*", "globalconstvolatileushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int3*", "globalconstvolatileint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint3*", "globalconstvolatileuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long3*", "globalconstvolatilelong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong3*", "globalconstvolatileulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float3*", "globalconstvolatilefloat3p",
        NULL
    },
    {
        "global_const_volatile_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "globalconstvolatilechar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "globalconstvolatileuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "globalconstvolatileshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "globalconstvolatileushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "globalconstvolatileint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "globalconstvolatileuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "globalconstvolatilelong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "globalconstvolatileulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "globalconstvolatilefloat3restrictp",
        NULL
    },
    {
        "local_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char3*", "localchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar3*", "localuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short3*", "localshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort3*", "localushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int3*", "localint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint3*", "localuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long3*", "locallong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong3*", "localulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float3*", "localfloat3p",
        NULL
    },
    {
        "local_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "localchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "localuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "localshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "localushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "localint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "localuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "locallong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "localulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "localfloat3restrictp",
        NULL
    },
    {
        "local_const_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char3*", "localconstchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar3*", "localconstuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short3*", "localconstshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort3*", "localconstushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int3*", "localconstint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint3*", "localconstuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long3*", "localconstlong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong3*", "localconstulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float3*", "localconstfloat3p",
        NULL
    },
    {
        "local_const_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "localconstchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "localconstuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "localconstshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "localconstushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "localconstint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "localconstuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "localconstlong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "localconstulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "localconstfloat3restrictp",
        NULL
    },
    {
        "local_volatile_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char3*", "localvolatilechar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar3*", "localvolatileuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short3*", "localvolatileshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort3*", "localvolatileushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int3*", "localvolatileint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint3*", "localvolatileuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long3*", "localvolatilelong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong3*", "localvolatileulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float3*", "localvolatilefloat3p",
        NULL
    },
    {
        "local_volatile_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "localvolatilechar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "localvolatileuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "localvolatileshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "localvolatileushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "localvolatileint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "localvolatileuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "localvolatilelong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "localvolatileulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "localvolatilefloat3restrictp",
        NULL
    },
    {
        "local_const_volatile_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char3*", "localconstvolatilechar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar3*", "localconstvolatileuchar3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short3*", "localconstvolatileshort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort3*", "localconstvolatileushort3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int3*", "localconstvolatileint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint3*", "localconstvolatileuint3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long3*", "localconstvolatilelong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong3*", "localconstvolatileulong3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float3*", "localconstvolatilefloat3p",
        NULL
    },
    {
        "local_const_volatile_vector3_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char3*", "localconstvolatilechar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar3*", "localconstvolatileuchar3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short3*", "localconstvolatileshort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort3*", "localconstvolatileushort3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int3*", "localconstvolatileint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint3*", "localconstvolatileuint3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long3*", "localconstvolatilelong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong3*", "localconstvolatileulong3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float3*", "localconstvolatilefloat3restrictp",
        NULL
    },
    {
        "vector3_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char3", "char3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar3", "uchar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short3", "short3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort3", "ushort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int3", "int3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint3", "uint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long3", "long3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong3", "ulong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float3", "float3d",
        NULL
    },
    {
        "const_vector3_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char3", "constchar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar3", "constuchar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short3", "constshort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort3", "constushort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int3", "constint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint3", "constuint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long3", "constlong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong3", "constulong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float3", "constfloat3d",
        NULL
    },
    {
        "private_vector3_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char3", "privatechar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar3", "privateuchar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short3", "privateshort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort3", "privateushort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int3", "privateint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint3", "privateuint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long3", "privatelong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong3", "privateulong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float3", "privatefloat3d",
        NULL
    },
    {
        "private_const_vector3_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char3", "privateconstchar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar3", "privateconstuchar3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short3", "privateconstshort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort3", "privateconstushort3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int3", "privateconstint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint3", "privateconstuint3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long3", "privateconstlong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong3", "privateconstulong3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float3", "privateconstfloat3d",
        NULL
    },
    {
        "constant_vector4_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char4*", "constantchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar4*", "constantuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short4*", "constantshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort4*", "constantushort4p",
        NULL
    },
    {
        "constant_vector4_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int4*", "constantint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint4*", "constantuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long4*", "constantlong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong4*", "constantulong4p",
        NULL
    },
    {
        "constant_vector4_p2",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float4*", "constantfloat4p",
        NULL
    },
    {
        "constant_vector4_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "constantchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "constantuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "constantshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "constantushort4restrictp",
        NULL
    },
    {
        "constant_vector4_restrict_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "constantint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "constantuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "constantlong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "constantulong4restrictp",
        NULL
    },
    {
        "constant_vector4_restrict_p2",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "constantfloat4restrictp",
        NULL
    },
    {
        "global_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char4*", "globalchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar4*", "globaluchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short4*", "globalshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort4*", "globalushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int4*", "globalint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint4*", "globaluint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long4*", "globallong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong4*", "globalulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float4*", "globalfloat4p",
        NULL
    },
    {
        "global_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "globalchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "globaluchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "globalshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "globalushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "globalint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "globaluint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "globallong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "globalulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "globalfloat4restrictp",
        NULL
    },
    {
        "global_const_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char4*", "globalconstchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar4*", "globalconstuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short4*", "globalconstshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort4*", "globalconstushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int4*", "globalconstint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint4*", "globalconstuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long4*", "globalconstlong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong4*", "globalconstulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float4*", "globalconstfloat4p",
        NULL
    },
    {
        "global_const_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "globalconstchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "globalconstuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "globalconstshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "globalconstushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "globalconstint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "globalconstuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "globalconstlong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "globalconstulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "globalconstfloat4restrictp",
        NULL
    },
    {
        "global_volatile_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char4*", "globalvolatilechar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar4*", "globalvolatileuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short4*", "globalvolatileshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort4*", "globalvolatileushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int4*", "globalvolatileint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint4*", "globalvolatileuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long4*", "globalvolatilelong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong4*", "globalvolatileulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float4*", "globalvolatilefloat4p",
        NULL
    },
    {
        "global_volatile_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "globalvolatilechar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "globalvolatileuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "globalvolatileshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "globalvolatileushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "globalvolatileint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "globalvolatileuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "globalvolatilelong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "globalvolatileulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "globalvolatilefloat4restrictp",
        NULL
    },
    {
        "global_const_volatile_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char4*", "globalconstvolatilechar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar4*", "globalconstvolatileuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short4*", "globalconstvolatileshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort4*", "globalconstvolatileushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int4*", "globalconstvolatileint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint4*", "globalconstvolatileuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long4*", "globalconstvolatilelong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong4*", "globalconstvolatileulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float4*", "globalconstvolatilefloat4p",
        NULL
    },
    {
        "global_const_volatile_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "globalconstvolatilechar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "globalconstvolatileuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "globalconstvolatileshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "globalconstvolatileushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "globalconstvolatileint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "globalconstvolatileuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "globalconstvolatilelong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "globalconstvolatileulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "globalconstvolatilefloat4restrictp",
        NULL
    },
    {
        "local_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char4*", "localchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar4*", "localuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short4*", "localshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort4*", "localushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int4*", "localint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint4*", "localuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long4*", "locallong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong4*", "localulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float4*", "localfloat4p",
        NULL
    },
    {
        "local_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "localchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "localuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "localshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "localushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "localint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "localuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "locallong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "localulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "localfloat4restrictp",
        NULL
    },
    {
        "local_const_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char4*", "localconstchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar4*", "localconstuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short4*", "localconstshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort4*", "localconstushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int4*", "localconstint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint4*", "localconstuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long4*", "localconstlong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong4*", "localconstulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float4*", "localconstfloat4p",
        NULL
    },
    {
        "local_const_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "localconstchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "localconstuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "localconstshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "localconstushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "localconstint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "localconstuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "localconstlong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "localconstulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "localconstfloat4restrictp",
        NULL
    },
    {
        "local_volatile_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char4*", "localvolatilechar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar4*", "localvolatileuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short4*", "localvolatileshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort4*", "localvolatileushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int4*", "localvolatileint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint4*", "localvolatileuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long4*", "localvolatilelong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong4*", "localvolatileulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float4*", "localvolatilefloat4p",
        NULL
    },
    {
        "local_volatile_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "localvolatilechar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "localvolatileuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "localvolatileshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "localvolatileushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "localvolatileint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "localvolatileuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "localvolatilelong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "localvolatileulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "localvolatilefloat4restrictp",
        NULL
    },
    {
        "local_const_volatile_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char4*", "localconstvolatilechar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar4*", "localconstvolatileuchar4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short4*", "localconstvolatileshort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort4*", "localconstvolatileushort4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int4*", "localconstvolatileint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint4*", "localconstvolatileuint4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long4*", "localconstvolatilelong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong4*", "localconstvolatileulong4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float4*", "localconstvolatilefloat4p",
        NULL
    },
    {
        "local_const_volatile_vector4_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char4*", "localconstvolatilechar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar4*", "localconstvolatileuchar4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short4*", "localconstvolatileshort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort4*", "localconstvolatileushort4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int4*", "localconstvolatileint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint4*", "localconstvolatileuint4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long4*", "localconstvolatilelong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong4*", "localconstvolatileulong4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float4*", "localconstvolatilefloat4restrictp",
        NULL
    },
    {
        "vector4_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char4", "char4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar4", "uchar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short4", "short4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort4", "ushort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int4", "int4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint4", "uint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long4", "long4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong4", "ulong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float4", "float4d",
        NULL
    },
    {
        "const_vector4_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char4", "constchar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar4", "constuchar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short4", "constshort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort4", "constushort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int4", "constint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint4", "constuint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long4", "constlong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong4", "constulong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float4", "constfloat4d",
        NULL
    },
    {
        "private_vector4_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char4", "privatechar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar4", "privateuchar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short4", "privateshort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort4", "privateushort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int4", "privateint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint4", "privateuint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long4", "privatelong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong4", "privateulong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float4", "privatefloat4d",
        NULL
    },
    {
        "private_const_vector4_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char4", "privateconstchar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar4", "privateconstuchar4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short4", "privateconstshort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort4", "privateconstushort4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int4", "privateconstint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint4", "privateconstuint4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long4", "privateconstlong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong4", "privateconstulong4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float4", "privateconstfloat4d",
        NULL
    },
    {
        "constant_vector8_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char8*", "constantchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar8*", "constantuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short8*", "constantshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort8*", "constantushort8p",
        NULL
    },
    {
        "constant_vector8_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int8*", "constantint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint8*", "constantuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long8*", "constantlong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong8*", "constantulong8p",
    NULL
    },
    {
        "constant_vector8_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float8*", "constantfloat8p",
        NULL
    },
    {
        "constant_vector8_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "constantchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "constantuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "constantshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "constantushort8restrictp",
        NULL
    },
    {
        "constant_vector8_restrict_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "constantint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "constantuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "constantlong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "constantulong8restrictp",
    NULL
    },
    {
        "constant_vector8_restrict_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "constantfloat8restrictp",
        NULL
    },
    {
        "global_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char8*", "globalchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar8*", "globaluchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short8*", "globalshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort8*", "globalushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int8*", "globalint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint8*", "globaluint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long8*", "globallong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong8*", "globalulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float8*", "globalfloat8p",
        NULL
    },
    {
        "global_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "globalchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "globaluchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "globalshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "globalushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "globalint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "globaluint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "globallong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "globalulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "globalfloat8restrictp",
        NULL
    },
    {
        "global_const_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char8*", "globalconstchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar8*", "globalconstuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short8*", "globalconstshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort8*", "globalconstushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int8*", "globalconstint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint8*", "globalconstuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long8*", "globalconstlong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong8*", "globalconstulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float8*", "globalconstfloat8p",
        NULL
    },
    {
        "global_const_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "globalconstchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "globalconstuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "globalconstshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "globalconstushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "globalconstint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "globalconstuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "globalconstlong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "globalconstulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "globalconstfloat8restrictp",
        NULL
    },
    {
        "global_volatile_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char8*", "globalvolatilechar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar8*", "globalvolatileuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short8*", "globalvolatileshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort8*", "globalvolatileushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int8*", "globalvolatileint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint8*", "globalvolatileuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long8*", "globalvolatilelong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong8*", "globalvolatileulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float8*", "globalvolatilefloat8p",
        NULL
    },
    {
        "global_volatile_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "globalvolatilechar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "globalvolatileuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "globalvolatileshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "globalvolatileushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "globalvolatileint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "globalvolatileuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "globalvolatilelong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "globalvolatileulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "globalvolatilefloat8restrictp",
        NULL
    },
    {
        "global_const_volatile_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char8*", "globalconstvolatilechar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar8*", "globalconstvolatileuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short8*", "globalconstvolatileshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort8*", "globalconstvolatileushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int8*", "globalconstvolatileint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint8*", "globalconstvolatileuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long8*", "globalconstvolatilelong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong8*", "globalconstvolatileulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float8*", "globalconstvolatilefloat8p",
        NULL
    },
    {
        "global_const_volatile_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "globalconstvolatilechar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "globalconstvolatileuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "globalconstvolatileshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "globalconstvolatileushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "globalconstvolatileint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "globalconstvolatileuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "globalconstvolatilelong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "globalconstvolatileulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "globalconstvolatilefloat8restrictp",
        NULL
    },
    {
        "local_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char8*", "localchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar8*", "localuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short8*", "localshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort8*", "localushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int8*", "localint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint8*", "localuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long8*", "locallong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong8*", "localulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float8*", "localfloat8p",
        NULL
    },
    {
        "local_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "localchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "localuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "localshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "localushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "localint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "localuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "locallong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "localulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "localfloat8restrictp",
        NULL
    },
    {
        "local_const_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char8*", "localconstchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar8*", "localconstuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short8*", "localconstshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort8*", "localconstushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int8*", "localconstint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint8*", "localconstuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long8*", "localconstlong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong8*", "localconstulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float8*", "localconstfloat8p",
        NULL
    },
    {
        "local_const_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "localconstchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "localconstuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "localconstshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "localconstushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "localconstint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "localconstuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "localconstlong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "localconstulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "localconstfloat8restrictp",
        NULL
    },
    {
        "local_volatile_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char8*", "localvolatilechar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar8*", "localvolatileuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short8*", "localvolatileshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort8*", "localvolatileushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int8*", "localvolatileint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint8*", "localvolatileuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long8*", "localvolatilelong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong8*", "localvolatileulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float8*", "localvolatilefloat8p",
        NULL
    },
    {
        "local_volatile_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "localvolatilechar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "localvolatileuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "localvolatileshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "localvolatileushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "localvolatileint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "localvolatileuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "localvolatilelong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "localvolatileulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "localvolatilefloat8restrictp",
        NULL
    },
    {
        "local_const_volatile_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char8*", "localconstvolatilechar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar8*", "localconstvolatileuchar8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short8*", "localconstvolatileshort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort8*", "localconstvolatileushort8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int8*", "localconstvolatileint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint8*", "localconstvolatileuint8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long8*", "localconstvolatilelong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong8*", "localconstvolatileulong8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float8*", "localconstvolatilefloat8p",
        NULL
    },
    {
        "local_const_volatile_vector8_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char8*", "localconstvolatilechar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar8*", "localconstvolatileuchar8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short8*", "localconstvolatileshort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort8*", "localconstvolatileushort8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int8*", "localconstvolatileint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint8*", "localconstvolatileuint8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long8*", "localconstvolatilelong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong8*", "localconstvolatileulong8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float8*", "localconstvolatilefloat8restrictp",
        NULL
    },
    {
        "vector8_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char8", "char8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar8", "uchar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short8", "short8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort8", "ushort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int8", "int8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint8", "uint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long8", "long8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong8", "ulong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float8", "float8d",
        NULL
    },
    {
        "const_vector8_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char8", "constchar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar8", "constuchar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short8", "constshort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort8", "constushort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int8", "constint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint8", "constuint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long8", "constlong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong8", "constulong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float8", "constfloat8d",
        NULL
    },
    {
        "private_vector8_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char8", "privatechar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar8", "privateuchar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short8", "privateshort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort8", "privateushort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int8", "privateint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint8", "privateuint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long8", "privatelong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong8", "privateulong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float8", "privatefloat8d",
        NULL
    },
    {
        "private_const_vector8_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char8", "privateconstchar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar8", "privateconstuchar8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short8", "privateconstshort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort8", "privateconstushort8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int8", "privateconstint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint8", "privateconstuint8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long8", "privateconstlong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong8", "privateconstulong8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float8", "privateconstfloat8d",
        NULL
    },
    {
        "constant_vector16_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char16*", "constantchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar16*", "constantuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short16*", "constantshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort16*", "constantushort16p",
        NULL
    },
    {
        "constant_vector16_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int16*", "constantint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint16*", "constantuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long16*", "constantlong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong16*", "constantulong16p",
    NULL
    },
    {
        "constant_vector16_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float16*", "constantfloat16p",
        NULL
    },
    {
        "constant_vector16_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "constantchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "constantuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "constantshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "constantushort16restrictp",
        NULL
    },
    {
        "constant_vector16_restrict_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "constantint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "constantuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "constantlong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "constantulong16restrictp",
    NULL
    },
    {
        "constant_vector16_restrict_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "constantfloat16restrictp",
        NULL
    },
    {
        "global_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char16*", "globalchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar16*", "globaluchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short16*", "globalshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort16*", "globalushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int16*", "globalint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint16*", "globaluint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long16*", "globallong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong16*", "globalulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float16*", "globalfloat16p",
        NULL
    },
    {
        "global_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "globalchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "globaluchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "globalshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "globalushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "globalint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "globaluint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "globallong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "globalulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "globalfloat16restrictp",
        NULL
    },
    {
        "global_const_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char16*", "globalconstchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar16*", "globalconstuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short16*", "globalconstshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort16*", "globalconstushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int16*", "globalconstint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint16*", "globalconstuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long16*", "globalconstlong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong16*", "globalconstulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float16*", "globalconstfloat16p",
        NULL
    },
    {
        "global_const_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "globalconstchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "globalconstuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "globalconstshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "globalconstushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "globalconstint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "globalconstuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "globalconstlong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "globalconstulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "globalconstfloat16restrictp",
        NULL
    },
    {
        "global_volatile_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char16*", "globalvolatilechar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar16*", "globalvolatileuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short16*", "globalvolatileshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort16*", "globalvolatileushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int16*", "globalvolatileint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint16*", "globalvolatileuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long16*", "globalvolatilelong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong16*", "globalvolatileulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float16*", "globalvolatilefloat16p",
        NULL
    },
    {
        "global_volatile_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "globalvolatilechar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "globalvolatileuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "globalvolatileshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "globalvolatileushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "globalvolatileint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "globalvolatileuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "globalvolatilelong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "globalvolatileulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "globalvolatilefloat16restrictp",
        NULL
    },
    {
        "global_const_volatile_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char16*", "globalconstvolatilechar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar16*", "globalconstvolatileuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short16*", "globalconstvolatileshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort16*", "globalconstvolatileushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int16*", "globalconstvolatileint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint16*", "globalconstvolatileuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long16*", "globalconstvolatilelong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong16*", "globalconstvolatileulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float16*", "globalconstvolatilefloat16p",
        NULL
    },
    {
        "global_const_volatile_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "globalconstvolatilechar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "globalconstvolatileuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "globalconstvolatileshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "globalconstvolatileushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "globalconstvolatileint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "globalconstvolatileuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "globalconstvolatilelong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "globalconstvolatileulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "globalconstvolatilefloat16restrictp",
        NULL
    },
    {
        "local_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char16*", "localchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar16*", "localuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short16*", "localshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort16*", "localushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int16*", "localint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint16*", "localuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long16*", "locallong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong16*", "localulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float16*", "localfloat16p",
        NULL
    },
    {
        "local_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "localchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "localuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "localshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "localushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "localint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "localuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "locallong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "localulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "localfloat16restrictp",
        NULL
    },
    {
        "local_const_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "char16*", "localconstchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uchar16*", "localconstuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "short16*", "localconstshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ushort16*", "localconstushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "int16*", "localconstint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "uint16*", "localconstuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "long16*", "localconstlong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "ulong16*", "localconstulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "float16*", "localconstfloat16p",
        NULL
    },
    {
        "local_const_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "localconstchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "localconstuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "localconstshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "localconstushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "localconstint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "localconstuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "localconstlong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "localconstulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "localconstfloat16restrictp",
        NULL
    },
    {
        "local_volatile_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "char16*", "localvolatilechar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uchar16*", "localvolatileuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "short16*", "localvolatileshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ushort16*", "localvolatileushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "int16*", "localvolatileint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "uint16*", "localvolatileuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "long16*", "localvolatilelong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "ulong16*", "localvolatileulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "float16*", "localvolatilefloat16p",
        NULL
    },
    {
        "local_volatile_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "localvolatilechar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "localvolatileuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "localvolatileshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "localvolatileushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "localvolatileint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "localvolatileuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "localvolatilelong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "localvolatileulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "localvolatilefloat16restrictp",
        NULL
    },
    {
        "local_const_volatile_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "char16*", "localconstvolatilechar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uchar16*", "localconstvolatileuchar16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "short16*", "localconstvolatileshort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ushort16*", "localconstvolatileushort16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "int16*", "localconstvolatileint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "uint16*", "localconstvolatileuint16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "long16*", "localconstvolatilelong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "ulong16*", "localconstvolatileulong16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "float16*", "localconstvolatilefloat16p",
        NULL
    },
    {
        "local_const_volatile_vector16_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "char16*", "localconstvolatilechar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uchar16*", "localconstvolatileuchar16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "short16*", "localconstvolatileshort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ushort16*", "localconstvolatileushort16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "int16*", "localconstvolatileint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "uint16*", "localconstvolatileuint16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "long16*", "localconstvolatilelong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "ulong16*", "localconstvolatileulong16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "float16*", "localconstvolatilefloat16restrictp",
        NULL
    },
    {
        "vector16_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char16", "char16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar16", "uchar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short16", "short16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort16", "ushort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int16", "int16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint16", "uint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long16", "long16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong16", "ulong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float16", "float16d",
        NULL
    },
    {
        "const_vector16_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char16", "constchar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar16", "constuchar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short16", "constshort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort16", "constushort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int16", "constint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint16", "constuint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long16", "constlong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong16", "constulong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float16", "constfloat16d",
        NULL
    },
    {
        "private_vector16_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char16", "privatechar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar16", "privateuchar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short16", "privateshort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort16", "privateushort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int16", "privateint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint16", "privateuint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long16", "privatelong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong16", "privateulong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float16", "privatefloat16d",
        NULL
    },
    {
        "private_const_vector16_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "char16", "privateconstchar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uchar16", "privateconstuchar16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "short16", "privateconstshort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ushort16", "privateconstushort16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "int16", "privateconstint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "uint16", "privateconstuint16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "long16", "privateconstlong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "ulong16", "privateconstulong16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "float16", "privateconstfloat16d",
        NULL
    },
    {
        "constant_derived_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_type*", "constanttypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "struct struct_type*", "constantstructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_struct_type*", "constanttypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "union union_type*", "constantunionunion_typep",
        NULL
    },
    {
        "constant_derived_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_union_type*", "constanttypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "enum enum_type*", "constantenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_enum_type*", "constanttypedef_enum_typep",
        NULL
    },
    {
        "constant_derived_restrict_p0",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "constanttypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "constantstructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "constanttypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "constantunionunion_typerestrictp",
        NULL
    },
    {
        "constant_derived_restrict_p1",
    (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "constanttypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "constantenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "constanttypedef_enum_typerestrictp",
        NULL
    },
    {
        "global_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_type*", "globaltypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "struct struct_type*", "globalstructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_struct_type*", "globaltypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "union union_type*", "globalunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_union_type*", "globaltypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "enum enum_type*", "globalenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_enum_type*", "globaltypedef_enum_typep",
        NULL
    },
    {
        "global_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "globaltypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "globalstructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "globaltypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "globalunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "globaltypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "globalenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "globaltypedef_enum_typerestrictp",
        NULL
    },
    {
        "global_const_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_type*", "globalconsttypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "struct struct_type*", "globalconststructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_struct_type*", "globalconsttypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "union union_type*", "globalconstunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_union_type*", "globalconsttypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "enum enum_type*", "globalconstenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_enum_type*", "globalconsttypedef_enum_typep",
        NULL
    },
    {
        "global_const_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "globalconsttypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "globalconststructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "globalconsttypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "globalconstunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "globalconsttypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "globalconstenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "globalconsttypedef_enum_typerestrictp",
        NULL
    },
    {
        "global_volatile_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_type*", "globalvolatiletypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "struct struct_type*", "globalvolatilestructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_struct_type*", "globalvolatiletypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "union union_type*", "globalvolatileunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_union_type*", "globalvolatiletypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "enum enum_type*", "globalvolatileenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_enum_type*", "globalvolatiletypedef_enum_typep",
        NULL
    },
    {
        "global_volatile_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "globalvolatiletypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "globalvolatilestructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "globalvolatiletypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "globalvolatileunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "globalvolatiletypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "globalvolatileenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "globalvolatiletypedef_enum_typerestrictp",
        NULL
    },
    {
        "global_const_volatile_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_type*", "globalconstvolatiletypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "struct struct_type*", "globalconstvolatilestructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_struct_type*", "globalconstvolatiletypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "union union_type*", "globalconstvolatileunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_union_type*", "globalconstvolatiletypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "enum enum_type*", "globalconstvolatileenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_enum_type*", "globalconstvolatiletypedef_enum_typep",
        NULL
    },
    {
        "global_const_volatile_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "globalconstvolatiletypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "globalconstvolatilestructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "globalconstvolatiletypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "globalconstvolatileunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "globalconstvolatiletypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "globalconstvolatileenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "globalconstvolatiletypedef_enum_typerestrictp",
        NULL
    },
    {
        "local_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_type*", "localtypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "struct struct_type*", "localstructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_struct_type*", "localtypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "union union_type*", "localunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_union_type*", "localtypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "enum enum_type*", "localenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_enum_type*", "localtypedef_enum_typep",
        NULL
    },
    {
        "local_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "localtypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "localstructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "localtypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "localunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "localtypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "localenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "localtypedef_enum_typerestrictp",
        NULL
    },
    {
        "local_const_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_type*", "localconsttypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "struct struct_type*", "localconststructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_struct_type*", "localconsttypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "union union_type*", "localconstunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_union_type*", "localconsttypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "enum enum_type*", "localconstenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "typedef_enum_type*", "localconsttypedef_enum_typep",
        NULL
    },
    {
        "local_const_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "localconsttypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "localconststructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "localconsttypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "localconstunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "localconsttypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "localconstenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "localconsttypedef_enum_typerestrictp",
        NULL
    },
    {
        "local_volatile_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_type*", "localvolatiletypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "struct struct_type*", "localvolatilestructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_struct_type*", "localvolatiletypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "union union_type*", "localvolatileunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_union_type*", "localvolatiletypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "enum enum_type*", "localvolatileenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_enum_type*", "localvolatiletypedef_enum_typep",
        NULL
    },
    {
        "local_volatile_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "localvolatiletypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "localvolatilestructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "localvolatiletypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "localvolatileunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "localvolatiletypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "localvolatileenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "localvolatiletypedef_enum_typerestrictp",
        NULL
    },
    {
        "local_const_volatile_derived_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_type*", "localconstvolatiletypedef_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "struct struct_type*", "localconstvolatilestructstruct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_struct_type*", "localconstvolatiletypedef_struct_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "union union_type*", "localconstvolatileunionunion_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_union_type*", "localconstvolatiletypedef_union_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "enum enum_type*", "localconstvolatileenumenum_typep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "typedef_enum_type*", "localconstvolatiletypedef_enum_typep",
        NULL
    },
    {
        "local_const_volatile_derived_restrict_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_type*", "localconstvolatiletypedef_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "struct struct_type*", "localconstvolatilestructstruct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_struct_type*", "localconstvolatiletypedef_struct_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "union union_type*", "localconstvolatileunionunion_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_union_type*", "localconstvolatiletypedef_union_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "enum enum_type*", "localconstvolatileenumenum_typerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "typedef_enum_type*", "localconstvolatiletypedef_enum_typerestrictp",
        NULL
    },
    {
        "derived_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_type", "typedef_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "struct struct_type", "structstruct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_struct_type", "typedef_struct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "union union_type", "unionunion_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_union_type", "typedef_union_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "enum enum_type", "enumenum_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_enum_type", "typedef_enum_typed",
        NULL
    },
    {
        "const_derived_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_type", "consttypedef_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "struct struct_type", "conststructstruct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_struct_type", "consttypedef_struct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "union union_type", "constunionunion_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_union_type", "consttypedef_union_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "enum enum_type", "constenumenum_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_enum_type", "consttypedef_enum_typed",
        NULL
    },
    {
        "private_derived_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_type", "privatetypedef_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "struct struct_type", "privatestructstruct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_struct_type", "privatetypedef_struct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "union union_type", "privateunionunion_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_union_type", "privatetypedef_union_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "enum enum_type", "privateenumenum_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_enum_type", "privatetypedef_enum_typed",
        NULL
    },
    {
        "private_const_derived_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_type", "privateconsttypedef_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "struct struct_type", "privateconststructstruct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_struct_type", "privateconsttypedef_struct_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "union union_type", "privateconstunionunion_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_union_type", "privateconsttypedef_union_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "enum enum_type", "privateconstenumenum_typed",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "typedef_enum_type", "privateconsttypedef_enum_typed",
        NULL
    },
};

// Support for optional image data type
static const char * image_kernel_args[] = {
    "#pragma OPENCL EXTENSION cl_khr_3d_image_writes: enable\n"
    "kernel void image_d(read_only image2d_t image2d_td0,\n"
    "                    write_only image2d_t image2d_td1,\n"
    "                    read_only image3d_t image3d_td2,\n"
    "                    write_only image3d_t image3d_td3,\n"
    "                    read_only image2d_array_t image2d_array_td4,\n"
    "                    write_only image2d_array_t image2d_array_td5,\n"
    "                    read_only image1d_t image1d_td6,\n"
    "                    write_only image1d_t image1d_td7,\n"
    "                    read_only image1d_buffer_t image1d_buffer_td8,\n"
    "                    write_only image1d_buffer_t image1d_buffer_td9,\n"
    "                    read_only image1d_array_t image1d_array_td10,\n"
    "                    write_only image1d_array_t image1d_array_td11,\n"
    "                    sampler_t sampler_td12)\n"
    "{}\n",
    "\n"
};

static const char * image_arg_info[][67] = {
    {
        "image_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_READ_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image2d_t", "image2d_td0",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_WRITE_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image2d_t", "image2d_td1",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_READ_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image3d_t", "image3d_td2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_WRITE_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image3d_t", "image3d_td3",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_READ_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image2d_array_t", "image2d_array_td4",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_WRITE_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image2d_array_t", "image2d_array_td5",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_READ_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image1d_t", "image1d_td6",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_WRITE_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image1d_t", "image1d_td7",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_READ_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image1d_buffer_t", "image1d_buffer_td8",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_WRITE_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image1d_buffer_t", "image1d_buffer_td9",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_READ_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image1d_array_t", "image1d_array_td10",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_WRITE_ONLY, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "image1d_array_t", "image1d_array_td11",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "sampler_t", "sampler_td12",
        NULL
    },
};

// Support for optional double data type
static const char * double_kernel_args[] = {
    "kernel void double_scalar_p(constant double*constantdoublep,\n"
    "                            constant double *restrict constantdoublerestrictp,\n"
    "                            global double*globaldoublep,\n"
    "                            global double *restrict globaldoublerestrictp,\n"
    "                            global const double* globalconstdoublep,\n"
    "                            global const double * restrict globalconstdoublerestrictp,\n"
    "                            global volatile double*globalvolatiledoublep,\n"
    "                            global volatile double *restrict globalvolatiledoublerestrictp,\n"
    "                            global const volatile double* globalconstvolatiledoublep)\n"
    "{}\n",
    "\n"
    "kernel void double_scalar_p2(global const volatile double * restrict globalconstvolatiledoublerestrictp,\n"
    "                             local double*localdoublep,\n"
    "                             local double *restrict localdoublerestrictp,\n"
    "                             local const double* localconstdoublep,\n"
    "                             local const double * restrict localconstdoublerestrictp,\n"
    "                             local volatile double*localvolatiledoublep,\n"
    "                             local volatile double *restrict localvolatiledoublerestrictp,\n"
    "                             local const volatile double* localconstvolatiledoublep,\n"
    "                             local const volatile double * restrict localconstvolatiledoublerestrictp)\n"
    "{}\n",
    "\n"
    "kernel void double_scalar_d(double doubled,\n"
    "                            const double constdoubled,\n"
    "                            private double privatedoubled,\n"
    "                            private const double privateconstdoubled)\n"
    "{}\n",
    "\n"
    "kernel void double_vector2_p(constant double2*constantdouble2p,\n"
    "                             constant double2 *restrict constantdouble2restrictp,\n"
    "                             global double2*globaldouble2p,\n"
    "                             global double2 *restrict globaldouble2restrictp,\n"
    "                             global const double2* globalconstdouble2p,\n"
    "                             global const double2 * restrict globalconstdouble2restrictp,\n"
    "                             global volatile double2*globalvolatiledouble2p,\n"
    "                             global volatile double2 *restrict globalvolatiledouble2restrictp,\n"
    "                             global const volatile double2* globalconstvolatiledouble2p)\n"
    "{}\n",
    "\n"
    "kernel void double_vector2_p2(global const volatile double2 * restrict globalconstvolatiledouble2restrictp,\n"
    "                              local double2*localdouble2p,\n"
    "                              local double2 *restrict localdouble2restrictp,\n"
    "                              local const double2* localconstdouble2p,\n"
    "                              local const double2 * restrict localconstdouble2restrictp,\n"
    "                              local volatile double2*localvolatiledouble2p,\n"
    "                              local volatile double2 *restrict localvolatiledouble2restrictp,\n"
    "                              local const volatile double2* localconstvolatiledouble2p,\n"
    "                              local const volatile double2 * restrict localconstvolatiledouble2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void double_vector2_d(double2 double2d,\n"
    "                             const double2 constdouble2d,\n"
    "                             private double2 privatedouble2d,\n"
    "                             private const double2 privateconstdouble2d)\n"
    "{}\n",
    "\n"
    "kernel void double_vector3_p(constant double3*constantdouble3p,\n"
    "                             constant double3 *restrict constantdouble3restrictp,\n"
    "                             global double3*globaldouble3p,\n"
    "                             global double3 *restrict globaldouble3restrictp,\n"
    "                             global const double3* globalconstdouble3p,\n"
    "                             global const double3 * restrict globalconstdouble3restrictp,\n"
    "                             global volatile double3*globalvolatiledouble3p,\n"
    "                             global volatile double3 *restrict globalvolatiledouble3restrictp,\n"
    "                             global const volatile double3* globalconstvolatiledouble3p)\n"
    "{}\n",
    "\n"
    "kernel void double_vector3_p2(global const volatile double3 * restrict globalconstvolatiledouble3restrictp,\n"
    "                              local double3*localdouble3p,\n"
    "                              local double3 *restrict localdouble3restrictp,\n"
    "                              local const double3* localconstdouble3p,\n"
    "                              local const double3 * restrict localconstdouble3restrictp,\n"
    "                              local volatile double3*localvolatiledouble3p,\n"
    "                              local volatile double3 *restrict localvolatiledouble3restrictp,\n"
    "                              local const volatile double3* localconstvolatiledouble3p,\n"
    "                              local const volatile double3 * restrict localconstvolatiledouble3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void double_vector3_d(double3 double3d,\n"
    "                             const double3 constdouble3d,\n"
    "                             private double3 privatedouble3d,\n"
    "                             private const double3 privateconstdouble3d)\n"
    "{}\n",
    "\n"
    "kernel void double_vector4_p(constant double4*constantdouble4p,\n"
    "                             constant double4 *restrict constantdouble4restrictp,\n"
    "                             global double4*globaldouble4p,\n"
    "                             global double4 *restrict globaldouble4restrictp,\n"
    "                             global const double4* globalconstdouble4p,\n"
    "                             global const double4 * restrict globalconstdouble4restrictp,\n"
    "                             global volatile double4*globalvolatiledouble4p,\n"
    "                             global volatile double4 *restrict globalvolatiledouble4restrictp,\n"
    "                             global const volatile double4* globalconstvolatiledouble4p)\n"
    "{}\n",
    "\n"
    "kernel void double_vector4_p2(global const volatile double4 * restrict globalconstvolatiledouble4restrictp,\n"
    "                              local double4*localdouble4p,\n"
    "                              local double4 *restrict localdouble4restrictp,\n"
    "                              local const double4* localconstdouble4p,\n"
    "                              local const double4 * restrict localconstdouble4restrictp,\n"
    "                              local volatile double4*localvolatiledouble4p,\n"
    "                              local volatile double4 *restrict localvolatiledouble4restrictp,\n"
    "                              local const volatile double4* localconstvolatiledouble4p,\n"
    "                              local const volatile double4 * restrict localconstvolatiledouble4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void double_vector4_d(double4 double4d,\n"
    "                             const double4 constdouble4d,\n"
    "                             private double4 privatedouble4d,\n"
    "                             private const double4 privateconstdouble4d)\n"
    "{}\n",
    "\n"
    "kernel void double_vector8_p(constant double8*constantdouble8p,\n"
    "                             constant double8 *restrict constantdouble8restrictp,\n"
    "                             global double8*globaldouble8p,\n"
    "                             global double8 *restrict globaldouble8restrictp,\n"
    "                             global const double8* globalconstdouble8p,\n"
    "                             global const double8 * restrict globalconstdouble8restrictp,\n"
    "                             global volatile double8*globalvolatiledouble8p,\n"
    "                             global volatile double8 *restrict globalvolatiledouble8restrictp,\n"
    "                             global const volatile double8* globalconstvolatiledouble8p)\n"
    "{}\n",
    "\n"
    "kernel void double_vector8_p2(global const volatile double8 * restrict globalconstvolatiledouble8restrictp,\n"
    "                              local double8*localdouble8p,\n"
    "                              local double8 *restrict localdouble8restrictp,\n"
    "                              local const double8* localconstdouble8p,\n"
    "                              local const double8 * restrict localconstdouble8restrictp,\n"
    "                              local volatile double8*localvolatiledouble8p,\n"
    "                              local volatile double8 *restrict localvolatiledouble8restrictp,\n"
    "                              local const volatile double8* localconstvolatiledouble8p,\n"
    "                              local const volatile double8 * restrict localconstvolatiledouble8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void double_vector8_d(double8 double8d,\n"
    "                             const double8 constdouble8d,\n"
    "                             private double8 privatedouble8d,\n"
    "                             private const double8 privateconstdouble8d)\n"
    "{}\n",
    "\n"
    "kernel void double_vector16_p(constant double16*constantdouble16p,\n"
    "                              constant double16 *restrict constantdouble16restrictp,\n"
    "                              global double16*globaldouble16p,\n"
    "                              global double16 *restrict globaldouble16restrictp,\n"
    "                              global const double16* globalconstdouble16p,\n"
    "                              global const double16 * restrict globalconstdouble16restrictp,\n"
    "                              global volatile double16*globalvolatiledouble16p,\n"
    "                              global volatile double16 *restrict globalvolatiledouble16restrictp,\n"
    "                              global const volatile double16* globalconstvolatiledouble16p)\n"
    "{}\n",
    "\n"
    "kernel void double_vector16_p2(global const volatile double16 * restrict globalconstvolatiledouble16restrictp,\n"
    "                               local double16*localdouble16p,\n"
    "                               local double16 *restrict localdouble16restrictp,\n"
    "                               local const double16* localconstdouble16p,\n"
    "                               local const double16 * restrict localconstdouble16restrictp,\n"
    "                               local volatile double16*localvolatiledouble16p,\n"
    "                               local volatile double16 *restrict localvolatiledouble16restrictp,\n"
    "                               local const volatile double16* localconstvolatiledouble16p,\n"
    "                               local const volatile double16 * restrict localconstvolatiledouble16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void double_vector16_d(double16 double16d,\n"
    "                              const double16 constdouble16d,\n"
    "                              private double16 privatedouble16d,\n"
    "                              private const double16 privateconstdouble16d)\n"
    "{}\n",
    "\n"
};

static const char * double_arg_info[][77] = {
    {
        "double_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double*", "constantdoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "constantdoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double*", "globaldoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "globaldoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double*", "globalconstdoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "globalconstdoublerestrictp",
    (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double*", "globalvolatiledoublep",
    (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "globalvolatiledoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double*", "globalconstvolatiledoublep",
        NULL
    },
    {
        "double_scalar_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "globalconstvolatiledoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double*", "localdoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "localdoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double*", "localconstdoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "localconstdoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double*", "localvolatiledoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "localvolatiledoublerestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double*", "localconstvolatiledoublep",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double*", "localconstvolatiledoublerestrictp",
        NULL
    },
    {
        "double_scalar_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double", "doubled",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double", "constdoubled",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double", "privatedoubled",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double", "privateconstdoubled",
        NULL
    },
    {
        "double_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double2*", "constantdouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "constantdouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double2*", "globaldouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "globaldouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double2*", "globalconstdouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "globalconstdouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double2*", "globalvolatiledouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "globalvolatiledouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double2*", "globalconstvolatiledouble2p",
        NULL
    },
    {
        "double_vector2_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "globalconstvolatiledouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double2*", "localdouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "localdouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double2*", "localconstdouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "localconstdouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double2*", "localvolatiledouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "localvolatiledouble2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double2*", "localconstvolatiledouble2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double2*", "localconstvolatiledouble2restrictp",
        NULL
    },
    {
        "double_vector2_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double2", "double2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double2", "constdouble2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double2", "privatedouble2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double2", "privateconstdouble2d",
        NULL
    },
    {
        "double_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double3*", "constantdouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "constantdouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double3*", "globaldouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "globaldouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double3*", "globalconstdouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "globalconstdouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double3*", "globalvolatiledouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "globalvolatiledouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double3*", "globalconstvolatiledouble3p",
        NULL
    },
    {
        "double_vector3_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "globalconstvolatiledouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double3*", "localdouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "localdouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double3*", "localconstdouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "localconstdouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double3*", "localvolatiledouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "localvolatiledouble3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double3*", "localconstvolatiledouble3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double3*", "localconstvolatiledouble3restrictp",
        NULL
    },
    {
        "double_vector3_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double3", "double3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double3", "constdouble3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double3", "privatedouble3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double3", "privateconstdouble3d",
        NULL
    },
    {
        "double_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double4*", "constantdouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "constantdouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double4*", "globaldouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "globaldouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double4*", "globalconstdouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "globalconstdouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double4*", "globalvolatiledouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "globalvolatiledouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double4*", "globalconstvolatiledouble4p",
        NULL
    },
    {
        "double_vector4_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "globalconstvolatiledouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double4*", "localdouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "localdouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double4*", "localconstdouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "localconstdouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double4*", "localvolatiledouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "localvolatiledouble4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double4*", "localconstvolatiledouble4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double4*", "localconstvolatiledouble4restrictp",
        NULL
    },
    {
        "double_vector4_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double4", "double4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double4", "constdouble4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double4", "privatedouble4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double4", "privateconstdouble4d",
        NULL
    },
    {
        "double_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double8*", "constantdouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "constantdouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double8*", "globaldouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "globaldouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double8*", "globalconstdouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "globalconstdouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double8*", "globalvolatiledouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "globalvolatiledouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double8*", "globalconstvolatiledouble8p",
        NULL
    },
    {
        "double_vector8_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "globalconstvolatiledouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double8*", "localdouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "localdouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double8*", "localconstdouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "localconstdouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double8*", "localvolatiledouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "localvolatiledouble8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double8*", "localconstvolatiledouble8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double8*", "localconstvolatiledouble8restrictp",
        NULL
    },
    {
        "double_vector8_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double8", "double8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double8", "constdouble8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double8", "privatedouble8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double8", "privateconstdouble8d",
        NULL
    },
    {
        "double_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double16*", "constantdouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "constantdouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double16*", "globaldouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "globaldouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double16*", "globalconstdouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "globalconstdouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double16*", "globalvolatiledouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "globalvolatiledouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double16*", "globalconstvolatiledouble16p",
        NULL
    },
    {
        "double_vector16_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "globalconstvolatiledouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double16*", "localdouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "localdouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "double16*", "localconstdouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "localconstdouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "double16*", "localvolatiledouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "localvolatiledouble16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "double16*", "localconstvolatiledouble16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "double16*", "localconstvolatiledouble16restrictp",
        NULL
    },
    {
        "double_vector16_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double16", "double16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double16", "constdouble16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double16", "privatedouble16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "double16", "privateconstdouble16d",
        NULL
    },
};


// Support for optional half data type
static const char * half_kernel_args[] = {
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "\n"
    "kernel void half_scalar_p(constant half*constanthalfp,\n"
    "                          constant half *restrict constanthalfrestrictp,\n"
    "                          global half*globalhalfp,\n"
    "                          global half *restrict globalhalfrestrictp,\n"
    "                          global const half* globalconsthalfp,\n"
    "                          global const half * restrict globalconsthalfrestrictp,\n"
    "                          global volatile half*globalvolatilehalfp,\n"
    "                          global volatile half *restrict globalvolatilehalfrestrictp,\n"
    "                          global const volatile half* globalconstvolatilehalfp)\n"
    "{}\n",
    "\n"
    "kernel void half_scalar_p2(global const volatile half * restrict globalconstvolatilehalfrestrictp,\n"
    "                           local half*localhalfp,\n"
    "                           local half *restrict localhalfrestrictp,\n"
    "                           local const half* localconsthalfp,\n"
    "                           local const half * restrict localconsthalfrestrictp,\n"
    "                           local volatile half*localvolatilehalfp,\n"
    "                           local volatile half *restrict localvolatilehalfrestrictp,\n"
    "                           local const volatile half* localconstvolatilehalfp,\n"
    "                           local const volatile half * restrict localconstvolatilehalfrestrictp)\n"
    "{}\n",
    "\n"
    "kernel void half_scalar_d(half halfd,\n"
    "                          const half consthalfd,\n"
    "                          private half privatehalfd,\n"
    "                          private const half privateconsthalfd)\n"
    "{}\n",
    "\n"
    "kernel void half_vector2_p(constant half2*constanthalf2p,\n"
    "                           constant half2 *restrict constanthalf2restrictp,\n"
    "                           global half2*globalhalf2p,\n"
    "                           global half2 *restrict globalhalf2restrictp,\n"
    "                           global const half2* globalconsthalf2p,\n"
    "                           global const half2 * restrict globalconsthalf2restrictp,\n"
    "                           global volatile half2*globalvolatilehalf2p,\n"
    "                           global volatile half2 *restrict globalvolatilehalf2restrictp,\n"
    "                           global const volatile half2* globalconstvolatilehalf2p)\n"
    "{}\n",
    "\n"
    "kernel void half_vector2_p2(global const volatile half2 * restrict globalconstvolatilehalf2restrictp,\n"
    "                            local half2*localhalf2p,\n"
    "                            local half2 *restrict localhalf2restrictp,\n"
    "                            local const half2* localconsthalf2p,\n"
    "                            local const half2 * restrict localconsthalf2restrictp,\n"
    "                            local volatile half2*localvolatilehalf2p,\n"
    "                            local volatile half2 *restrict localvolatilehalf2restrictp,\n"
    "                            local const volatile half2* localconstvolatilehalf2p,\n"
    "                            local const volatile half2 * restrict localconstvolatilehalf2restrictp)\n"
    "{}\n",
    "\n"
    "kernel void half_vector2_d(half2 half2d,\n"
    "                           const half2 consthalf2d,\n"
    "                           private half2 privatehalf2d,\n"
    "                           private const half2 privateconsthalf2d)\n"
    "{}\n",
    "\n"
    "kernel void half_vector3_p(constant half3*constanthalf3p,\n"
    "                           constant half3 *restrict constanthalf3restrictp,\n"
    "                           global half3*globalhalf3p,\n"
    "                           global half3 *restrict globalhalf3restrictp,\n"
    "                           global const half3* globalconsthalf3p,\n"
    "                           global const half3 * restrict globalconsthalf3restrictp,\n"
    "                           global volatile half3*globalvolatilehalf3p,\n"
    "                           global volatile half3 *restrict globalvolatilehalf3restrictp,\n"
    "                           global const volatile half3* globalconstvolatilehalf3p)\n"
    "{}\n",
    "\n"
    "kernel void half_vector3_p2(global const volatile half3 * restrict globalconstvolatilehalf3restrictp,\n"
    "                            local half3*localhalf3p,\n"
    "                            local half3 *restrict localhalf3restrictp,\n"
    "                            local const half3* localconsthalf3p,\n"
    "                            local const half3 * restrict localconsthalf3restrictp,\n"
    "                            local volatile half3*localvolatilehalf3p,\n"
    "                            local volatile half3 *restrict localvolatilehalf3restrictp,\n"
    "                            local const volatile half3* localconstvolatilehalf3p,\n"
    "                            local const volatile half3 * restrict localconstvolatilehalf3restrictp)\n"
    "{}\n",
    "\n"
    "kernel void half_vector3_d(half3 half3d,\n"
    "                           const half3 consthalf3d,\n"
    "                           private half3 privatehalf3d,\n"
    "                           private const half3 privateconsthalf3d)\n"
    "{}\n",
    "\n"
    "kernel void half_vector4_p(constant half4*constanthalf4p,\n"
    "                           constant half4 *restrict constanthalf4restrictp,\n"
    "                           global half4*globalhalf4p,\n"
    "                           global half4 *restrict globalhalf4restrictp,\n"
    "                           global const half4* globalconsthalf4p,\n"
    "                           global const half4 * restrict globalconsthalf4restrictp,\n"
    "                           global volatile half4*globalvolatilehalf4p,\n"
    "                           global volatile half4 *restrict globalvolatilehalf4restrictp,\n"
    "                           global const volatile half4* globalconstvolatilehalf4p)\n"
    "{}\n",
    "\n"
    "kernel void half_vector4_p2(global const volatile half4 * restrict globalconstvolatilehalf4restrictp,\n"
    "                            local half4*localhalf4p,\n"
    "                            local half4 *restrict localhalf4restrictp,\n"
    "                            local const half4* localconsthalf4p,\n"
    "                            local const half4 * restrict localconsthalf4restrictp,\n"
    "                            local volatile half4*localvolatilehalf4p,\n"
    "                            local volatile half4 *restrict localvolatilehalf4restrictp,\n"
    "                            local const volatile half4* localconstvolatilehalf4p,\n"
    "                            local const volatile half4 * restrict localconstvolatilehalf4restrictp)\n"
    "{}\n",
    "\n"
    "kernel void half_vector4_d(half4 half4d,\n"
    "                           const half4 consthalf4d,\n"
    "                           private half4 privatehalf4d,\n"
    "                           private const half4 privateconsthalf4d)\n"
    "{}\n",
    "\n"
    "kernel void half_vector8_p(constant half8*constanthalf8p,\n"
    "                           constant half8 *restrict constanthalf8restrictp,\n"
    "                           global half8*globalhalf8p,\n"
    "                           global half8 *restrict globalhalf8restrictp,\n"
    "                           global const half8* globalconsthalf8p,\n"
    "                           global const half8 * restrict globalconsthalf8restrictp,\n"
    "                           global volatile half8*globalvolatilehalf8p,\n"
    "                           global volatile half8 *restrict globalvolatilehalf8restrictp,\n"
    "                           global const volatile half8* globalconstvolatilehalf8p)\n"
    "{}\n",
    "\n"
    "kernel void half_vector8_p2(global const volatile half8 * restrict globalconstvolatilehalf8restrictp,\n"
    "                            local half8*localhalf8p,\n"
    "                            local half8 *restrict localhalf8restrictp,\n"
    "                            local const half8* localconsthalf8p,\n"
    "                            local const half8 * restrict localconsthalf8restrictp,\n"
    "                            local volatile half8*localvolatilehalf8p,\n"
    "                            local volatile half8 *restrict localvolatilehalf8restrictp,\n"
    "                            local const volatile half8* localconstvolatilehalf8p,\n"
    "                            local const volatile half8 * restrict localconstvolatilehalf8restrictp)\n"
    "{}\n",
    "\n"
    "kernel void half_vector8_d(half8 half8d,\n"
    "                           const half8 consthalf8d,\n"
    "                           private half8 privatehalf8d,\n"
    "                           private const half8 privateconsthalf8d)\n"
    "{}\n",
    "\n"
    "kernel void half_vector16_p(constant half16*constanthalf16p,\n"
    "                            constant half16 *restrict constanthalf16restrictp,\n"
    "                            global half16*globalhalf16p,\n"
    "                            global half16 *restrict globalhalf16restrictp,\n"
    "                            global const half16* globalconsthalf16p,\n"
    "                            global const half16 * restrict globalconsthalf16restrictp,\n"
    "                            global volatile half16*globalvolatilehalf16p,\n"
    "                            global volatile half16 *restrict globalvolatilehalf16restrictp,\n"
    "                            global const volatile half16* globalconstvolatilehalf16p)\n"
    "{}\n",
    "\n"
    "kernel void half_vector16_p2(global const volatile half16 * restrict globalconstvolatilehalf16restrictp,\n"
    "                             local half16*localhalf16p,\n"
    "                             local half16 *restrict localhalf16restrictp,\n"
    "                             local const half16* localconsthalf16p,\n"
    "                             local const half16 * restrict localconsthalf16restrictp,\n"
    "                             local volatile half16*localvolatilehalf16p,\n"
    "                             local volatile half16 *restrict localvolatilehalf16restrictp,\n"
    "                             local const volatile half16* localconstvolatilehalf16p,\n"
    "                             local const volatile half16 * restrict localconstvolatilehalf16restrictp)\n"
    "{}\n",
    "\n"
    "kernel void half_vector16_d(half16 half16d,\n"
    "                            const half16 consthalf16d,\n"
    "                            private half16 privatehalf16d,\n"
    "                            private const half16 privateconsthalf16d)\n"
    "{}\n",
    "\n"
};

static const char * half_arg_info[][77] = {
    {
        "half_scalar_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half*", "constanthalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "constanthalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half*", "globalhalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "globalhalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half*", "globalconsthalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "globalconsthalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half*", "globalvolatilehalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "globalvolatilehalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half*", "globalconstvolatilehalfp",
        NULL
    },
    {
        "half_scalar_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "globalconstvolatilehalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half*", "localhalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "localhalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half*", "localconsthalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "localconsthalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half*", "localvolatilehalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "localvolatilehalfrestrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half*", "localconstvolatilehalfp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half*", "localconstvolatilehalfrestrictp",
        NULL
    },
    {
        "half_scalar_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half", "halfd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half", "consthalfd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half", "privatehalfd",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half", "privateconsthalfd",
        NULL
    },
    {
        "half_vector2_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half2*", "constanthalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "constanthalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half2*", "globalhalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "globalhalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half2*", "globalconsthalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "globalconsthalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half2*", "globalvolatilehalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "globalvolatilehalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half2*", "globalconstvolatilehalf2p",
        NULL
    },
    {
        "half_vector2_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "globalconstvolatilehalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half2*", "localhalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "localhalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half2*", "localconsthalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "localconsthalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half2*", "localvolatilehalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "localvolatilehalf2restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half2*", "localconstvolatilehalf2p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half2*", "localconstvolatilehalf2restrictp",
        NULL
    },
    {
        "half_vector2_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half2", "half2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half2", "consthalf2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half2", "privatehalf2d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half2", "privateconsthalf2d",
        NULL
    },
    {
        "half_vector3_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half3*", "constanthalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "constanthalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half3*", "globalhalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "globalhalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half3*", "globalconsthalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "globalconsthalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half3*", "globalvolatilehalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "globalvolatilehalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half3*", "globalconstvolatilehalf3p",
        NULL
    },
    {
        "half_vector3_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "globalconstvolatilehalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half3*", "localhalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "localhalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half3*", "localconsthalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "localconsthalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half3*", "localvolatilehalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "localvolatilehalf3restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half3*", "localconstvolatilehalf3p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half3*", "localconstvolatilehalf3restrictp",
        NULL
    },
    {
        "half_vector3_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half3", "half3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half3", "consthalf3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half3", "privatehalf3d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half3", "privateconsthalf3d",
        NULL
    },
    {
        "half_vector4_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half4*", "constanthalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "constanthalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half4*", "globalhalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "globalhalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half4*", "globalconsthalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "globalconsthalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half4*", "globalvolatilehalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "globalvolatilehalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half4*", "globalconstvolatilehalf4p",
        NULL
    },
    {
        "half_vector4_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "globalconstvolatilehalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half4*", "localhalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "localhalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half4*", "localconsthalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "localconsthalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half4*", "localvolatilehalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "localvolatilehalf4restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half4*", "localconstvolatilehalf4p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half4*", "localconstvolatilehalf4restrictp",
        NULL
    },
    {
        "half_vector4_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half4", "half4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half4", "consthalf4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half4", "privatehalf4d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half4", "privateconsthalf4d",
        NULL
    },
    {
        "half_vector8_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half8*", "constanthalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "constanthalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half8*", "globalhalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "globalhalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half8*", "globalconsthalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "globalconsthalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half8*", "globalvolatilehalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "globalvolatilehalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half8*", "globalconstvolatilehalf8p",
        NULL
    },
    {
        "half_vector8_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "globalconstvolatilehalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half8*", "localhalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "localhalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half8*", "localconsthalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "localconsthalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half8*", "localvolatilehalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "localvolatilehalf8restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half8*", "localconstvolatilehalf8p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half8*", "localconstvolatilehalf8restrictp",
        NULL
    },
    {
        "half_vector8_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half8", "half8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half8", "consthalf8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half8", "privatehalf8d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half8", "privateconsthalf8d",
        NULL
    },
    {
        "half_vector16_p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half16*", "constanthalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_CONSTANT, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "constanthalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half16*", "globalhalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "globalhalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half16*", "globalconsthalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "globalconsthalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half16*", "globalvolatilehalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "globalvolatilehalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half16*", "globalconstvolatilehalf16p",
        NULL
    },
    {
        "half_vector16_p2",
        (const char *)CL_KERNEL_ARG_ADDRESS_GLOBAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "globalconstvolatilehalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half16*", "localhalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "localhalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST), "half16*", "localconsthalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "localconsthalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE), "half16*", "localvolatilehalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "localvolatilehalf16restrictp",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE), "half16*", "localconstvolatilehalf16p",
        (const char *)CL_KERNEL_ARG_ADDRESS_LOCAL, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_CONST|CL_KERNEL_ARG_TYPE_VOLATILE|CL_KERNEL_ARG_TYPE_RESTRICT), "half16*", "localconstvolatilehalf16restrictp",
        NULL
    },
    {
        "half_vector16_d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half16", "half16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half16", "consthalf16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half16", "privatehalf16d",
        (const char *)CL_KERNEL_ARG_ADDRESS_PRIVATE, (const char *)CL_KERNEL_ARG_ACCESS_NONE, (const char *)(CL_KERNEL_ARG_TYPE_NONE), "half16", "privateconsthalf16d",
        NULL
    },
};


template<typename arg_info_t>
int test(cl_device_id deviceID, cl_context context, kernel_args_t kernel_args, cl_uint lines_count, arg_info_t arg_info, size_t total_kernels_in_program) {

    const size_t max_name_len = 512;
    cl_char name[ max_name_len ];
    cl_uint arg_count, numArgs;
    size_t i, j, size;
    int error;

    clProgramWrapper program =
    clCreateProgramWithSource(context, lines_count, kernel_args, NULL, &error);
    if ( program == NULL || error != CL_SUCCESS )
    {
        print_error( error, "Unable to create required arguments kernel program" );
        return -1;
    }

    // Compile the program
    log_info( "Building kernels...\n" );
    clBuildProgram( program, 1, &deviceID, "-cl-kernel-arg-info", NULL, NULL );

    // check for build errors and exit if things didn't work
    size_t size_ret;
    cl_build_status build_status;
    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, &size_ret);
    test_error( error, "Unable to query build status" );
    if (build_status == CL_BUILD_ERROR) {
        printf("CL_PROGRAM_BUILD_STATUS=%d\n", (int) build_status);
        error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &size_ret);
        test_error( error, "Unable to get build log size" );
        char *build_log = (char *)malloc(size_ret);
        error = clGetProgramBuildInfo(program,deviceID, CL_PROGRAM_BUILD_LOG, size_ret, build_log, &size_ret);
        test_error( error, "Unable to get build log" );
        printf("CL_PROGRAM_BUILD_LOG:\n%s\n", build_log);
        printf("CL_BUILD_ERROR. exiting\n");
        free(build_log);
        return -1;
    }

    // Lookup the number of kernels in the program.
    log_info( "Testing kernels...\n" );
    size_t total_kernels = 0;
    error = clGetProgramInfo( program, CL_PROGRAM_NUM_KERNELS, sizeof( size_t ), &total_kernels, NULL );
    test_error( error, "Unable to get program info num kernels" );

    if ( total_kernels != total_kernels_in_program )
    {
        print_error( error, "Program did not build all kernels" );
        return -1;
    }

    // Lookup the kernel names.
    size_t kernel_names_len = 0;
    error = clGetProgramInfo( program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &kernel_names_len );
    test_error( error, "Unable to get length of kernel names list." );

    size_t expected_kernel_names_len = 0;
    for ( i = 0; i < total_kernels; ++i )
    {
        expected_kernel_names_len += 1 + strlen( arg_info[ i ][ 0 ] );
    }
    if ( kernel_names_len != expected_kernel_names_len )
    {
        log_error( "Kernel names string is not the right length, expected %d, got %d\n", (int) expected_kernel_names_len, (int) kernel_names_len );
        return -1;
    }

    const size_t len = ( kernel_names_len + 1 ) * sizeof( char );
    char* kernel_names = (char*) malloc( len );
    error = clGetProgramInfo( program, CL_PROGRAM_KERNEL_NAMES, len, kernel_names, &kernel_names_len );
    test_error( error, "Unable to get kernel names list." );

    // Check to see if the kernel name array is null terminated.
    if ( kernel_names[ kernel_names_len - 1 ] != '\0' )
    {
        free( kernel_names );
        print_error( error, "Kernel name list was not null terminated" );
        return -1;
    }

    // Check to see if the correct kernel name string was returned.
    // Does the string contain each expected kernel name?
    for ( i = 0; i < total_kernels; ++i )
        if ( !strstr( kernel_names, arg_info[ i ][ 0 ] ) )
            break;
    if ( i != total_kernels )
    {
        log_error( "Kernel names string is missing \"%s\"\n", arg_info[ i ][ 0 ] );
        free( kernel_names );
        return -1;
    }

    // Are the kernel names delimited by ';'?
    if ( !strtok( kernel_names, ";" ) )
    {
        error = -1;
    }
    else
    {
        for ( i = 1; i < total_kernels; ++i )
        {
            if ( !strtok( NULL, ";" ) )
            {
                error = -1;
            }
        }
    }
    if ( error )
    {
        log_error( "Kernel names string was not properly delimited by ';'\n" );
        free( kernel_names );
        return -1;
    }
    free( kernel_names );

    // Create kernel objects and query them.
    int rc = 0;
    for ( i = 0; i < total_kernels; ++i )
    {
        int kernel_rc = 0;
        const char* kernel_name = arg_info[ i ][ 0 ];
        clKernelWrapper kernel = clCreateKernel(program, kernel_name, &error);
        if( kernel == NULL || error != CL_SUCCESS )
        {
            log_error( "ERROR: Could not get kernel: %s\n", kernel_name );
            kernel_rc = -1;
        }

        if(kernel_rc == 0)
        {
            // Determine the expected number of arguments.
            arg_count = 0;
            while (arg_info[ i ][ (ARG_INFO_FIELD_COUNT * arg_count) + 1 ] != NULL)
                ++arg_count;

            // Try to get the number of arguments.
            error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, 0, NULL, &size );
            test_error( error, "Unable to get kernel arg count param size" );
            if( size != sizeof( numArgs ) )
            {
                log_error( "ERROR: Kernel arg count param returns invalid size (expected %d, got %d) for kernel: %s\n", (int)sizeof( numArgs ), (int)size, kernel_name );
                kernel_rc = -1;
            }
        }


        if(kernel_rc == 0)
        {
            error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, sizeof( numArgs ), &numArgs, NULL );
            test_error( error, "Unable to get kernel arg count" );
            if( numArgs != arg_count )
            {
                log_error( "ERROR: Kernel arg count returned invalid value (expected %d, got %d) for kernel: %s\n", arg_count, numArgs, kernel_name );
                kernel_rc = -1;
            }
        }

        if(kernel_rc == 0)
        {
            for ( j = 0; j < numArgs; ++j )
            {

                int arg_rc = 0;
                cl_kernel_arg_address_qualifier expected_address_qualifier = (cl_kernel_arg_address_qualifier)(uintptr_t)arg_info[ i ][ (ARG_INFO_FIELD_COUNT * j) + ARG_INFO_ADDR_OFFSET ];
                cl_kernel_arg_access_qualifier expected_access_qualifier =  (cl_kernel_arg_access_qualifier)(uintptr_t)arg_info[ i ][ (ARG_INFO_FIELD_COUNT * j) + ARG_INFO_ACCESS_OFFSET ];
                cl_kernel_arg_type_qualifier expected_type_qualifier = (cl_kernel_arg_type_qualifier)(uintptr_t)arg_info[ i ][ (ARG_INFO_FIELD_COUNT * j) + ARG_INFO_TYPE_QUAL_OFFSET ];
                const char* expected_type_name = arg_info[ i ][ (ARG_INFO_FIELD_COUNT * j) + ARG_INFO_TYPE_NAME_OFFSET ];
                const char* expected_arg_name = arg_info[ i ][ (ARG_INFO_FIELD_COUNT * j) + ARG_INFO_ARG_NAME_OFFSET ];

                // Try to get the address qualifier of each argument.
                cl_kernel_arg_address_qualifier address_qualifier = 0;
                error = clGetKernelArgInfo( kernel, (cl_uint)j, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof address_qualifier, &address_qualifier, &size );
                test_error( error, "Unable to get argument address qualifier" );
                error = (address_qualifier != expected_address_qualifier);
                if ( error )
                {
                    log_error( "ERROR: Bad address qualifier, kernel: \"%s\", argument number: %d, expected \"0x%X\", got \"0x%X\"\n", kernel_name, (unsigned int)j, (unsigned int)expected_address_qualifier, (unsigned int)address_qualifier );
                    arg_rc = -1;
                }

                // Try to get the access qualifier of each argument.
                cl_kernel_arg_access_qualifier access_qualifier = 0;
                error = clGetKernelArgInfo( kernel, (cl_uint)j, CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof access_qualifier, &access_qualifier, &size );
                test_error( error, "Unable to get argument access qualifier" );
                error = (access_qualifier != expected_access_qualifier);
                if ( error )
                {
                    log_error( "ERROR: Bad access qualifier, kernel: \"%s\", argument number: %d, expected \"0x%X\", got \"0x%X\"\n", kernel_name, (unsigned int)j, (unsigned int)expected_access_qualifier, (unsigned int)access_qualifier );
                    arg_rc = -1;
                }

                // Try to get the type qualifier of each argument.
                cl_kernel_arg_type_qualifier arg_type_qualifier = 0;
                error = clGetKernelArgInfo( kernel, (cl_uint)j, CL_KERNEL_ARG_TYPE_QUALIFIER, sizeof arg_type_qualifier, &arg_type_qualifier, &size );
                test_error( error, "Unable to get argument type qualifier" );
                error = (arg_type_qualifier != expected_type_qualifier);
                if ( error )
                {
                    log_error( "ERROR: Bad type qualifier, kernel: \"%s\", argument number: %d, expected \"0x%X\", got \"0x%X\"\n", kernel_name, (unsigned int)j, (unsigned int)expected_type_qualifier, (unsigned int)arg_type_qualifier );
                    arg_rc = -1;
                }

                // Try to get the type of each argument.
                memset( name, 0, max_name_len );
                error = clGetKernelArgInfo(kernel, (cl_uint)j, CL_KERNEL_ARG_TYPE_NAME, max_name_len, name, &size );
                test_error( error, "Unable to get argument type name" );
                error = strcmp( (const char*) name, expected_type_name );
                if ( error )
                {
                    log_error( "ERROR: Bad argument type name, kernel: \"%s\", argument number: %d, expected \"%s\", got \"%s\"\n", kernel_name, (unsigned int)j, expected_type_name, name );
                    arg_rc = -1;
                }

                // Try to get the name of each argument.
                memset( name, 0, max_name_len );
                error = clGetKernelArgInfo( kernel, (cl_uint)j, CL_KERNEL_ARG_NAME, max_name_len, name, &size );
                test_error( error, "Unable to get argument name" );
                error = strcmp( (const char*) name, expected_arg_name );
                if ( error )
                {
                    log_error( "ERROR: Bad argument name, kernel: \"%s\", argument number: %d, expected \"%s\", got \"%s\"\n", kernel_name, (unsigned int)j, expected_arg_name, name );
                    arg_rc = -1;
                }

                if(arg_rc != 0) {
                    kernel_rc = -1;
                }
            }
        }

        //log_info( "%s ... %s\n",arg_info[i][0],kernel_rc == 0 ? "passed" : "failed" );
        if(kernel_rc != 0) {
            rc = -1;
        }
    }
  return rc;
}


int test_get_kernel_arg_info_compatibility( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    size_t size;
    int error;

    cl_bool supports_double = 0; // assume not
    cl_bool supports_half = 0; // assume not
  cl_bool supports_images = 0; // assume not

    // Check if this device supports images
  error = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof supports_images, &supports_images, NULL);
  test_error(error, "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

  if (supports_images) {
    log_info(" o Device supports images\n");
    log_info(" o Expecting SUCCESS when testing image kernel arguments.\n");
  }
  else {
    log_info(" o Device lacks image support\n");
    log_info(" o Not testing image kernel arguments.\n");
  }

    if (is_extension_available(deviceID, "cl_khr_fp64")) {
        log_info(" o Device claims extension 'cl_khr_fp64'\n");
        log_info(" o Expecting SUCCESS when testing double kernel arguments.\n");
        supports_double = 1;
    } else {
        cl_device_fp_config double_fp_config;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(double_fp_config), &double_fp_config, NULL);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_DOUBLE_FP_CONFIG failed");
        if (double_fp_config != 0)
            supports_double = 1;
        else {
            log_info(" o Device lacks extension 'cl_khr_fp64'\n");
            log_info(" o Not testing double kernel arguments.\n");
            supports_double = 0;
        }
    }

    if (is_extension_available(deviceID, "cl_khr_fp16")) {
        log_info(" o Device claims extension 'cl_khr_fp16'\n");
        log_info(" o Expecting SUCCESS when testing halfn* kernel arguments.\n");
        supports_half = 1;
    } else {
        log_info(" o Device lacks extension 'cl_khr_fp16'\n");
        log_info(" o Not testing halfn* kernel arguments.\n");
        supports_half = 0;
    }


  int test_failed = 0;

    // Now create a test program using required arguments
  log_info("Testing required kernel arguments...\n");
  error = test(deviceID, context, required_kernel_args, sizeof(required_kernel_args)/sizeof(required_kernel_args[0]), required_arg_info, sizeof(required_arg_info)/sizeof(required_arg_info[0]));
  test_failed = (error) ? -1 : test_failed;

  if ( supports_images ) {
    log_info("Testing optional image arguments...\n");
    error = test(deviceID, context, image_kernel_args, sizeof(image_kernel_args)/sizeof(image_kernel_args[0]), image_arg_info, sizeof(image_arg_info)/sizeof(image_arg_info[0]));
    test_failed = (error) ? -1 : test_failed;
  }

    if ( supports_double ) {
    log_info("Testing optional double arguments...\n");
    error = test(deviceID, context, double_kernel_args, sizeof(double_kernel_args)/sizeof(double_kernel_args[0]), double_arg_info, sizeof(double_arg_info)/sizeof(double_arg_info[0]));
    test_failed = (error) ? -1 : test_failed;
  }

    if ( supports_half ) {
    log_info("Testing optional half arguments...\n");
    error = test(deviceID, context, half_kernel_args, sizeof(half_kernel_args)/sizeof(half_kernel_args[0]), half_arg_info, sizeof(half_arg_info)/sizeof(half_arg_info[0]));
    test_failed = (error) ? -1 : test_failed;
  }

    return test_failed;
}


