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
#include <stdio.h>
#include <string.h>
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include <vector>

#include "procs.h"
#include "utils.h"
#include <time.h>


#ifdef CL_VERSION_2_0

static const char* block_global_scope[] =
{
    NL, "int __constant globalVar = 7;"
    NL, "int (^__constant globalBlock)(int) = ^int(int num)"
    NL, "{"
    NL, "   return globalVar * num * (1+ get_global_id(0));"
    NL, "};"
    NL, "kernel void block_global_scope(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = globalBlock(3) - 21*(tid + 1);"
    NL, "}"
    NL
};

static const char* block_kernel_scope[] =
{
    NL, "kernel void block_kernel_scope(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  multiplier = 8;"
    NL, "  res[tid] = kernelBlock(7) - 21;"
    NL, "}"
    NL
};

static const char* block_statement_scope[] =
{
    NL, "kernel void block_statement_scope(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 0;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  multiplier = 9;"
    NL, "  res[tid] = ^int(int num) { return multiplier * num; } (11) - 99;"
    NL, "}"
    NL
};

static const char* block_function_scope[] =
{
    NL, "int fnTest(int a)"
    NL, "{"
    NL, "  int localVar = 17;"
    NL, "  int (^functionBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return localVar * num;"
    NL, "  };"
    NL, "  return 111 - functionBlock(a+1);"
    NL, "}"
    NL, "kernel void block_function_scope(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = fnTest(5) - 9;"
    NL, "}"
    NL
};

static const char* block_nested_scope[] =
{
    NL, "kernel void block_nested_scope(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    int (^innerBlock)(int) = ^(int n)"
    NL, "    {"
    NL, "      return multiplier * n;"
    NL, "    };"
    NL, "    return num * innerBlock(23);"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  multiplier = 8;"
    NL, "  res[tid] = kernelBlock(13) - 897;"
    NL, "}"
    NL
};

static const char* block_arg_struct[] =
{
    NL, "struct two_ints {"
    NL, "    short x;"
    NL, "    long y;"
    NL, "};"
    NL, "struct two_structs {"
    NL, "    struct two_ints a;"
    NL, "    struct two_ints b;"
    NL, "};"
    NL, "kernel void block_arg_struct(__global int* res)"
    NL, "{"
    NL, "  int (^kernelBlock)(struct two_ints, struct two_structs) = ^int(struct two_ints ti, struct two_structs ts)"
    NL, "  {"
    NL, "    return ti.x * ti.y * ts.a.x * ts.a.y * ts.b.x * ts.b.y;"
    NL, "  };"
    NL, "  struct two_ints i;"
    NL, "  i.x = 2;"
    NL, "  i.y = 3;"
    NL, "  struct two_structs s;"
    NL, "  s.a.x = 4;"
    NL, "  s.a.y = 5;"
    NL, "  s.b.x = 6;"
    NL, "  s.b.y = 7;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = kernelBlock(i,s) - 5040;"
    NL, "}"
    NL
};

static const char* block_arg_types_mix[] =
{
    NL, "union number {"
    NL, "    long l;"
    NL, "    float f;"
    NL, "};"
    NL, "enum color {"
    NL, "    RED = 0,"
    NL, "    GREEN,"
    NL, "    BLUE" // Using this value - it is actualy "2"
    NL, "};"
    NL, "typedef int _INT ;"
    NL, "typedef char _ACHAR[3] ;"
    NL, "kernel void block_arg_types_mix(__global int* res)"
    NL, "{"
    NL, "  int (^kernelBlock)(_INT, _ACHAR, union number, enum color, int, int, int, int, int, int, int, int, int, int, int, int, int) ="
    NL, "    ^int(_INT bi, _ACHAR bch, union number bn, enum color bc, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8,"
    NL, "      int i9, int i10, int i11, int i12, int i13)"
    NL, "  {"
    NL, "    return bi * bch[0] * bch[1] * bch[2] * bn.l * bc - i1 - i2 - i3 - i4 - i5 - i6 - i7 - i8 - i9 - i10 - i11 - i12 - i13;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  _INT x = -5;"
    NL, "  _ACHAR char_arr = { 1, 2, 3 };"
    NL, "  union number n;"
    NL, "  n.l = 4;"
    NL, "  enum color c = BLUE;"
    NL, "  res[tid] = kernelBlock(x,char_arr,n,c,1,2,3,4,5,6,7,8,9,10,11,12,13) + 331;"
    NL, "}"
    NL
};

static const char* block_arg_pointer[] =
{
    NL, "struct two_ints {"
    NL, "    short x;"
    NL, "    long y;"
    NL, "};"
    NL, "kernel void block_arg_pointer(__global int* res)"
    NL, "{"
    NL, "  int (^kernelBlock)(struct two_ints*, struct two_ints*, int*, int*) = "
    NL, "    ^int(struct two_ints* bs1, struct two_ints* bs2, int* bi1, int* bi2)"
    NL, "  {"
    NL, "    return (*bs1).x * (*bs1).y * (*bs2).x * (*bs2).y * (*bi1) * (*bi2);"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  struct two_ints s[2];"
    NL, "  s[0].x = 4;"
    NL, "  s[0].y = 5;"
    NL, "  struct two_ints* ps = s + 1;"
    NL, "  (*ps).x = 6;"
    NL, "  (*ps).y = 7;"
    NL, "  int i = 2;"
    NL, "  int * pi = &i;"
    NL, "  res[tid] = kernelBlock(s,ps,&i,pi) - 3360;"
    NL, "}"
    NL
};

static const char* block_arg_global_p[] =
{
    NL, "kernel void block_arg_global_p(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  typedef __global int* int_ptr_to_global_t;"
    NL, "  int_ptr_to_global_t (^kernelBlock)(__global int*, int) =^ int_ptr_to_global_t (__global int* bres, int btid)"
    NL, "  {"
    NL, "    bres[tid] = 5;"
    NL, "    return bres;"
    NL, "  };"
    NL, "  res = kernelBlock(res, tid);"
    NL, "  res[tid] -= 5;"
    NL, "}"
    NL
};

static const char* block_arg_const_p[] =
{
    NL, "constant int ci = 8;"
    NL, "kernel void block_arg_const_p(__global int* res)"
    NL, "{"
    NL, "  __constant int* (^kernelBlock)(__constant int*) = ^(__constant int* bpci)"
    NL, "  {"
    NL, "    return bpci;"
    NL, "  };"
    NL, "  constant int* pci = &ci;"
    NL, "  constant int* pci_check;"
    NL, "  pci_check = kernelBlock(pci);"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = pci == pci_check ? 0 : -1;"
    NL, "}"
    NL
};

static const char* block_ret_struct[] =
{
    NL, "kernel void block_ret_struct(__global int* res)"
    NL, "{"
    NL, "  struct A {"
    NL, "      int a;"
    NL, "  };      "
    NL, "  struct A (^kernelBlock)(struct A) = ^struct A(struct A a)"
    NL, "  {        "
    NL, "    a.a = 6;"
    NL, "    return a;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  struct A aa;"
    NL, "  aa.a = 5;"
    NL, "  res[tid] = kernelBlock(aa).a - 6;"
    NL, "}"
    NL
};

static const char* block_arg_global_var[] =
{
    NL, "constant int gi = 8;"
    NL, "kernel void block_arg_global_var(__global int* res)"
    NL, "{"
    NL, "  int (^kernelBlock)(int) = ^(int bgi)"
    NL, "  {"
    NL, "    return bgi - 8;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = kernelBlock(gi);"
    NL, "}"
    NL
};

static const char* block_in_for_init[] =
{
    NL, "kernel void block_in_for_init(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 27;"
    NL, "  for(int i=kernelBlock(9); i>0; i--)"
    NL, "  {"
    NL, "       res[tid]--;"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_in_for_cond[] =
{
    NL, "kernel void block_in_for_cond(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 39;"
    NL, "  for(int i=0; i<kernelBlock(13); i++)"
    NL, "  {"
    NL, "       res[tid]--;"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_in_for_iter[] =
{
    NL, "kernel void block_in_for_iter(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 2;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 4;"
    NL, "  for(int i=2; i<17; i=kernelBlock(i))"
    NL, "  {"
    NL, "       res[tid]--;"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_in_while_cond[] =
{
    NL, "kernel void block_in_while_cond(__global int* res)"
    NL, "{"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return res[num];"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 27*(tid+1);"
    NL, "  while(kernelBlock(tid))"
    NL, "  {"
    NL, "      res[tid]--;"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_in_while_body[] =
{
    NL, "kernel void block_in_while_body(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  int i = 7;"
    NL, "  res[tid] = 3*(7+6+5+4+3+2+1);"
    NL, "  while(i)"
    NL, "  {"
    NL, "      res[tid]-=kernelBlock(i--);"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_in_do_while_body[] =
{
    NL, "kernel void block_in_do_while_body(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  int i = 100;"
    NL, "  res[tid] = 3*5050;"
    NL, "  do"
    NL, "  {"
    NL, "      int (^kernelBlock)(int) = ^(int num)"
    NL, "      {"
    NL, "          return num * multiplier;"
    NL, "      };"
    NL, "      res[tid]-=kernelBlock(i--);"
    NL, "  } while(i);"
    NL, "}"
    NL
};

static const char* block_cond_statement[] =
{
    NL, "kernel void block_cond_statement(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 2;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 120;"
    NL, "  res[tid] = (kernelBlock(2) == 4) ? res[tid] - 30              : res[tid] - 1;"
    NL, "  res[tid] = (kernelBlock(2) == 5) ? res[tid] - 3               : res[tid] - 30;"
    NL, "  res[tid] = (1)                   ? res[tid] - kernelBlock(15) : res[tid] - 7;"
    NL, "  res[tid] = (0)                   ? res[tid] - 13              : res[tid] - kernelBlock(15);"
    NL, "}"
    NL
};

static const char* block_in_if_cond[] =
{
    NL, "kernel void block_in_if_cond(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 2;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 7;"
    NL, "  if (kernelBlock(5))"
    NL, "  {"
    NL, "      res[tid]-= 3;"
    NL, "  }"
    NL, "  if (kernelBlock(0))"
    NL, "  {"
    NL, "      res[tid]-= 2;"
    NL, "  }"
    NL, "  else"
    NL, "  {"
    NL, "      res[tid]-= 4;"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_in_if_branch[] =
{
    NL, "kernel void block_in_if_branch(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 2;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 7;"
    NL, "  if (kernelBlock(5))"
    NL, "  {"
    NL, "      res[tid]-= ^(int num){ return num - 1; }(4);" // res[tid]-=3;
    NL, "  }"
    NL, "  if (kernelBlock(0))"
    NL, "  {"
    NL, "      res[tid]-= ^(int num){ return num - 1; }(3);" // res[tid]-=2;
    NL, "  }"
    NL, "  else"
    NL, "  {"
    NL, "      int (^ifBlock)(int) = ^(int num)"
    NL, "      {"
    NL, "          return num + 1;"
    NL, "      };"
    NL, "      res[tid]-= ifBlock(3);"                     // res[tid]-=4;
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_switch_cond[] =
{
    NL, "kernel void block_switch_cond(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 2;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 12;"
    NL, "  int i = 1;"
    NL, "  while(i <= 3)"
    NL, "  {"
    NL, "      switch (kernelBlock(i))"
    NL, "      {"
    NL, "          case 2:"
    NL, "              res[tid] = res[tid] - 2;"
    NL, "              break;"
    NL, "          case 4:"
    NL, "              res[tid] = res[tid] - 4;"
    NL, "              break;"
    NL, "          case 6:"
    NL, "              res[tid] = res[tid] - 6;"
    NL, "              break;"
    NL, "          default:"
    NL, "              break;"
    NL, "      }"
    NL, "      i++;"
    NL, "  }"
    NL, "}"
    NL
};

static const char* block_switch_case[] =
{
    NL, "kernel void block_switch_case(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 2;"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    return num * multiplier;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = 12;"
    NL, "  int i = 1;"
    NL, "  while(i <= 3)"
    NL, "  {"
    NL, "      switch (kernelBlock(i))"
    NL, "      {"
    NL, "          case 2:"
    NL, "              res[tid]-=^(int num){ return num - 1; }(3);" // res[tid]-=2;
    NL, "              break;"
    NL, "          case 4:"
    NL, "          {"
    NL, "              int (^caseBlock)(int) = ^(int num)"
    NL, "              {"
    NL, "                  return num + 1;"
    NL, "              };"
    NL, "              res[tid]-=caseBlock(3);"                   // res[tid]-=4;
    NL, "              break;"
    NL, "          }"
    NL, "          case 6:"
    NL, "              res[tid]-=kernelBlock(3);"                 // res[tid]-=6;
    NL, "              break;"
    NL, "          default:"
    NL, "              break;"
    NL, "      }"
    NL, "      i++;"
    NL, "  }"
    NL, "}"
    NL
};

// Accessing data from Block

static const char* block_access_program_data[] =
{
    NL, "int __constant globalVar1 = 7;"
    NL, "int __constant globalVar2 = 11;"
    NL, "int __constant globalVar3 = 13;"
    NL, "int (^__constant globalBlock)(int) = ^int(int num)"
    NL, "{"
    NL, "    return globalVar1 * num;"
    NL, "};"
    NL, "kernel void block_access_program_data(__global int* res)"
    NL, "{"
    NL, "    int (^ kernelBlock)(int) = ^int(int num)"
    NL, "    {"
    NL, "        return globalVar2 * num;"
    NL, "    };"
    NL, "    size_t tid = get_global_id(0);"
    NL, "    res[tid] = tid + 1;"
    NL, "    res[tid] = globalBlock(res[tid]);"
    NL, "    res[tid] = kernelBlock(res[tid]);"
    NL, "    res[tid] = ^(int num){ return globalVar3*num; }(res[tid]) - (7*11*13)*(tid + 1);"
    NL, "}"
    NL
};

static const char* block_access_kernel_data[] =
{
    NL, "kernel void block_access_kernel_data(__global int* res)"
    NL, "{"
    NL, "    int var1 = 7;"
    NL, "    int var2 = 11;"
    NL, "    int var3 = 13;"
    NL, "    int (^ kernelBlock)(int) = ^int(int num)"
    NL, "    {"
    NL, "        int (^ nestedBlock)(int) = ^int (int num)"
    NL, "        {"
    NL, "            return var1 * num;"
    NL, "        };"
    NL, "        return var2 * nestedBlock(num);"
    NL, "    };"
    NL, "    size_t tid = get_global_id(0);"
    NL, "    res[tid] = tid + 1;"
    NL, "    res[tid] = kernelBlock(res[tid]);"
    NL, "    res[tid] = ^(int num){ return var3*num; }(res[tid]) - (7*11*13)*(tid + 1);"
    NL, "}"
    NL
};

static const char* block_access_chained_data[] =
{
    NL, "kernel void block_access_chained_data(__global int* res)"
    NL, "{"
    NL, "    int (^ kernelBlock)(int) = ^int(int num)"
    NL, "    {"
    NL, "        int var1 = 7;"
    NL, "        int var2 = 11;"
    NL, "        int var3 = 13;"
    NL, "        int (^ nestedBlock1)(int) = ^int (int num)"
    NL, "        {"
    NL, "            int (^ nestedBlock2) (int) = ^int (int num)"
    NL, "            {"
    NL, "                return var2 * ^(int num){ return var3*num; }(num);"
    NL, "            };"
    NL, "            return var1 * nestedBlock2(num);"
    NL, "        };"
    NL, "        return nestedBlock1(num);"
    NL, "    };"
    NL, "    size_t tid = get_global_id(0);"
    NL, "    res[tid] = tid + 1;"
    NL, "    res[tid] = kernelBlock(res[tid]) - (7*11*13)*(tid + 1);"
    NL, "}"
    NL
};

static const char* block_access_volatile_data[] =
{
    NL, "kernel void block_access_volatile_data(__global int* res)"
    NL, "{"
    NL, "    int var1 = 7;"
    NL, "    int var2 = 11;"
    NL, "    volatile int var3 = 13;"
    NL, ""
    NL, "    int (^ kernelBlock)(int) = ^int(int num)"
    NL, "    {"
    NL, "        int (^ nestedBlock)(int) = ^int (int num)"
    NL, "        {"
    NL, "            return var1 * num;"
    NL, "        };"
    NL, "        return var2 * nestedBlock(num);"
    NL, "    };"
    NL, "    size_t tid = get_global_id(0);"
    NL, "    res[tid] = tid + 1;"
    NL, "    res[tid] = kernelBlock(res[tid]);"
    NL, "    res[tid] = ^(int num){ return var3*num; }(res[tid]) - (7*11*13)*(tid + 1);"
    NL, "}"
    NL
};

static const char* block_typedef_kernel[] =
{
    NL, "kernel void block_typedef_kernel(__global int* res)"
    NL, "{"
    NL, "  typedef int* (^block_t)(int*);"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  int i[4] = { 3, 4, 4, 1 };"
    NL, "  int *temp = i; // workaround clang bug"
    NL, "  block_t kernelBlock = ^(int* pi)"
    NL, "  {"
    NL, "    block_t b = ^(int* n) { return n - 1; };"
    NL, "    return pi + *(b(temp+4));"
    NL, "  };"
    NL, "  switch (*(kernelBlock(i))) {"
    NL, "    case 4:"
    NL, "      res[tid] += *(kernelBlock(i+1));"
    NL, "      break;"
    NL, "    default:"
    NL, "      res[tid] = -100;"
    NL, "      break;"
    NL, "  }"
    NL, "  res[tid] += *(kernelBlock(i)) - 7;"
    NL, "}"
    NL
};

static const char* block_typedef_func[] =
{
    NL, "int func(int fi)"
    NL, "{"
    NL, "  typedef int (^block_t)(int);"
    NL, "  const block_t funcBlock = ^(int bi)"
    NL, "  {"
    NL, "    typedef short (^block2_t)(short);"
    NL, "    block2_t nestedBlock = ^(short ni)"
    NL, "    {"
    NL, "      return (short)(ni - 1);"
    NL, "    };"
    NL, "    return bi * nestedBlock(3);"
    NL, "  };"
    NL, "  return funcBlock(fi * 2);"
    NL, "}"
    NL, "kernel void block_typedef_func(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = func(1) - 4;"
    NL, "}"
    NL
};

static const char* block_typedef_stmnt_if[] =
{
    NL, "kernel void block_typedef_stmnt_if(__global int* res)"
    NL, "{      "
    NL, "  int flag = 1;"
    NL, "  int sum = 0;"
    NL, "  if (flag) {"
    NL, "    typedef int (^block_t)(int);"
    NL, "    const block_t kernelBlock = ^(int bi)"
    NL, "    {"
    NL, "      block_t b = ^(int bi)"
    NL, "      {"
    NL, "        return bi + 1;"
    NL, "      };"
    NL, "      return bi + b(1);"
    NL, "    };"
    NL, "    sum = kernelBlock(sum);"
    NL, "  }"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = sum - 2;"
    NL, "}"
    NL
};

static const char* block_typedef_loop[] =
{
    NL, "kernel void block_typedef_loop(__global int* res)"
    NL, "{      "
    NL, "  int sum = -1;"
    NL, "  for (int i = 0; i < 3; i++) {"
    NL, "    typedef int (^block_t)(void);"
    NL, "    const block_t kernelBlock = ^()"
    NL, "    {"
    NL, "      return i + 1;"
    NL, "    };"
    NL, "    sum += kernelBlock();"
    NL, "  }"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = sum - 5;"
    NL, "}"
    NL
};

static const char* block_typedef_mltpl_func[] =
{
    NL, "int func(int fi)"
    NL, "{"
    NL, "  typedef int (^block_t)(int);"
    NL, "  typedef int (^block2_t)(int);"
    NL, "  const block_t funcBlock1 = ^(int bi) { return bi; };"
    NL, "  const block2_t funcBlock2 = ^(int bi)"
    NL, "  {"
    NL, "    typedef short (^block3_t)(short);"
    NL, "    typedef short (^block4_t)(short);"
    NL, "    const block3_t nestedBlock1 = ^(short ni)"
    NL, "    {"
    NL, "      return (short)(ni - 1);"
    NL, "    };"
    NL, "    const block4_t nestedBlock2 = ^(short ni)"
    NL, "    {"
    NL, "      return (short)(ni - 2);"
    NL, "    };"
    NL, "    return bi * nestedBlock1(3) * nestedBlock2(3);"
    NL, "  };"
    NL, "  return funcBlock2(fi * 2) + funcBlock1(1);"
    NL, "}"
    NL, "kernel void block_typedef_mltpl_func(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  typedef int (^block1_t)(int);"
    NL, "  typedef int (^block2_t)(int);"
    NL, "  const block1_t kernelBlock1 = ^(int bi) { return bi + 8; };"
    NL, "  const block2_t kernelBlock2 = ^(int bi) { return bi + 3; };"
    NL, "  res[tid] = func(1) -  kernelBlock1(2) / kernelBlock2(-1);"
    NL, "}"
    NL
};

static const char* block_typedef_mltpl_stmnt[] =
{
    NL, "kernel void block_typedef_mltpl_stmnt(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  int a;"
    NL, "  do"
    NL, "  {"
    NL, "    typedef float (^blockf_t)(float);"
    NL, "    typedef int (^blocki_t)(int);"
    NL, "    const blockf_t blockF = ^(float bi) { return (float)(bi + 3.3); };"
    NL, "    const blocki_t blockI = ^(int bi) { return bi + 2; };"
    NL, "    if ((blockF(.0)-blockI(0)) > 0)"
    NL, "    {"
    NL, "      typedef uint (^block_t)(uint);"
    NL, "      const block_t nestedBlock = ^(uint bi) { return (uint)(bi + 4); };"
    NL, "      a = nestedBlock(1) + nestedBlock(2);"
    NL, "      break;"
    NL, "    }"
    NL, "  } while(1);  "
    NL, "  res[tid] = a - 11;"
    NL, "}"
    NL
};

static const char* block_typedef_mltpl_g[] =
{
    NL, "typedef int (^block1_t)(float, int); "
    NL, "constant block1_t b1 = ^(float fi, int ii) { return (int)(ii + fi); };"
    NL, "typedef int (^block2_t)(float, int);"
    NL, "constant block2_t b2 = ^(float fi, int ii) { return (int)(ii + fi); };"
    NL, "typedef float (^block3_t)(int, int);"
    NL, "constant block3_t b3 = ^(int i1, int i2) { return (float)(i1 + i2); };"
    NL, "typedef int (^block4_t)(float, float);"
    NL, "kernel void block_typedef_mltpl_g(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  block4_t b4 = ^(float f1, float f2) { return (int)(f1 + f2); };"
    NL, "  res[tid] = b1(1.1, b2(1.1, 1)) - b4(b3(1,1), 1.1);"
    NL, "}"
    NL
};

static const char* block_literal[] =
{
    NL, "int func()"
    NL, "{"
    NL, "  return ^(int i) {"
    NL, "    return ^(ushort us)"
    NL, "    {"
    NL, "      return (int)us + i;"
    NL, "    }(3);"
    NL, "  }(7) - 10;"
    NL, "}"
    NL, "kernel void block_literal(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = func();"
    NL, "}"
    NL
};

static const char* block_complex[] =
{
    NL, "kernel void block_complex(__global int* res)"
    NL, "{"
    NL, "  int (^kernelBlock)(int) = ^(int num)"
    NL, "  {"
    NL, "    int result = 1;"
    NL, "    for (int i = 0; i < num; i++)"
    NL, "    {"
    NL, "      switch(i)"
    NL, "      {"
    NL, "      case 0:"
    NL, "      case 1:"
    NL, "      case 2:"
    NL, "        result += i;"
    NL, "        break;"
    NL, "      case 3:"
    NL, "        if (result < num)"
    NL, "          result += i;"
    NL, "        else"
    NL, "          result += i * 2;"
    NL, "        break;"
    NL, "      case 4:"
    NL, "        while (1)"
    NL, "        {"
    NL, "          result++;"
    NL, "          if (result)"
    NL, "            goto ret;"
    NL, "        }"
    NL, "        break;"
    NL, "      default:"
    NL, "        return 777;"
    NL, "      }"
    NL, "    }"
    NL, "    ret: ;"
    NL, "    while (num) {"
    NL, "      num--;"
    NL, "      if (num % 2 == 0)"
    NL, "        continue;"
    NL, "      result++;"
    NL, "    }"
    NL, "    return result;"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = kernelBlock(7) - 11;"
    NL, "}"
    NL
};

static const char* block_empty[] =
{
    NL, "kernel void block_empty(__global int* res)"
    NL, "{"
    NL, "  void (^kernelBlock)(void) = ^(){};"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  kernelBlock();"
    NL, "  res[tid] = 0;"
    NL, "}"
    NL
};

static const char* block_builtin[] =
{
    NL, "kernel void block_builtin(__global int* res)"
    NL, "{"
    NL, "  int b = 3;"
    NL, "  int (^kernelBlock)(int) = ^(int a)"
    NL, "  {"
    NL, "    return (int)abs(a - b);"
    NL, "  };"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  res[tid] = -1;"
    NL, "  res[tid] = kernelBlock(2) - 1;"
    NL, "}"
    NL
};

static const char* block_barrier[] =
{
    NL, "kernel void block_barrier(__global int* res)"
    NL, "{"
    NL, "  int b = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t lsz = get_local_size(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t idx = gid*lsz;"
    NL, ""
    NL, "  res[tid]=lsz;"
    NL, "  barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, "  int (^kernelBlock)(int) = ^(int a)"
    NL, "  {"
    NL, "    atomic_dec(res+idx);"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, "    return (int)abs(a - b) - (res[idx] != 0 ? 0 : 1);"
    NL, "  };"
    NL, ""
    NL, "  int d = kernelBlock(2);"
    NL, "  res[tid] = d;"
    NL, "}"
    NL
};



static const kernel_src sources_execute_block[] =
{
    // Simple blocks
    KERNEL(block_global_scope),
    KERNEL(block_kernel_scope),
    KERNEL(block_statement_scope),
    KERNEL(block_function_scope),
    KERNEL(block_nested_scope),

    // Kernels with Block in for/while/if/switch
    KERNEL(block_in_for_init),
    KERNEL(block_in_for_cond),
    KERNEL(block_in_for_iter),
    KERNEL(block_in_while_cond),
    KERNEL(block_in_while_body),
    KERNEL(block_in_do_while_body),
    KERNEL(block_cond_statement),
    KERNEL(block_in_if_cond),
    KERNEL(block_in_if_branch),
    KERNEL(block_switch_cond),
    KERNEL(block_switch_case),
    KERNEL(block_literal),

    // Accessing data from block
    KERNEL(block_access_program_data),
    KERNEL(block_access_kernel_data),
    KERNEL(block_access_chained_data),
    KERNEL(block_access_volatile_data),

    // Block args
    KERNEL(block_arg_struct),
    KERNEL(block_arg_types_mix),
    KERNEL(block_arg_pointer),
    KERNEL(block_arg_global_p),
    KERNEL(block_arg_const_p),
    KERNEL(block_ret_struct),
    KERNEL(block_arg_global_var),

    // Block in typedef
    KERNEL(block_typedef_kernel),
    KERNEL(block_typedef_func),
    KERNEL(block_typedef_stmnt_if),
    KERNEL(block_typedef_loop),
    KERNEL(block_typedef_mltpl_func),
    KERNEL(block_typedef_mltpl_stmnt),
    KERNEL(block_typedef_mltpl_g),

    // Non - trivial blocks
    KERNEL(block_complex),
    KERNEL(block_empty),
    KERNEL(block_builtin),
    KERNEL(block_barrier),

};
static const size_t num_kernels_execute_block = arr_size(sources_execute_block);

static int check_kernel_results(cl_int* results, cl_int len)
{
    for(cl_int i = 0; i < len; ++i)
    {
        if(results[i] != 0) return i;
    }
    return -1;
}

int test_execute_block(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t i;
    size_t ret_len;
    cl_int n, err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    cl_int kernel_results[MAX_GWS] = {0xDEADBEEF};

    size_t max_local_size = 1;
    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed");

    size_t global_size = MAX_GWS;
    size_t local_size = (max_local_size > global_size/16) ? global_size/16 : max_local_size;

    size_t failCnt = 0;
    for(i = 0; i < num_kernels_execute_block; ++i)
    {
        if (!gKernelName.empty() && gKernelName != sources_execute_block[i].kernel_name)
            continue;

        log_info("Running '%s' kernel (%d of %d) ...\n", sources_execute_block[i].kernel_name, i + 1, num_kernels_execute_block);
        err_ret = run_n_kernel_args(context, queue, sources_execute_block[i].lines, sources_execute_block[i].num_lines, sources_execute_block[i].kernel_name, local_size, global_size, kernel_results, sizeof(kernel_results), 0, NULL);
        if(check_error(err_ret, "'%s' kernel execution failed", sources_execute_block[i].kernel_name)) { ++failCnt; res = -1; }
        else if((n = check_kernel_results(kernel_results, arr_size(kernel_results))) >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d] returned %d expected 0", sources_execute_block[i].kernel_name, n, kernel_results[n])) { ++failCnt; res = -1; }
        else log_info("'%s' kernel is OK.\n", sources_execute_block[i].kernel_name);
    }

    if (failCnt > 0)
    {
      log_error("ERROR: %d of %d kernels failed.\n", failCnt, num_kernels_execute_block);
    }

    return res;
}


#endif

