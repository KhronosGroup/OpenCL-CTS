//
// Copyright (c) 2021 The Khronos Group Inc.
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

#ifndef INTEGER_OPS_TEST_INFO_H
#define INTEGER_OPS_TEST_INFO_H

#include "conversions.h"
#include "testHarness.h"

// TODO: expand usage to other tests.

template <typename T> struct TestInfo
{
};
template <> struct TestInfo<cl_char>
{
    static const ExplicitType explicitType = kChar;
    static constexpr const char* deviceTypeName = "char";
    static constexpr const char* deviceTypeNameSigned = "char";
    static constexpr const char* deviceTypeNameUnsigned = "uchar";
};
template <> struct TestInfo<cl_uchar>
{
    static const ExplicitType explicitType = kUChar;
    static constexpr const char* deviceTypeName = "uchar";
    static constexpr const char* deviceTypeNameSigned = "char";
    static constexpr const char* deviceTypeNameUnsigned = "uchar";
};
template <> struct TestInfo<cl_short>
{
    static const ExplicitType explicitType = kShort;
    static constexpr const char* deviceTypeName = "short";
    static constexpr const char* deviceTypeNameSigned = "short";
    static constexpr const char* deviceTypeNameUnsigned = "ushort";
};
template <> struct TestInfo<cl_ushort>
{
    static const ExplicitType explicitType = kUShort;
    static constexpr const char* deviceTypeName = "ushort";
    static constexpr const char* deviceTypeNameSigned = "short";
    static constexpr const char* deviceTypeNameUnsigned = "ushort";
};
template <> struct TestInfo<cl_int>
{
    static const ExplicitType explicitType = kInt;
    static constexpr const char* deviceTypeName = "int";
    static constexpr const char* deviceTypeNameSigned = "int";
    static constexpr const char* deviceTypeNameUnsigned = "uint";
};
template <> struct TestInfo<cl_uint>
{
    static const ExplicitType explicitType = kUInt;
    static constexpr const char* deviceTypeName = "uint";
    static constexpr const char* deviceTypeNameSigned = "int";
    static constexpr const char* deviceTypeNameUnsigned = "uint";
};
template <> struct TestInfo<cl_long>
{
    static const ExplicitType explicitType = kLong;
    static constexpr const char* deviceTypeName = "long";
    static constexpr const char* deviceTypeNameSigned = "long";
    static constexpr const char* deviceTypeNameUnsigned = "ulong";
};
template <> struct TestInfo<cl_ulong>
{
    static const ExplicitType explicitType = kULong;
    static constexpr const char* deviceTypeName = "ulong";
    static constexpr const char* deviceTypeNameSigned = "long";
    static constexpr const char* deviceTypeNameUnsigned = "ulong";
};

template <typename T>
static void fill_vector_with_random_data(std::vector<T>& v)
{
    MTdataHolder d(gRandomSeed);
    generate_random_data(TestInfo<T>::explicitType, v.size(), d, v.data());
}

#endif /* INTEGER_OPS_TEST_INFO_H */
