// Copyright (c) 2024-2026 The Khronos Group Inc.
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

#include "cooperative_matrix.hpp"

#include <CL/cl_platform.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"
#include "harness/mt19937.h"

#include "halffp.hpp"
#include "program.hpp"

extern const TestContext *gTestContext;


uint32_t elementsPerStride(const Layout layout, const uint32_t rows,
                           const uint32_t cols)
{
    switch (layout)
    {
        case Layout::RowMajor: return cols;
        case Layout::ColumnMajor: return rows;
    }
    assert(false && "Unreachable");
    std::abort();
}

uint32_t numberOfStrides(const Layout layout, const uint32_t rows,
                         const uint32_t cols)
{
    switch (layout)
    {
        case Layout::RowMajor: return rows;
        case Layout::ColumnMajor: return cols;
    }
    assert(false && "Unreachable");
    std::abort();
}


uint32_t elementSizeOf(const cl_device_cooperative_matrix_component_type_khr t)
{
    switch (t)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR: return 2;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR: return 4;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: return 8;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR: return 1;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR: return 2;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR: return 4;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR: return 8;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR: return 1;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR: return 2;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR: return 4;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR: return 8;
    }
    assert(false && "Unreachable");
    std::abort();
}

bool isFloatType(const cl_device_cooperative_matrix_component_type_khr t)
{
    return t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR;
}

bool isSignedType(const cl_device_cooperative_matrix_component_type_khr t)
{
    return t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR
        || t == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR;
}

// Example.
//
// Input buffer has a i64 vec3 element type, a stride of 4 elements, and there
// are 4 strides in total. Moreover, the vec3 is stored as a vec4 respecting the
// OpenCL C Specification s6.3.5.
//
// Stride 0
// +------------+------------+------------+------------+---------+
// |  0  1  2 _ |  3  4  5 _ |  6  7  8 _ |  9 10 11 _ | Padding |
// +------------+------------+------------+------------+---------+
// Stride 1
// +------------+------------+------------+------------+---------+
// | 12 13 14 _ | 15 16 17 _ | 18 19 20 _ | 21 22 23 _ | Padding |
// +------------+------------+------------+------------+---------+
// Stride 2
// +------------+------------+------------+------------+---------+
// | 24 25 26 _ | 27 28 29 _ | 30 31 32 _ | 33 34 35 _ | Padding |
// +------------+------------+------------+------------+---------+
// Stride 3
// +------------+------------+------------+------------+---------+
// | 36 37 38 _ | 39 40 41 _ | 42 43 44 _ | 45 46 47 _ | Padding |
// +------------+------------+------------+------------+---------+
//
// We load the input buffer into a 4x4 matrix with an f32 element type.
// The result depends on the layout argument.
//
// Row major               Column major
// +----+----+----+----+   +----+----+----+----+
// | 0.0  1.0  2.0  3.0|   | 0.0  4.0  8.0 12.0|
// +----+----+----+----+   +----+----+----+----+
// |12.0 13.0 14.0 15.0|   | 1.0  5.0  9.0 13.0|
// +----+----+----+----+   +----+----+----+----+
// |24.0 25.0 26.0 27.0|   | 2.0  6.0 10.0 14.0|
// +----+----+----+----+   +----+----+----+----+
// |36.0 37.0 38.0 39.0|   | 3.0  7.0 11.0 15.0|
// +----+----+----+----+   +----+----+----+----+
Matrix Matrix::cooperativeMatrixLoad(
    const cl_device_cooperative_matrix_component_type_khr matElementType,
    const uint32_t matRows, const uint32_t matCols, SemBuffer &buf,
    const Layout layout, const uint32_t stride)
{
    Matrix mat(matElementType, matRows, matCols);
    const uint32_t bufStrideSize = bufferStrideSizeOf(buf.descriptor);
    const uint32_t matElementSize = elementSizeOf(matElementType);

    for (uint32_t r = 0; r < matRows; ++r)
    {
        for (uint32_t c = 0; c < matCols; ++c)
        {
            const uint32_t dstOffset = (r * matCols + c) * matElementSize;
            uint32_t srcOffset = 0;
            switch (layout)
            {
                case Layout::RowMajor: {
                    srcOffset = (stride == 0 ? 0u : r * bufStrideSize)
                        + c * matElementSize;
                    break;
                }
                case Layout::ColumnMajor: {
                    srcOffset = (stride == 0 ? 0u : c * bufStrideSize)
                        + r * matElementSize;
                    break;
                }
            }
            memcpy(mat.data.data() + dstOffset, buf.data.data() + srcOffset,
                   matElementSize);
        }
    }

    return mat;
}

// Example.
//
// 4x4 matrix with an f32 element type.
//
// +----+----+----+----+
// | 0.0  1.0  2.0  3.0|
// +----+----+----+----+
// |12.0 13.0 14.0 15.0|
// +----+----+----+----+
// |24.0 25.0 26.0 27.0|
// +----+----+----+----+
// |36.0 37.0 38.0 39.0|
// +----+----+----+----+
//
// The output buffer have an f32 element type and a stride of 4 elements.
//
// Row major                         Column major
//
// Stride 0                          Stride 0
// +----+----+----+----+---------+   +----+----+----+----+---------+
// |  0 |  1 |  2 |  3 | Padding |   |  0 | 12 | 24 | 36 | Padding |
// +----+----+----+----+---------+   +----+----+----+----+---------+
// Stride 1                          Stride 1
// +----+----+----+----+---------+   +----+----+----+----+---------+
// | 12 | 13 | 14 | 15 | Padding |   |  1 | 13 | 25 | 37 | Padding |
// +----+----+----+----+---------+   +----+----+----+----+---------+
// Stride 2                          Stride 2
// +----+----+----+----+---------+   +----+----+----+----+---------+
// | 24 | 25 | 26 | 27 | Padding |   |  2 | 14 | 26 | 38 | Padding |
// +----+----+----+----+---------+   +----+----+----+----+---------+
// Stride 3                          Stride 3
// +----+----+----+----+---------+   +----+----+----+----+---------+
// | 36 | 37 | 38 | 39 | Padding |   |  3 | 15 | 27 | 39 | Padding |
// +----+----+----+----+---------+   +----+----+----+----+---------+
void Matrix::cooperativeMatrixStore(SemBuffer &buf, const Matrix &mat,
                                    Layout layout, uint32_t stride)
{
    const uint32_t bufStrideSize = bufferStrideSizeOf(buf.descriptor);
    const uint32_t matElementSize = elementSizeOf(mat.elementType);

    for (uint32_t r = 0; r < mat.nRows; ++r)
    {
        for (uint32_t c = 0; c < mat.nCols; ++c)
        {
            const uint32_t srcOffset = (r * mat.nCols + c) * matElementSize;
            uint32_t dstOffset = 0;

            // Stride 0 is undefined per the SPIR-V specification but we need to
            // support it here because of the way the tests are structured.
            switch (layout)
            {
                case Layout::RowMajor: {
                    if (stride == 0 && r >= 1)
                    {
                        continue;
                    }
                    dstOffset = r * bufStrideSize + c * matElementSize;
                    break;
                }
                case Layout::ColumnMajor: {
                    if (stride == 0 && c >= 1)
                    {
                        continue;
                    }
                    dstOffset = c * bufStrideSize + r * matElementSize;
                    break;
                }
            }
            memcpy(buf.data.data() + dstOffset, mat.data.data() + srcOffset,
                   matElementSize);
        }
    }
}

static uint32_t roundUpToNearestMultiple(uint32_t val, uint32_t multiple)
{
    if (multiple == 0)
    {
        return val;
    }
    const uint32_t remainder = val % multiple;
    return remainder == 0 ? val : val + multiple - remainder;
}

BufferDescriptor BufferDescriptor::makeBufferDescriptor(
    const cl_device_cooperative_matrix_component_type_khr matElementType,
    const uint32_t stride, const uint32_t strideCount,
    const BufferElementType elementType)
{
    const uint32_t matElementSize = elementSizeOf(matElementType);
    const uint32_t matStrideMultiple =
        std::max(matElementSize, gTestContext->deviceStrideMultiple);

    const uint32_t dataStrideSize =
        bufferElementTypeSizeOf(elementType) * stride;
    const uint32_t strideSize =
        roundUpToNearestMultiple(dataStrideSize, matStrideMultiple);
    const uint32_t stridePadding = strideSize - dataStrideSize;

    return BufferDescriptor(stride, strideCount, elementType, stridePadding);
}

uint32_t bufferElementTypeSizeOf(const BufferElementType &t)
{
    // vec3 must be vec4-aligned according to OpenCL C Specification s6.3.5.
    const uint32_t scalarSize = elementSizeOf(t.scalarType);
    const uint32_t vecLen = t.vectorLength;
    return vecLen == 3 ? scalarSize * 4 : scalarSize * vecLen;
}

uint32_t bufferStrideSizeOf(const BufferDescriptor &d)
{
    return bufferElementTypeSizeOf(d.elementType) * d.stride + d.stridePadding;
}

uint32_t bufferSizeOf(const BufferDescriptor &d)
{
    return bufferStrideSizeOf(d) * d.strideCount;
}

uint32_t bufferElementStride(const BufferDescriptor &d)
{
    const uint32_t dataStrideSize = bufferStrideSizeOf(d) - d.stridePadding;
    const uint32_t elementSize = bufferElementTypeSizeOf(d.elementType);
    assert(dataStrideSize % elementSize == 0);
    return dataStrideSize / elementSize;
}

namespace {

// clang-format off
// |------BUFFER ---------------------------------------------------------------------------|
//                     |-----SUB-BUFFER ----------------------------------------------------|
//
// +-------------------+----------+---------+----------+---------+-----+----------+---------+
// | Alignment padding | Stride 0 | padding | Stride 1 | padding | ... | Stride n | padding |
// +-------------------+----------+---------+----------+---------+-----+----------+---------+
// clang-format on
// SUB-BUFFER is aligned to the maximum of the matrix element size and
// CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR.
//
// The stride padding is chosen such that each stride (including padding) is a
// multiple of the larger of the matrix element size and
// CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR.
struct ClBuffer
{
    clMemWrapper bufferHandle;
    clMemWrapper subBufferHandle;

    const BufferDescriptor descriptor;

    ClBuffer(clMemWrapper bufferHandle, clMemWrapper subBufferHandle,
             const BufferDescriptor descriptor)
        : bufferHandle(bufferHandle), subBufferHandle(subBufferHandle),
          descriptor(std::move(descriptor))
    {}
};

static std::optional<ClBuffer>
makeClBuffer(cl_context context, cl_command_queue queue, cl_mem_flags flags,
             const BufferDescriptor desc, const Matrix &mat)
{
    const uint32_t matElementSize = elementSizeOf(mat.elementType);
    const uint32_t matBufferSize = bufferSizeOf(desc);
    const uint32_t bufAlignment =
        std::max(matElementSize, gTestContext->devicePointerAlignment);
    // We overallocate the buffer such that we later can pick an alignment
    // padding in the range [0, bufAlignment).
    const uint32_t bufSize = matBufferSize + bufAlignment;
    cl_int err = CL_SUCCESS;
    clMemWrapper bufHandle = clCreateBuffer(
        context, flags | CL_MEM_ALLOC_HOST_PTR, bufSize, nullptr, &err);
    test_error_ret(err, "Unable to create buffer.\n", std::nullopt);

    // We need to map the buffer to learn the actual alignment.
    void *bufPointer =
        clEnqueueMapBuffer(queue, bufHandle, CL_TRUE, CL_MAP_READ, 0, bufSize,
                           0, nullptr, nullptr, &err);
    test_error_ret(err, "Unable to map buffer.\n", std::nullopt);
    const uintptr_t bufAddress = reinterpret_cast<uintptr_t>(bufPointer);
    const uint32_t alignmentRemainder = bufAddress % bufAlignment;
    const uint32_t alignmentPadding =
        alignmentRemainder == 0 ? 0 : bufAlignment - alignmentRemainder;
    // Unmap the buffer pointer since we do not need it anymore.
    err = clEnqueueUnmapMemObject(queue, bufHandle, bufPointer, 0, nullptr,
                                  nullptr);
    test_error_ret(err, "Unable to unmap parent buffer.\n", std::nullopt);

    const cl_buffer_region bufRegion = { alignmentPadding, matBufferSize };
    clMemWrapper subBufHandle = clCreateSubBuffer(
        bufHandle, flags, CL_BUFFER_CREATE_TYPE_REGION, &bufRegion, &err);
    test_error_ret(err, "Unable to create sub-buffer.\n", std::nullopt);
    return ClBuffer(bufHandle, subBufHandle, desc);
}

const char *getTypeStr(cl_device_cooperative_matrix_component_type_khr t)
{
    switch (t)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR: return "f16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR: return "f32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: return "f64";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR: return "s8";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
            return "s16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
            return "s32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            return "s64";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR: return "u8";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            return "u16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            return "u32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return "u64";
        default: return "???";
    }
}

std::string
describeBufferKind(cl_device_cooperative_matrix_component_type_khr kind)
{
    switch (kind)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR: return "f16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR: return "f32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: return "f64";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR: return "i8";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            return "i16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            return "i32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return "i64";
        default: return "???";
    }
}

static std::string describeBuffer(const BufferDescriptor &d)
{
    std::string res;
    if (d.elementType.vectorLength > 1)
    {
        res += "vec" + std::to_string(d.elementType.vectorLength) + "_";
    }
    res += describeBufferKind(d.elementType.scalarType);
    res += "_stride" + std::to_string(d.stride);
    return res;
}

#ifndef NDEBUG
static std::set<uint32_t> makePossibleStrideSizes()
{
    std::set<uint32_t> sizes;
    for (const uint32_t scalarSize : { 1, 2, 4, 8 })
    {
        // vec3 is stored as a vec4.
        for (const uint32_t vectorLength : { 1, 2, 4, 8, 16 })
        {
            sizes.insert(scalarSize * vectorLength);
        }
    }
    return sizes;
}
#endif // NDEBUG

// Produce a relevant subset of the possible buffer element types.
std::set<BufferElementType> bufferElementTypeCombinations(bool fp64_support)
{
    // It is impractical to test all the combinations so we pick a few
    // interesting combinations that tests all the possible stride sizes.
    std::set<BufferElementType> bufferElementTypes = {
        IndexedBufferElementType<1>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR),
        IndexedBufferElementType<2>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR),
        IndexedBufferElementType<4>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR),
        IndexedBufferElementType<16>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR),

        IndexedBufferElementType<1>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR),
        IndexedBufferElementType<3>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR),
        IndexedBufferElementType<8>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR),

        IndexedBufferElementType<1>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR),
        IndexedBufferElementType<3>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR),
        IndexedBufferElementType<8>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR),

        IndexedBufferElementType<1>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR),
        IndexedBufferElementType<2>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR),
        IndexedBufferElementType<4>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR),
        IndexedBufferElementType<16>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR),

        IndexedBufferElementType<1>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR),
        IndexedBufferElementType<3>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR),
        IndexedBufferElementType<8>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR),

        IndexedBufferElementType<1>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR),
        IndexedBufferElementType<2>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR),
        IndexedBufferElementType<4>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR),
        IndexedBufferElementType<8>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR),
        IndexedBufferElementType<16>(
            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR),
    };

    // Add 64 bit floats if supported.
    if (fp64_support)
    {
        bufferElementTypes.insert(
            { IndexedBufferElementType<1>(
                  CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR),
              IndexedBufferElementType<2>(
                  CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR),
              IndexedBufferElementType<4>(
                  CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR),
              IndexedBufferElementType<16>(
                  CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR) });
    }

#ifndef NDEBUG
    // Make sure all stride sizes are tested.
    const auto containsAllStrideSizes = [&bufferElementTypes]() -> bool {
        std::set<uint32_t> strideSizes = makePossibleStrideSizes();
        for (const BufferElementType t : bufferElementTypes)
        {
            strideSizes.erase(bufferElementTypeSizeOf(t));
        }

        return strideSizes.empty();
    };
    assert(containsAllStrideSizes() && "Some stride sizes are untested!");
#endif // NDEBUG

    return bufferElementTypes;
}

} // anonymous namespace

const char *showScalarType(cl_device_cooperative_matrix_component_type_khr t)
{
    switch (t)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR: return "f16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR: return "f32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: return "f64";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR: return "i8";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
            return "i16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
            return "i32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            return "i64";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR: return "i8";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            return "i16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            return "i32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return "i64";
        default: return "???";
    }
}

std::string
describeBufferKind(cl_device_cooperative_matrix_component_type_khr kind)
{
    switch (kind)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR: return "f16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR: return "f32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: return "f64";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR: return "i8";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            return "i16";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            return "i32";
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return "i64";
        default: return "???";
    }
}

bool MatrixType::operator<(const MatrixType &other) const
{
    return type < other.type || (type == other.type && nRows < other.nRows)
        || (type == other.type && nRows == other.nRows && nCols < other.nCols)
        || (type == other.type && nRows == other.nRows && nCols == other.nCols
            && use < other.use);
}

bool MatrixType::operator==(const MatrixType &other) const
{
    return type == other.type && nRows == other.nRows && nCols == other.nCols
        && use == other.use;
}

bool MatrixType::isConvertibleTo(const MatrixType &other) const
{
    return nRows == other.nRows && nCols == other.nCols && use == other.use;
}

unsigned Matrix::getIndex(unsigned row, unsigned col) const
{
    assert(row < nRows);
    assert(col < nCols);

    return (row * nCols + col) * elementSizeOf(elementType);
}


std::string Matrix::showElementAtIndex(size_t index) const
{
    switch (elementType)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: {
            std::stringstream s;
            s << std::fixed << std::setprecision(2) << get<double>(index);
            return s.str();
        }
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            return std::to_string(get<int64_t>(index));
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return std::to_string(get<uint64_t>(index));
        default: assert(false && "Encountered unknown type."); std::abort();
    }
}

void Matrix::print(const char *title) const
{
    std::stringstream out;

    out << title << "  Rows(" << unsigned(nRows) << ") x Cols("
        << unsigned(nCols) << ")"
        << "  " << getTypeStr(elementType) << " element type" << '\n';

    for (unsigned row = 0; row < nRows; row++)
    {
        for (unsigned col = 0; col < nCols; col++)
        {
            const unsigned i = getIndex(row, col);
            const std::string element = showElementAtIndex(i);
            out << (col == 0 ? '[' : ' ') << std::setw(10) << element;
        }
        out << "]\n";
    }

    log_info("%s\n", out.str().c_str());
}

template <class Ty>
Ty getHelper(size_t i,
             cl_device_cooperative_matrix_component_type_khr elementType,
             std::vector<uint8_t> data)
{
    assert((i % elementSizeOf(elementType)) == 0
           && "Unaligned access detected");
    const void *ptr = data.data() + i;
    switch (elementType)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
            return *static_cast<const HalfFP *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
            return *static_cast<const float *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR:
            return *static_cast<const double *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
            return *static_cast<const int8_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
            return *static_cast<const int16_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
            return *static_cast<const int32_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            return *static_cast<const int64_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
            return *static_cast<const uint8_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            return *static_cast<const uint16_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            return *static_cast<const uint32_t *>(ptr);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return *static_cast<const uint64_t *>(ptr);
    }
    return 0;
}

double Matrix::getF64(size_t i) const
{
    return getHelper<double>(i, elementType, data);
}

uint64_t Matrix::getU64(size_t i) const
{
    return getHelper<uint64_t>(i, elementType, data);
}

int64_t Matrix::getS64(size_t i) const
{
    return getHelper<int64_t>(i, elementType, data);
}

template <class Ty>
void setHelper(size_t i, Ty value,
               cl_device_cooperative_matrix_component_type_khr elementType,
               std::vector<uint8_t> &data)
{
    assert((i % elementSizeOf(elementType)) == 0
           && "Unaligned access detected");
    void *ptr = data.data() + i;
    switch (elementType)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
            *static_cast<HalfFP *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
            *static_cast<float *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR:
            *static_cast<double *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
            *static_cast<int8_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
            *static_cast<int16_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
            *static_cast<int32_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            *static_cast<int64_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
            *static_cast<uint8_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            *static_cast<uint16_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            *static_cast<uint32_t *>(ptr) = value;
            break;
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            *static_cast<uint64_t *>(ptr) = value;
            break;
    }
}

void Matrix::setF64(size_t i, double value)
{
    return setHelper<double>(i, value, elementType, data);
}

void Matrix::setU64(size_t i, uint64_t value)
{
    return setHelper<uint64_t>(i, value, elementType, data);
}

void Matrix::setS64(size_t i, int64_t value)
{
    return setHelper<int64_t>(i, value, elementType, data);
}

FloatBounds getFBounds(cl_device_cooperative_matrix_component_type_khr type)
{
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR)
    {
        double max = CL_HALF_MAX;
        return { -max, max, true };
    }
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR)
    {
        double max = CL_FLT_MAX;
        return { -max, max, true };
    }
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR)
    {
        double max = CL_DBL_MAX;
        return { -max, max, true };
    }
    assert(false && "Non-float type passed to getFMax");
    std::abort();
}

SignedBounds getSBounds(cl_device_cooperative_matrix_component_type_khr type)
{
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR)
    {
        int64_t max = CL_SCHAR_MAX;
        int64_t min = CL_SCHAR_MIN;
        return { min, max };
    }
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR)
    {
        int64_t max = CL_SHRT_MAX;
        int64_t min = CL_SHRT_MIN;
        return { min, max };
    }
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR)
    {
        int64_t max = CL_INT_MAX;
        int64_t min = CL_INT_MIN;
        return { min, max };
    }
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR)
    {
        int64_t max = CL_LONG_MAX;
        int64_t min = CL_LONG_MIN;
        return { min, max };
    }
    assert(false && "Non-signed-integer type passed to getSBounds");
    std::abort();
}

UnsignedBounds getUBounds(cl_device_cooperative_matrix_component_type_khr type)
{
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR)
        return { 0, CL_UCHAR_MAX };
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR)
        return { 0, CL_USHRT_MAX };
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR)
        return { 0, CL_UINT_MAX };
    if (type == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR)
        return { 0, CL_ULONG_MAX };
    assert(false && "Non-unsigned-integer type passed to getUBounds");
    std::abort();
}

Bounds getBounds(cl_device_cooperative_matrix_component_type_khr type)
{
    switch (type)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: {
            return getFBounds(type);
        }
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            return getSBounds(type);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR: {
            return getUBounds(type);
        }
    }
    assert(false && "Unknown component type passed to getBounds");
    std::abort();
}

template <typename T, typename... Tys>
inline constexpr bool is_one_of_v = std::disjunction_v<std::is_same<T, Tys>...>;

template <typename T>
inline constexpr bool is_i64_or_u64_v = is_one_of_v<T, int64_t, uint64_t>;

template <typename T>
inline constexpr bool is_bounds_type_v =
    is_one_of_v<T, FloatBounds, UnsignedBounds, SignedBounds>;

/// When the input integer is not representable as a double, perform the
/// conversion rounding up (towards positive infinity).
template <typename Int, std::enable_if_t<is_i64_or_u64_v<Int>, int> = 0>
double castIntToDoubleRoundUp(Int i)
{
    double d = static_cast<double>(i);
    // If the default casting rounded down, we increment it to the next
    // double value.
    if (static_cast<Int>(d) < i)
    {
        d = std::nextafter(d, std::numeric_limits<double>::infinity());
    }
    return d;
}

/// When the input integer is not representable as a double, perform the
/// conversion rounding down (towards negative infinity).
template <typename Int, std::enable_if_t<is_i64_or_u64_v<Int>, int> = 0>
double castIntToDoubleRoundDown(Int i)
{
    double d = static_cast<double>(i);
    // If the default casting rounded up, we decrement it to the next
    // double value.
    if (static_cast<Int>(d) > i)
    {
        d = std::nextafter(d, -std::numeric_limits<double>::infinity());
    }
    return d;
}

template <typename Int, std::enable_if_t<is_i64_or_u64_v<Int>, int> = 0>
bool isFloatGreaterThanInt(double f, const Int i)
{
    if (std::isnan(f)) return false;
    double iAsDouble = castIntToDoubleRoundUp(i);
    return (f >= iAsDouble && static_cast<Int>(static_cast<double>(i)) != i)
        || f > iAsDouble;
}

template <typename Int, std::enable_if_t<is_i64_or_u64_v<Int>, int> = 0>
bool isFloatLessThanInt(double f, const Int i)
{
    if (std::isnan(f)) return false;
    double iAsDouble = castIntToDoubleRoundDown(i);
    return (f <= iAsDouble && static_cast<Int>(static_cast<double>(i)) != i)
        || f < iAsDouble;
}


template <typename Int, std::enable_if_t<is_i64_or_u64_v<Int>, int> = 0>
Int castDoubleToIntRoundDown(double d)
{
    auto min = std::numeric_limits<Int>::min();
    auto max = std::numeric_limits<Int>::max();
    if (isFloatGreaterThanInt(d, max)) return max;
    if (isFloatLessThanInt(d, min)) return min;
    return static_cast<Int>(std::floor(d));
}

template <typename Int, std::enable_if_t<is_i64_or_u64_v<Int>, int> = 0>
Int castDoubleToIntRoundUp(double d)
{
    auto min = std::numeric_limits<Int>::min();
    auto max = std::numeric_limits<Int>::max();
    if (isFloatGreaterThanInt(d, max)) return max;
    if (isFloatLessThanInt(d, min)) return min;
    return static_cast<Int>(std::ceil(d));
}

int64_t castUIntToInt(uint64_t i)
{
    uint64_t max = std::numeric_limits<int64_t>::max();
    return std::min(max, i);
}

uint64_t castIntToUInt(int64_t i) { return std::max(int64_t{ 0 }, i); }

/// maps a U-bound to greatest T-Bound that is contained within it. This ensures
/// that a number generated from the new bound is still inside the old bound.
template <typename T, std::enable_if_t<is_bounds_type_v<T>, int> = 0,
          typename U, std::enable_if_t<is_bounds_type_v<U>, int> = 0>
T mapBounds(U bounds)
{
    if constexpr (std::is_same_v<T, FloatBounds>)
    {
        if constexpr (std::is_same_v<U, FloatBounds>)
        {
            return bounds;
        }
        else if constexpr (std::is_same_v<U, SignedBounds>)
        {
            double min = castIntToDoubleRoundUp(bounds.min);
            double max = castIntToDoubleRoundDown(bounds.max);
            return { min, max, false };
        }
        else if constexpr (std::is_same_v<U, UnsignedBounds>)
        {
            double min = castIntToDoubleRoundUp(bounds.min);
            double max = castIntToDoubleRoundDown(bounds.max);
            return { min, max, false };
        }
    }
    else if constexpr (std::is_same_v<T, SignedBounds>)
    {
        if constexpr (std::is_same_v<U, FloatBounds>)
        {
            int64_t min = castDoubleToIntRoundUp<int64_t>(bounds.min);
            int64_t max = castDoubleToIntRoundDown<int64_t>(bounds.max);
            return { min, max };
        }
        else if constexpr (std::is_same_v<U, SignedBounds>)
        {
            return bounds;
        }
        else if constexpr (std::is_same_v<U, UnsignedBounds>)
        {
            int64_t min = castUIntToInt(bounds.min);
            int64_t max = castUIntToInt(bounds.max);
            return { min, max };
        }
    }
    else if constexpr (std::is_same_v<T, UnsignedBounds>)
    {
        if constexpr (std::is_same_v<U, FloatBounds>)
        {
            uint64_t min = castDoubleToIntRoundUp<uint64_t>(bounds.min);
            uint64_t max = castDoubleToIntRoundDown<uint64_t>(bounds.max);
            return { min, max };
        }
        else if constexpr (std::is_same_v<U, SignedBounds>)
        {
            uint64_t min = castIntToUInt(bounds.min);
            uint64_t max = castIntToUInt(bounds.max);
            return { min, max };
        }
        else if constexpr (std::is_same_v<U, UnsignedBounds>)
        {
            return bounds;
        }
    }
}

template <typename T, std::enable_if_t<is_bounds_type_v<T>, int> = 0>
T boundsIntersection(T left, T right)
{
    auto min = std::min(left.min, right.min);
    auto max = std::min(left.max, right.max);
    if constexpr (std::is_same_v<T, FloatBounds>)
    {
        bool canBeNonFinite = left.canBeNonFinite & right.canBeNonFinite;
        return { min, max, canBeNonFinite };
    }
    else
        return { min, max };
}

void Matrix::fill(int8_t seed, std::optional<Bounds> bounds)
{
    MTdataHolder rng = MTdataHolder(seed);

    for (cl_uint row = 0; row < nRows; row++)
    {
        for (cl_uint col = 0; col < nCols; col++)
        {
            cl_uint offset = getIndex(row, col);
            switch (elementType)
            {
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: {
                    FloatBounds fBounds = bounds
                        ? (assert(std::holds_alternative<FloatBounds>(*bounds)),
                           std::get<FloatBounds>(*bounds))
                        : getFBounds(elementType);

                    auto ratio = uint64_t{ 2 * nRows * nCols };
                    uint64_t discriminator = genrand_int64(rng) % ratio;
                    double value;
                    if (fBounds.canBeNonFinite && discriminator == 0)
                        value = std::numeric_limits<double>::quiet_NaN();
                    else if (fBounds.canBeNonFinite && discriminator == 1)
                        value = std::numeric_limits<double>::infinity();
                    else if (fBounds.canBeNonFinite && discriminator == 2)
                        value = -std::numeric_limits<double>::infinity();
                    else
                        value =
                            get_random_double(fBounds.min, fBounds.max, rng);
                    setF64(offset, value);
                    break;
                }
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR: {
                    SignedBounds sBounds = bounds
                        ? (assert(
                               std::holds_alternative<SignedBounds>(*bounds)),
                           std::get<SignedBounds>(*bounds))
                        : getSBounds(elementType);
                    int64_t value =
                        static_cast<int64_t>(
                            static_cast<uint64_t>(genrand_int64(rng))
                            % (static_cast<uint64_t>(sBounds.max)
                               - static_cast<uint64_t>(sBounds.min)))
                        + sBounds.min;
                    setS64(offset, value);
                    break;
                }
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
                    UnsignedBounds uBounds = bounds
                        ? (assert(
                               std::holds_alternative<UnsignedBounds>(*bounds)),
                           std::get<UnsignedBounds>(*bounds))
                        : getUBounds(elementType);
                    uint64_t value = genrand_int64(rng) % uBounds.max;
                    setU64(offset, value);
                    break;
            }
        }
    }
}

void Matrix::copyFrom(const Matrix &other)
{
    // Allow different uses/layouts as long as element type and shape match.
    assert(elementType == other.elementType);
    assert(nRows == other.nRows);
    assert(nCols == other.nCols);
    for (cl_uint row = 0; row < nRows; row++)
    {
        for (cl_uint col = 0; col < nCols; col++)
        {
            const size_t dst = getIndex(row, col);
            const size_t src = other.getIndex(row, col);

            switch (elementType)
            {
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
                    set(dst, other.get<HalfFP>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
                    set(dst, other.get<float>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR:
                    set(dst, other.get<double>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
                    set(dst, other.get<int8_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
                    set(dst, other.get<int16_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
                    set(dst, other.get<int32_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
                    set(dst, other.get<int64_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
                    set(dst, other.get<uint8_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
                    set(dst, other.get<uint16_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
                    set(dst, other.get<uint32_t>(src));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
                    set(dst, other.get<uint64_t>(src));
                    break;
            }
        }
    }
}

std::string Variant::describe() const
{
    std::ostringstream OS;

    if (isConversion)
    {
        assert(order == OperandOrder::OpA || order == OperandOrder::OpB
               || order == OperandOrder::OpC);
        switch (order)
        {
            case OperandOrder::OpA:
                OS << inputA.nRows << "x" << inputA.nCols << ".A."
                   << getTypeStr(inputA.elementType);
                break;
            case OperandOrder::OpB:
                OS << inputB.nRows << "x" << inputB.nCols << ".B."
                   << getTypeStr(inputB.elementType);
                break;
            case OperandOrder::OpC:
                OS << inputC.nRows << "x" << inputC.nCols << ".C."
                   << getTypeStr(inputC.elementType);
                break;
            default: break;
        }
        OS << ".to." << output.nRows << "x" << output.nCols << ".C."
           << getTypeStr(output.elementType);
        return OS.str();
    }

    switch (order)
    {
        case OperandOrder::OpA:
        case OperandOrder::OpAA:
            OS << inputA.nRows << "x" << inputA.nCols << ".A."
               << getTypeStr(inputA.elementType);
            break;
        case OperandOrder::OpB:
        case OperandOrder::OpBB:
            OS << inputB.nRows << "x" << inputB.nCols << ".B."
               << getTypeStr(inputB.elementType);
            break;
        case OperandOrder::OpC:
        case OperandOrder::OpCC:
            OS << inputC.nRows << "x" << inputC.nCols << ".C."
               << getTypeStr(inputC.elementType);
            break;
        case OperandOrder::OpABC:
            OS << inputC.nRows << "x" << inputC.nCols << "x" << inputB.nRows
               << "." << getTypeStr(inputA.elementType) << "x"
               << getTypeStr(inputB.elementType) << "_"
               << getTypeStr(inputC.elementType) << "_"
               << getTypeStr(output.elementType)
               << (isSaturating ? ".sat" : ".nosat");
            break;
    }

    if (layoutA != Layout::RowMajor || layoutRes != Layout::RowMajor)
    {
        auto appendLayout = [&OS](Layout layout) {
            switch (layout)
            {
                case Layout::RowMajor: OS << ".rowm"; break;
                case Layout::ColumnMajor: OS << ".colm"; break;
            }
        };
        // Tests assume that all input matrices share the same layout.
        assert(layoutA == layoutB);
        assert(layoutB == layoutC);
        appendLayout(layoutA);
        appendLayout(layoutRes);
    }

    if (isMulticomponent)
    {
        const BufferDescriptor *inDesc = nullptr;
        switch (order)
        {
            case OperandOrder::OpA: inDesc = &inputADesc; break;
            case OperandOrder::OpB: inDesc = &inputBDesc; break;
            case OperandOrder::OpC: inDesc = &inputCDesc; break;
            default:
                assert(false && "isMulticomponent tests are Unary!");
                std::abort();
        }
        OS << ".inBuf_" << describeBuffer(*inDesc);
        OS << ".outBuf_" << describeBuffer(outputDesc);
    }

    return OS.str();
}

const Matrix &Variant::getMatrix(uint8_t matrixID) const
{
    switch (matrixID)
    {
        case 0: return inputA;
        case 1: return inputB;
        case 2: return inputC;
        case 3: return output;
        default: log_error("Invalid matrix index"); std::abort();
    }
}

const BufferDescriptor &Variant::getBufferDescriptor(uint8_t bufferID) const
{
    switch (bufferID)
    {
        case 0: return inputADesc;
        case 1: return inputBDesc;
        case 2: return inputCDesc;
        case 3: return outputDesc;
        default: log_error("Invalid buffer index"); std::abort();
    }
}

void Variant::getInputsForOperation(std::vector<const Matrix *> &inputs) const
{
    switch (order)
    {
        case OperandOrder::OpA: inputs.push_back(&inputA); break;
        case OperandOrder::OpB: inputs.push_back(&inputB); break;
        case OperandOrder::OpC: inputs.push_back(&inputC); break;

        case OperandOrder::OpAA:
        case OperandOrder::OpBB:
        case OperandOrder::OpCC:
            // Always use the A & B input matrices for binary operations.
            inputs.push_back(&inputA);
            inputs.push_back(&inputB);
            break;

        case OperandOrder::OpABC:
            inputs.push_back(&inputA);
            inputs.push_back(&inputB);
            inputs.push_back(&inputC);
    }
}


namespace {

// Base class for a single cooperative_matrix operation test, testing all
// supported matrix variants.  This is an abstract class; a concrete subclass is
// defined for each operation.
class CoopMatTest {
public:
    CoopMatTest(cl_device_id device, cl_context context, cl_command_queue queue,
                const char *opName, CoopMatOp op, CoopMatOpKind kind)
        : device(device), context(context), queue(queue), opName(opName),
          op(op), kind(kind)
    {}

    // Run tests for all variants.
    int runAll();

private:
    // Construct a program for a single variant, then run and verify.
    int buildAndRun(Variant &variant);

    // Verify the output data. Return true on success.
    bool verify(const Variant &variant) const;

    // Fill the reference output matrix.
    void fillRefOutput(Variant &variant);

protected:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    const char *opName;
    CoopMatOp op;
    CoopMatOpKind kind;
    const bool isSpirv = true;

    virtual void calcRef(std::vector<const Matrix *> &inputs, Matrix &m) = 0;
};

// Subclass for each operation.
#define COOPMAT(op, kind_ignore)                                               \
    class CoopMatTest_##op : public CoopMatTest {                              \
    public:                                                                    \
        CoopMatTest_##op(cl_device_id device, cl_context context,              \
                         cl_command_queue queue, const char *opName,           \
                         CoopMatOp op, CoopMatOpKind kind)                     \
            : CoopMatTest(device, context, queue, opName, op, kind)            \
        {}                                                                     \
                                                                               \
        void calcRef(std::vector<const Matrix *> &inputs, Matrix &m) override; \
    };
#include "cooperative_matrix.def"
#undef COOPMAT

void CoopMatTest::fillRefOutput(Variant &variant)
{
    std::vector<const Matrix *> inputs;
    variant.getInputsForOperation(inputs);
    calcRef(inputs, variant.outputRef);
}

template <typename T> double ULPError(T val, T reference)
{
    if constexpr (std::is_same_v<T, HalfFP>)
    {
        return Ulp_Error_Half(val.data, static_cast<float>(reference));
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return Ulp_Error(val, reference);
    }
    else
    {
        return Ulp_Error_Double(val, reference);
    }
}

template <typename T>
bool verifyOutputBuffer(const Variant &v, bool isDiv, double maxULPError)
{
    std::vector<const Matrix *> inputs;
    bool isFP =
        std::is_floating_point<T>::value || std::is_same<T, HalfFP>::value;
    if (!isFP && isDiv)
    {
        v.getInputsForOperation(inputs);
    }

    const int MAX_ERRORS = 20;
    int errorCount = 0;
    for (size_t res_row = 0;
         res_row < v.output.nRows && errorCount < MAX_ERRORS; res_row++)
    {
        for (size_t res_col = 0;
             res_col < v.output.nCols && errorCount < MAX_ERRORS; res_col++)
        {
            const unsigned offset = v.output.getIndex(res_row, res_col);

            assert(offset == v.outputRef.getIndex(res_row, res_col)
                   && "Reference and output buffer Strides do not match!");

            // Discard verification on divisions by 0.
            if (!isFP && isDiv && inputs[1]->get<T>(offset) == T(0)) continue;

            const T expected = v.outputRef.get<T>(offset);
            const T actual = v.output.get<T>(offset);

            // Do not try to compare signaling nans.
            if (isFP && isnan(actual) && isnan(expected)) continue;

            if ((!isFP && actual != expected)
                || (isFP && ULPError(actual, expected) > maxULPError))
            {
                std::ostringstream os;
                os << "Verification failed for element at position (" << res_row
                   << "," << res_col << "): expected " << +expected
                   << ", actual " << +actual;
                log_error("%s\n", os.str().c_str());
                errorCount++;
                if (errorCount == MAX_ERRORS)
                {
                    // Stop reporting errors after seeing many, to avoid
                    // generating unwieldy large error logs.
                    log_error(
                        "More than %d errors found, stopping verification.\n",
                        MAX_ERRORS);
                }
            }
        }
    }

    if (errorCount)
    {
        // Print the full input and output matrices in case of failure.
        switch (v.order)
        {
            case Variant::OperandOrder::OpA: v.inputA.print("Input A"); break;
            case Variant::OperandOrder::OpB: v.inputB.print("Input B"); break;
            case Variant::OperandOrder::OpC: v.inputC.print("Input C"); break;
            case Variant::OperandOrder::OpAA:
            case Variant::OperandOrder::OpBB:
            case Variant::OperandOrder::OpCC:
                v.inputA.print("Input LHS");
                v.inputB.print("Input RHS");
                break;
            case Variant::OperandOrder::OpABC:
                v.inputA.print("Input A");
                v.inputB.print("Input B");
                v.inputC.print("Input C");
                break;
        }
        v.outputRef.print("Reference output");
        v.output.print("Actual output");
    }

    return errorCount == 0;
}

bool CoopMatTest::verify(const Variant &variant) const
{
    if (op == CoopMatOp::length)
    {
        // The distribution of matrix elements across work-items is
        // implementation defined and there is no guarantee of an even
        // distribution. The only thing we can check is that the values from all
        // work-items add up to the matrix size.
        uint32_t sum = 0;
        const uint32_t expected = variant.output.elementCount();
        const size_t outputBytes = bufferSizeOf(variant.outputDesc);
        for (size_t i = 0; i < variant.globalSize; i++)
        {
            // Outputs are packed starting at index 0.
            const size_t byteOffset = i * sizeof(uint32_t);
            if (byteOffset + sizeof(uint32_t) > outputBytes) break;
            sum += variant.output.get<uint32_t>(byteOffset);
        }
        if (sum != expected)
        {
            log_error(
                "Verification failed; expected length to be %u; actual %u\n",
                expected, sum);
            return false;
        }
        return true;
    }

    const bool isDiv = (op == CoopMatOp::div);

    double maxULPError = 0;
    switch (op)
    {
        case CoopMatOp::div:
            // FP64 division must be correctly rounded.
            if (variant.output.elementType
                == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR)
                break;
            // ULP requirements are based on type.
            maxULPError = variant.output.elementType
                    == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR
                ? 1.0
                : 2.5;
            break;
        case CoopMatOp::matrixmuladd:
        case CoopMatOp::matrixmuladd_array:
        case CoopMatOp::matrixmuladd_saturating:
        case CoopMatOp::matrixmuladd_stride0:
        case CoopMatOp::matrixmuladd_wrapping:
            // This number is arbitrary as the SPV_KHR_cooperative_matrix
            // specification does not provide accuracy requirements.
            maxULPError = 1117;
            break;
        default: break;
    }


    switch (variant.output.elementType)
    {
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
            return verifyOutputBuffer<HalfFP>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
            return verifyOutputBuffer<float>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR:
            return verifyOutputBuffer<double>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
            return verifyOutputBuffer<int8_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
            return verifyOutputBuffer<int16_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
            return verifyOutputBuffer<int32_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
            return verifyOutputBuffer<int64_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
            return verifyOutputBuffer<uint8_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
            return verifyOutputBuffer<uint16_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
            return verifyOutputBuffer<uint32_t>(variant, isDiv, maxULPError);
        case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
            return verifyOutputBuffer<uint64_t>(variant, isDiv, maxULPError);
    }
    return false;
}

void CoopMatTest_length::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    // Nothing to do; the result does not depend on any input matrix data.
}

void CoopMatTest_constant::calcRef(std::vector<const Matrix *> &inputs,
                                   Matrix &m)
{
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            m.set(m.getIndex(row, col), 123);
        }
    }
}

static void fillCopy(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 1);
    const Matrix *in = inputs.front();
    for (size_t row = 0; row < in->nRows; row++)
    {
        for (size_t col = 0; col < in->nCols; col++)
        {
            const size_t srcIndex = in->getIndex(row, col);
            const size_t dstIndex = m.getIndex(row, col);
            assert(m.elementType == inputs[0]->elementType);

            switch (m.elementType)
            {
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
                    m.set(dstIndex, in->get<HalfFP>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
                    m.set(dstIndex, in->get<float>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR:
                    m.set(dstIndex, in->get<double>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
                    m.set(dstIndex, in->get<int8_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
                    m.set(dstIndex, in->get<int16_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
                    m.set(dstIndex, in->get<int32_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
                    m.set(dstIndex, in->get<int64_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
                    m.set(dstIndex, in->get<uint8_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
                    m.set(dstIndex, in->get<uint16_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
                    m.set(dstIndex, in->get<uint32_t>(srcIndex));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
                    m.set(dstIndex, in->get<uint64_t>(srcIndex));
                    break;
            }
        }
    }
}

void CoopMatTest_copy::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    fillCopy(inputs, m);
}

void CoopMatTest_copy_stride0::calcRef(std::vector<const Matrix *> &inputs,
                                       Matrix &m)
{
    fillCopy(inputs, m);
}

void CoopMatTest_copy_workgroup::calcRef(std::vector<const Matrix *> &inputs,
                                         Matrix &m)
{
    fillCopy(inputs, m);
}

void CoopMatTest_composite::calcRef(std::vector<const Matrix *> &inputs,
                                    Matrix &m)
{
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            m.set(m.getIndex(row, col), 123);
        }
    }
}

void CoopMatTest_composite_array::calcRef(std::vector<const Matrix *> &inputs,
                                          Matrix &m)
{
    fillCopy(inputs, m);
}

void CoopMatTest_composite_rvalue::calcRef(std::vector<const Matrix *> &inputs,
                                           Matrix &m)
{
    fillCopy(inputs, m);
}

void CoopMatTest_convert::calcRef(std::vector<const Matrix *> &inputs,
                                  Matrix &m)
{
    assert(inputs.size() == 1);
    const Matrix *input_matrix = inputs.front();
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            const unsigned from = input_matrix->getIndex(row, col);
            const unsigned to = m.getIndex(row, col);
            switch (input_matrix->elementType)
            {
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR: {
                    m.set(to, input_matrix->get<double>(from));
                    break;
                }
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
                    m.set(to, input_matrix->get<int64_t>(from));
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
                    m.set(to, input_matrix->get<uint64_t>(from));
                    break;
                default:
                    assert(false && "Encountered unknown type.");
                    std::abort();
            }
        }
    }
}

void CoopMatTest_negate::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 1);
    assert(m.elementType == inputs[0]->elementType);
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            unsigned idx = m.getIndex(row, col);
            if (isFloatType(m.elementType))
                m.set(idx, -inputs[0]->get<double>(idx));
            else if (isSignedType(m.elementType))
                m.set(idx, -inputs[0]->get<int64_t>(idx));
            else
                m.set(idx, -inputs[0]->get<uint64_t>(idx));
        }
    }
}

void CoopMatTest_add::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 2);
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            unsigned idx = m.getIndex(row, col);
            if (isFloatType(m.elementType))
                m.set(idx,
                      inputs[0]->get<double>(idx)
                          + inputs[1]->get<double>(idx));
            else if (isSignedType(m.elementType))
                m.set(idx,
                      inputs[0]->get<int64_t>(idx)
                          + inputs[1]->get<int64_t>(idx));
            else
                m.set(idx,
                      inputs[0]->get<uint64_t>(idx)
                          + inputs[1]->get<uint64_t>(idx));
        }
    }
}

void CoopMatTest_sub::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 2);
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            unsigned idx = m.getIndex(row, col);
            if (isFloatType(m.elementType))
                m.set(idx,
                      inputs[0]->get<double>(idx)
                          - inputs[1]->get<double>(idx));
            else if (isSignedType(m.elementType))
                m.set(idx,
                      inputs[0]->get<int64_t>(idx)
                          - inputs[1]->get<int64_t>(idx));
            else
                m.set(idx,
                      inputs[0]->get<uint64_t>(idx)
                          - inputs[1]->get<uint64_t>(idx));
        }
    }
}

void CoopMatTest_mul::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 2);
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            unsigned idx = m.getIndex(row, col);
            if (isFloatType(m.elementType))
                m.set(idx,
                      inputs[0]->get<double>(idx)
                          * inputs[1]->get<double>(idx));
            else if (isSignedType(m.elementType))
                m.set(idx,
                      inputs[0]->get<int64_t>(idx)
                          * inputs[1]->get<int64_t>(idx));
            else
                m.set(idx,
                      inputs[0]->get<uint64_t>(idx)
                          * inputs[1]->get<uint64_t>(idx));
        }
    }
}

template <typename T> T safeDivision(T num, T div)
{
    if (div == 0) return 0;
    return num / div;
}

void CoopMatTest_div::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 2);
    for (size_t row = 0; row < m.nRows; row++)
    {
        for (size_t col = 0; col < m.nCols; col++)
        {
            unsigned idx = m.getIndex(row, col);
            if (isFloatType(m.elementType))
                m.set(idx,
                      inputs[0]->get<double>(idx)
                          / inputs[1]->get<double>(idx));
            else if (isSignedType(m.elementType))
                m.set(idx,
                      safeDivision(inputs[0]->get<int64_t>(idx),
                                   inputs[1]->get<int64_t>(idx)));
            else
                m.set(idx,
                      safeDivision(inputs[0]->get<uint64_t>(idx),
                                   inputs[1]->get<uint64_t>(idx)));
        }
    }
}

void matrixmuladd(std::vector<const Matrix *> &inputs, Matrix &m)
{
    assert(inputs.size() == 3);
    const unsigned M = m.nRows;
    const unsigned N = m.nCols;
    const unsigned K = inputs[0]->nCols;
    for (unsigned row = 0; row < M; row++)
    {
        for (unsigned col = 0; col < N; col++)
        {
            const unsigned outIdx = m.getIndex(row, col);
            const unsigned CIndex = inputs[2]->getIndex(row, col);
            union {
                double d;
                int64_t s;
                uint64_t u;
            } acc;
            if (isFloatType(m.elementType))
                acc.d = inputs[2]->get<double>(CIndex);
            else if (isSignedType(m.elementType))
                acc.s = inputs[2]->get<int64_t>(CIndex);
            else
                acc.u = inputs[2]->get<uint64_t>(CIndex);

            for (unsigned i = 0; i < K; i++)
            {
                const unsigned AIndex = inputs[0]->getIndex(row, i);
                const unsigned BIndex = inputs[1]->getIndex(i, col);

                if (isFloatType(m.elementType))
                    acc.d += inputs[0]->get<double>(AIndex)
                        * inputs[1]->get<double>(BIndex);
                else if (isSignedType(m.elementType))
                    acc.s += inputs[0]->get<int64_t>(AIndex)
                        * inputs[1]->get<int64_t>(BIndex);
                else
                    acc.u += inputs[0]->get<uint64_t>(AIndex)
                        * inputs[1]->get<uint64_t>(BIndex);
            }

            if (isFloatType(m.elementType))
                m.set(outIdx, acc.d);
            else if (isSignedType(m.elementType))
                m.set(outIdx, acc.s);
            else
                m.set(outIdx, acc.u);
        }
    }
}

uint64_t saturating_add(uint64_t a, uint64_t b, size_t widthInBytes,
                        bool isSigned)
{
    assert(widthInBytes <= 4 && "maximum byte width is 4");
    if (isSigned)
    {
        const int64_t signedMax =
            ((uint64_t{ 1 } << (widthInBytes * 8 - 1)) - 1);
        const int64_t signedMin = -(uint64_t{ 1 } << (widthInBytes * 8 - 1));
        const int64_t signedA = a;
        const int64_t signedB = b;

        if (signedA > 0)
        {
            if (signedB > signedMax - signedA) return signedMax;
        }
        else
        {
            if (signedB < signedMin - signedA) return signedMin;
        }
        return signedA + signedB;
    }
    const uint64_t maxVal = (uint64_t{ 1 } << (widthInBytes * 8)) - 1;

    if (b > maxVal - a) return maxVal;
    return a + b;
}

void matrixmuladd_saturating(std::vector<const Matrix *> &inputs, Matrix &m)
{
    if (isFloatType(m.elementType))
    {
        matrixmuladd(inputs, m);
        return;
    }
    const uint8_t resultWidthInBytes = elementSizeOf(m.elementType);
    const uint8_t resultSign = isSignedType(m.elementType);
    assert(inputs.size() == 3);
    const unsigned M = m.nRows;
    const unsigned N = m.nCols;
    const unsigned K = inputs[0]->nCols;
    const uint64_t RESULT_MASK =
        ~uint64_t{ 0 } >> (64 - resultWidthInBytes * 8);
    for (unsigned row = 0; row < M; row++)
    {
        for (unsigned col = 0; col < N; col++)
        {
            const unsigned outIdx = m.getIndex(row, col);
            const unsigned CIndex = inputs[2]->getIndex(row, col);
            uint64_t acc = 0;
            for (unsigned i = 0; i < K; i++)
            {
                const unsigned AIndex = inputs[0]->getIndex(row, i);
                const unsigned BIndex = inputs[1]->getIndex(i, col);
                // Signed inputs must be sign extended.
                uint64_t a = isSignedType(inputs[0]->elementType)
                    ? inputs[0]->get<int64_t>(AIndex)
                    : inputs[0]->get<uint64_t>(AIndex);
                uint64_t b = isSignedType(inputs[1]->elementType)
                    ? inputs[1]->get<int64_t>(BIndex)
                    : inputs[1]->get<uint64_t>(BIndex);

                const uint64_t multiplication = (a * b) & RESULT_MASK;
                acc = saturating_add(multiplication, acc, resultWidthInBytes,
                                     resultSign);
            }
            uint64_t c = isSignedType(inputs[2]->elementType)
                ? inputs[2]->get<int64_t>(CIndex)
                : inputs[2]->get<uint64_t>(CIndex);
            m.set(outIdx,
                  saturating_add(acc, c, resultWidthInBytes, resultSign));
        }
    }
}

void CoopMatTest_matrixmuladd::calcRef(std::vector<const Matrix *> &inputs,
                                       Matrix &m)
{
    matrixmuladd(inputs, m);
}

void CoopMatTest_matrixmuladd_array::calcRef(
    std::vector<const Matrix *> &inputs, Matrix &m)
{
    matrixmuladd(inputs, m);
}

void CoopMatTest_matrixmuladd_stride0::calcRef(
    std::vector<const Matrix *> &inputs, Matrix &m)
{
    matrixmuladd(inputs, m);
}

void CoopMatTest_matrixmuladd_wrapping::calcRef(
    std::vector<const Matrix *> &inputs, Matrix &m)
{
    matrixmuladd(inputs, m);
}

void CoopMatTest_matrixmuladd_saturating::calcRef(
    std::vector<const Matrix *> &inputs, Matrix &m)
{
    matrixmuladd_saturating(inputs, m);
}

void CoopMatTest_multicomponent_load::calcRef(
    std::vector<const Matrix *> &inputs, Matrix &m)
{
    // Same as copy
    fillCopy(inputs, m);
}

void CoopMatTest_multicomponent_store::calcRef(
    std::vector<const Matrix *> &inputs, Matrix &m)
{
    // Same as copy
    fillCopy(inputs, m);
}

void CoopMatTest_func::calcRef(std::vector<const Matrix *> &inputs, Matrix &m)
{
    fillCopy(inputs, m);
}

int CoopMatTest::buildAndRun(Variant &variant)
{
    // Generate a SPIR-V program for this variant.
    Program prog;
    ProgramGenerator progGen(variant, op);
    if (!progGen.generateSpirv(&prog))
    {
        log_info("SPIR-V assembling failed.\n");
        return TEST_FAIL;
    }

    // Create a program object from SPIR-V.
    cl_int err = 0;
    clProgramWrapper clprog =
        clCreateProgramWithIL(context, prog.spirvBinary.data(),
                              prog.spirvBinary.size() * sizeof(uint32_t), &err);
    test_error_fail(err, "Failed to create program with clCreateProgramWithIL");

    // Build the program.
    clKernelWrapper kernel;
    err = build_program_create_kernel_helper(context, &clprog, &kernel, 0,
                                             nullptr, "testCoopMat", "");
    test_error_fail(err, "Failed to build program and create kernel");

    // Return success if user requested link check only.
    if (gTestContext->linkCheckOnly)
    {
        return TEST_PASS;
    }

    // Set up input buffers.
    // Regardless of the vectorSize of the operation being tested, always set up
    // 3 input arguments to simplify generation of the program.
    switch (op)
    {
        case CoopMatOp::length:
        case CoopMatOp::constant:
        case CoopMatOp::copy:
        case CoopMatOp::copy_stride0:
        case CoopMatOp::copy_workgroup:
        case CoopMatOp::composite:
        case CoopMatOp::composite_array:
        case CoopMatOp::composite_rvalue:
        case CoopMatOp::negate:
        case CoopMatOp::add:
        case CoopMatOp::sub:
        case CoopMatOp::div:
        case CoopMatOp::matrixmuladd_saturating:
        case CoopMatOp::matrixmuladd_wrapping:
        case CoopMatOp::matrixmuladd_stride0:
        case CoopMatOp::multicomponent_load:
        case CoopMatOp::multicomponent_store:
        case CoopMatOp::func:
            variant.inputA.fill(0);
            variant.inputB.fill(1);
            variant.inputC.fill(2);
            break;
        case CoopMatOp::mul: {
            MatrixType outType(variant.output.elementType, variant.output.nRows,
                               variant.output.nCols, MatrixType::Use::Acc);
            Bounds bounds = getBounds(outType.type);
            Bounds newBounds = std::visit(
                [&](auto x) -> Bounds {
                    auto min = x.min;
                    auto max = x.max;
                    auto maxMag = min >= 0 ? max : std::min(-min, max);
                    // Allows some but not all to overflow.
                    maxMag = std::sqrt(maxMag) / 4 * 5;
                    x.max = maxMag;
                    x.min = -maxMag;
                    return x;
                },
                bounds);
            variant.inputA.fill(0, newBounds);
            variant.inputB.fill(1, newBounds);
            variant.inputC.fill(2, newBounds);
            break;
        }
        case CoopMatOp::matrixmuladd:
        case CoopMatOp::matrixmuladd_array: {
            MatrixType outType(variant.output.elementType, variant.output.nRows,
                               variant.output.nCols, MatrixType::Use::Acc);
            Bounds bounds = getBounds(outType.type);
            Bounds newBounds = std::visit(
                [&](auto x) -> Bounds {
                    auto min = x.min;
                    auto max = x.max;
                    auto maxMag = min >= 0 ? max : std::min(-min, max);
                    // Stops overflows in accumulation.
                    maxMag = std::sqrt(maxMag) / variant.inputA.nCols / 2;
                    x.max = maxMag;
                    x.min = -maxMag;
                    return x;
                },
                bounds);
            variant.inputA.fill(0, newBounds);
            variant.inputB.fill(1, newBounds);
            variant.inputC.fill(2, newBounds);
            break;
        }
        case CoopMatOp::convert: {
            // According to the SPIR-V specification for OpConvertFToU;
            // "Behavior is undefined if Result Type is not wide enough to hold
            // the converted value" (Section 3.3.11). Hence, we need avoid
            // filling the input matrices with any infinity, NaN, out-of-range
            // values.
            Bounds inBounds = getBounds(variant.inputA.elementType);
            Bounds outBounds = getBounds(variant.output.elementType);
            Bounds bounds = std::visit(
                [&](auto in) -> Bounds {
                    return Bounds{ std::visit(
                        [&](auto out) -> decltype(in) {
                            auto mappedBounds = mapBounds<decltype(in)>(out);
                            return boundsIntersection(mappedBounds, in);
                        },
                        outBounds) };
                },
                inBounds);
            variant.inputA.fill(0, bounds);
            variant.inputB.fill(0, bounds);
            variant.inputC.fill(0, bounds);
            break;
        }
    }

    const auto prepareInput = [&](Matrix &mat, const BufferDescriptor &desc,
                                  Layout layout,
                                  uint32_t stride) -> std::optional<ClBuffer> {
        std::optional<ClBuffer> maybe_clBuf =
            makeClBuffer(context, queue, CL_MEM_READ_WRITE, desc, mat);
        if (!maybe_clBuf.has_value())
        {
            return std::nullopt;
        }

        ClBuffer clBuf = maybe_clBuf.value();
        SemBuffer buf(desc, std::vector<uint8_t>(bufferSizeOf(desc)));

        Matrix::cooperativeMatrixStore(buf, mat, layout, stride);

        err = clEnqueueWriteBuffer(queue, clBuf.subBufferHandle, CL_TRUE, 0,
                                   buf.data.size(), buf.data.data(), 0, nullptr,
                                   nullptr);
        test_error_ret(err, "Unable to fill input buffer", std::nullopt);

        // Load the matrix from the buffer the get the "correct" input matrix.
        // This is necessary since OpCooperativeMatrixStoreKHR followed by a
        // OpCooperativeMatrixLoadKHR does not have a round-trip property, due
        // to the stride 0 edge case.
        Matrix newMat = Matrix::cooperativeMatrixLoad(
            mat.elementType, mat.nRows, mat.nCols, buf, layout, stride);
        mat.copyFrom(newMat);

        return clBuf;
    };

    std::optional<ClBuffer> maybe_inputA = prepareInput(
        variant.inputA, variant.inputADesc, variant.layoutA, variant.strideA);
    if (!maybe_inputA.has_value())
    {
        return TEST_FAIL;
    }
    const ClBuffer inputA = maybe_inputA.value();

    std::optional<ClBuffer> maybe_inputB = prepareInput(
        variant.inputB, variant.inputBDesc, variant.layoutB, variant.strideB);
    if (!maybe_inputB.has_value())
    {
        return TEST_FAIL;
    }
    const ClBuffer inputB = maybe_inputB.value();

    std::optional<ClBuffer> maybe_inputC = prepareInput(
        variant.inputC, variant.inputCDesc, variant.layoutC, variant.strideC);
    if (!maybe_inputC.has_value())
    {
        return TEST_FAIL;
    }
    const ClBuffer inputC = maybe_inputC.value();

    // Compute the reference output.
    fillRefOutput(variant);

    // Set up output buffer.
    std::optional<ClBuffer> maybe_output = makeClBuffer(
        context, queue, CL_MEM_WRITE_ONLY, variant.outputDesc, variant.output);
    if (!maybe_output.has_value())
    {
        return TEST_FAIL;
    }
    const ClBuffer output = maybe_output.value();

    const size_t outReadSize = bufferSizeOf(variant.outputDesc);
    assert(bufferSizeOf(variant.outputDesc) <= outReadSize
           && "Output buffer size exceeds padded size after stride rounding");

    // Fill output buffer to give indication of if the test has written to it.
    unsigned char pattern = 13;
    clEnqueueFillBuffer(queue, output.subBufferHandle, &pattern,
                        sizeof(pattern), 0, bufferSizeOf(variant.outputDesc), 0,
                        nullptr, nullptr);

    test_error_fail(err, "Unable to create output buffer");

    // Add the kernel arguments.
    std::array<const ClBuffer, 4> buffers = { inputA, inputB, inputC, output };
    uint8_t i = 0;
    for (ClBuffer buf : buffers)
    {
        err = clSetKernelArg(kernel, i, sizeof(buf.subBufferHandle),
                             &buf.subBufferHandle);
        test_error_fail(err, "Unable to set kernel argument");
        i++;
    }

    // Query the local size needed for a single subgroup, so we can
    // use this as the global size for our work.
    const size_t numSubgroups = 1;
    size_t globalSize;
    err |= clGetKernelSubGroupInfo(
        kernel, device, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
        sizeof(numSubgroups), (void *)&numSubgroups, sizeof(globalSize),
        (void *)&globalSize, nullptr);
    test_error_fail(err, "Failed to get required work group size");
    variant.globalSize = globalSize;

    // Enqueue the work. Cooperative matrices need full subgroups so we set the
    // global size to the (local) size of a single subgroup.
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize,
                                 nullptr, 0, nullptr, nullptr);
    test_error_fail(err, "Unable to enqueue kernel");

    // Read back output into a semantic buffer, then unpack into Matrix.
    SemBuffer buf(variant.outputDesc, std::vector<uint8_t>(outReadSize));
    err =
        clEnqueueReadBuffer(queue, output.subBufferHandle, CL_TRUE, 0,
                            outReadSize, buf.data.data(), 0, nullptr, nullptr);
    test_error_fail(err, "Unable to read destination buffer");

    Matrix outputMat = Matrix::cooperativeMatrixLoad(
        variant.output.elementType, variant.output.nRows, variant.output.nCols,
        buf, variant.layoutRes, variant.strideRes);
    variant.output.copyFrom(outputMat);

    if (!verify(variant)) return TEST_FAIL;

    return TEST_PASS;
}

Variant::OperandOrder getOrderFromKindAndUse(CoopMatOpKind kind,
                                             MatrixType::Use use)
{
    switch (kind)
    {
        case CoopMatOpKind::Unary:
        case CoopMatOpKind::Conversion:
            return (use == MatrixType::Use::A
                        ? Variant::OperandOrder::OpA
                        : (use == MatrixType::Use::B
                               ? Variant::OperandOrder::OpB
                               : Variant::OperandOrder::OpC));
        case CoopMatOpKind::Binary:
            return (use == MatrixType::Use::A
                        ? Variant::OperandOrder::OpAA
                        : (use == MatrixType::Use::B
                               ? Variant::OperandOrder::OpBB
                               : Variant::OperandOrder::OpCC));
        case CoopMatOpKind::Ternary: return Variant::OperandOrder::OpABC;
    }
    assert(false);
    std::abort();
}

int CoopMatTest::runAll()
{
    if (isSpirv && !is_extension_available(device, "cl_khr_cooperative_matrix"))
    {
        log_info("The device does not support the "
                 "cl_khr_cooperative_matrix extension.\n");
        return TEST_SKIPPED_ITSELF;
    }

    // Construct list of variants to test.
    std::vector<Variant> variantsToRun;
    switch (kind)
    {
        case CoopMatOpKind::Unary:
        case CoopMatOpKind::Binary:
            // For unary and binary operations, derive the variants to test from
            // the list of unique types, to avoid testing the same types
            // multiple times.
            for (auto &mt : gTestContext->types)
            {
                Variant::OperandOrder order =
                    getOrderFromKindAndUse(kind, mt.use);

                switch (op)
                {
                    case CoopMatOp::length:
                        // Override the output type, as length always outputs
                        // uint32_t.
                        variantsToRun.emplace_back(
                            mt.type,
                            CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR,
                            mt.nRows, mt.nCols, order, Layout::RowMajor,
                            Layout::RowMajor);
                        break;
                    case CoopMatOp::copy:
                    case CoopMatOp::copy_stride0: {
                        // Test various combinations of stride and layout.
                        const std::vector<Layout> layouts{
                            Layout::RowMajor, Layout::ColumnMajor
                        };
                        for (const auto &srcLayout : layouts)
                        {
                            for (const auto &dstLayout : layouts)
                            {
                                variantsToRun.emplace_back(
                                    mt.type, mt.nRows, mt.nCols, order,
                                    srcLayout, dstLayout);
                                if (op == CoopMatOp::copy_stride0)
                                {
                                    variantsToRun.back().strideA = 0;
                                    variantsToRun.back().strideB = 0;
                                    variantsToRun.back().strideC = 0;
                                    // Keep output stride as is.
                                }
                            }
                        }
                        break;
                    }
                    case CoopMatOp::multicomponent_load:
                    case CoopMatOp::multicomponent_store: {
                        const std::set<BufferElementType> bufElementTypes =
                            bufferElementTypeCombinations(
                                gTestContext->supportFP64);
                        const IndexedBufferElementType<1> defaultChoice =
                            IndexedBufferElementType<1>(mt.type);
                        for (const BufferElementType &choice : bufElementTypes)
                        {
                            // Exclude the default configuration from testing
                            // (which is covered by the normal copy test)
                            if (choice.vectorLength
                                    == defaultChoice.vectorLength
                                && choice.scalarType
                                    == defaultChoice.scalarType)
                            {
                                continue;
                            }
                            // Cannot load a float4 into a 2x2 matrix (size of
                            // the element to be loaded must be less than the
                            // length of the row/col we want to load)
                            if (mt.nCols < choice.vectorLength)
                            {
                                continue;
                            }
                            // Element to be loaded must be at least the size of
                            // the matrix row/col we want to load)
                            if (mt.nCols * elementSizeOf(mt.type)
                                > elementSizeOf(choice.scalarType)
                                    * choice.vectorLength)
                            {
                                continue;
                            }

                            const BufferDescriptor inputDesc =
                                BufferDescriptor::makeBufferDescriptor(
                                    mt.type,
                                    elementsPerStride(Layout::RowMajor,
                                                      mt.nRows, mt.nCols),
                                    numberOfStrides(Layout::RowMajor, mt.nRows,
                                                    mt.nCols),
                                    BufferElementType(choice.vectorLength,
                                                      choice.scalarType));
                            const BufferDescriptor outputDesc =
                                BufferDescriptor::makeBufferDescriptor(
                                    mt.type,
                                    elementsPerStride(Layout::RowMajor,
                                                      mt.nRows, mt.nCols),
                                    numberOfStrides(Layout::RowMajor, mt.nRows,
                                                    mt.nCols),
                                    IndexedBufferElementType<1>(mt.type));

                            if (op == CoopMatOp::multicomponent_load)
                            {
                                variantsToRun.emplace_back(
                                    mt.type, mt.nRows, mt.nCols, order,
                                    inputDesc, outputDesc, Layout::RowMajor,
                                    Layout::RowMajor);
                            }
                            else
                            {
                                const BufferDescriptor outDesc =
                                    BufferDescriptor::makeBufferDescriptor(
                                        mt.type,
                                        elementsPerStride(Layout::RowMajor,
                                                          mt.nRows, mt.nCols),
                                        numberOfStrides(Layout::RowMajor,
                                                        mt.nRows, mt.nCols),
                                        BufferElementType(choice.vectorLength,
                                                          choice.scalarType));
                                const BufferDescriptor inDesc =
                                    BufferDescriptor::makeBufferDescriptor(
                                        mt.type,
                                        elementsPerStride(Layout::RowMajor,
                                                          mt.nRows, mt.nCols),
                                        numberOfStrides(Layout::RowMajor,
                                                        mt.nRows, mt.nCols),
                                        IndexedBufferElementType<1>(mt.type));
                                variantsToRun.emplace_back(
                                    mt.type, mt.nRows, mt.nCols, order, inDesc,
                                    outDesc, Layout::RowMajor,
                                    Layout::RowMajor);
                            }
                        }
                        break;
                    }
                    default:
                        variantsToRun.emplace_back(mt.type, mt.nRows, mt.nCols,
                                                   order, Layout::RowMajor,
                                                   Layout::RowMajor);
                }
            }
            break;

        case CoopMatOpKind::Ternary:
            // For ternary operations (notably multiply-accumulate), derive the
            // variants to test directly from the runtime-reported variants.
            for (auto &rv : gTestContext->variants)
            {
                if (op == CoopMatOp::matrixmuladd_wrapping
                    && isFloatType(rv.result_type))
                    break;
                // Saturated variants are tested in the separate
                // matrixmuladd_saturating test.  Only add the saturating
                // variants for the matrixmuladd_saturating test, and only add
                // the non-saturating variants for the other tests.
                if (rv.saturating_accumulation
                    == (op == CoopMatOp::matrixmuladd_saturating))
                {
                    variantsToRun.emplace_back(rv, Variant::OperandOrder::OpABC,
                                               Layout::RowMajor,
                                               Layout::RowMajor);
                    if (op == CoopMatOp::matrixmuladd_stride0)
                    {
                        variantsToRun.back().strideA = 0;
                        variantsToRun.back().strideB = 0;
                        variantsToRun.back().strideC = 0;
                        // Keep output stride as is.
                    }
                }
            }

            if (variantsToRun.empty())
            {
                log_info(
                    "Device does not support any variants for this subtest.\n");
                return TEST_SKIPPED_ITSELF;
            }
            break;

        case CoopMatOpKind::Conversion:
            // For conversion/bitcast operations, derive the variants to test
            // from the Cartesian square of the list of unique types.
            for (auto &src : gTestContext->types)
            {
                for (auto &dst : gTestContext->types)
                {
                    // Skip identity conversions.
                    // Number of rows, columns, and use must be equal.
                    // Skip integer signed/unsigned conversions.
                    if (src == dst || !src.isConvertibleTo(dst)
                        || (!isFloatType(src.type) && !isFloatType(dst.type)
                            && elementSizeOf(src.type)
                                == elementSizeOf(dst.type)))
                        continue;

                    Variant::OperandOrder order =
                        getOrderFromKindAndUse(kind, src.use);
                    variantsToRun.emplace_back(
                        src.type, dst.type, src.nRows, src.nCols, src.use,
                        order, Layout::RowMajor, Layout::RowMajor);
                }
            }

            if (variantsToRun.empty())
            {
                log_info("No conversions are possible between any of the "
                         "matrix types supported by the device.\n");
                return TEST_SKIPPED_ITSELF;
            }
            break;
    }

    // Run the variants.
    int result = TEST_PASS;
    bool didAnyTestsRun = false;
    for (auto &v : variantsToRun)
    {
        if (!gTestContext->runSingleVariant.empty())
        {
            if (v.describe() != gTestContext->runSingleVariant) continue;
        }

        log_info("%s%s --variant %s\n", isSpirv ? "spirv_" : "", opName,
                 v.describe().c_str());
        if (buildAndRun(v) != TEST_PASS)
        {
            result = TEST_FAIL;
        }
        didAnyTestsRun = true;
    }

    // If a single variant was specified, make sure it is valid.
    if (!gTestContext->runSingleVariant.empty() && !didAnyTestsRun)
    {
        log_error("Unknown or unsupported variant '%s'.\n",
                  gTestContext->runSingleVariant.c_str());
        return TEST_FAIL;
    }

    return result;
}

} // anonymous namespace

// Define the top level test functions.
#define COOPMAT(op, kind)                                                      \
    REGISTER_TEST(spirv_##op)                                                  \
    {                                                                          \
        CoopMatTest_##op test(device, context, queue, #op, CoopMatOp::op,      \
                              CoopMatOpKind::kind);                            \
        return test.runAll();                                                  \
    }
#include "cooperative_matrix.def"
#undef COOPMAT
