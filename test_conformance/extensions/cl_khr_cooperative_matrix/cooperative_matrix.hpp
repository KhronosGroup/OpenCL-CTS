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

#ifndef COOPERATIVE_MATRIX_HPP
#define COOPERATIVE_MATRIX_HPP

#include <cassert>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "halffp.hpp"

////////////////////////////////>8////////////////////////////////
// TODO: drop these when OpenCL-Headers supports the extension.
#define CL_DEVICE_COOPERATIVE_MATRIX_DEFAULT_SUB_GROUP_VARIANTS_KHR 0x1078
#define CL_DEVICE_COOPERATIVE_MATRIX_SUB_GROUP_VARIANTS_KHR 0x1079
#define CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR 0x107A
#define CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR 0x107B

typedef cl_uint cl_device_cooperative_matrix_info_khr;
typedef cl_uint cl_device_cooperative_matrix_component_type_khr;

#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR 0
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR 1
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR 2
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR 3
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR 4
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT32_KHR 5
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR 6
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR 7
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR 8
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT32_KHR 9
#define CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR 10

typedef struct _cl_device_cooperative_matrix_variant_khr_st
{
    cl_uint m_size;
    cl_uint n_size;
    cl_uint k_size;
    cl_device_cooperative_matrix_component_type_khr a_type;
    cl_device_cooperative_matrix_component_type_khr b_type;
    cl_device_cooperative_matrix_component_type_khr c_type;
    cl_device_cooperative_matrix_component_type_khr result_type;
    cl_bool saturating_accumulation;
} cl_device_cooperative_matrix_variant_khr;

// clang-format off
typedef cl_int CL_API_CALL clGetDeviceCooperativeMatrixInfoKHR_t(
    cl_device_id device,
    cl_device_cooperative_matrix_info_khr param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);
// clang-format on

typedef clGetDeviceCooperativeMatrixInfoKHR_t
    *clGetDeviceCooperativeMatrixInfoKHR_fn;
////////////////////////////////>8////////////////////////////////

struct FloatBounds
{
    double min;
    double max;
    bool canBeNonFinite;
};

struct SignedBounds
{
    int64_t min;
    int64_t max;
};

struct UnsignedBounds
{
    uint64_t min;
    uint64_t max;
};

typedef std::variant<FloatBounds, SignedBounds, UnsignedBounds> Bounds;

// Type of cooperative matrix operation.
enum class CoopMatOpKind
{
    Unary,
    Binary,
    Ternary,
    Conversion,
};

// Operations to test.
enum class CoopMatOp
{
#define COOPMAT(x, kind) x,
#include "cooperative_matrix.def"
#undef COOPMAT
};

struct BufferElementType
{
    const uint32_t vectorLength;
    const cl_device_cooperative_matrix_component_type_khr scalarType;

    constexpr BufferElementType(
        const uint32_t vectorLength,
        const cl_device_cooperative_matrix_component_type_khr scalarType)
        : vectorLength(vectorLength), scalarType(scalarType)
    {}

    constexpr bool operator<(const BufferElementType &other) const
    {
        if (scalarType != other.scalarType)
        {
            return scalarType < other.scalarType;
        }
        return vectorLength < other.vectorLength;
    }

    constexpr bool operator==(const BufferElementType &other) const
    {
        return vectorLength == other.vectorLength
            && scalarType == other.scalarType;
    }
};

uint32_t bufferElementTypeSizeOf(const BufferElementType &t);

template <uint32_t n> struct IndexedBufferElementType : BufferElementType
{
    constexpr IndexedBufferElementType(
        const cl_device_cooperative_matrix_component_type_khr scalarType)
        : BufferElementType(n, scalarType)
    {}
};

// Type and dimensions of a matrix.  Corresponds to the matrix data contained in
// the device query.
struct MatrixType
{
    // Use of this Cooperative Matrix type; i.e., whether it is the LHS (A), RHS
    // (B), or Accumulator matrix in a multiply-accumulate operation.
    // Must match SPV_KHR_cooperative_matrix's Cooperative Matrix Use.
    enum class Use
    {
        A = 0, // MatrixAKHR
        B = 1, // MatrixBKHR
        Acc = 2 // MatrixAccumulatorKHR
    };

    MatrixType(cl_device_cooperative_matrix_component_type_khr type,
               cl_uint nRows, cl_uint nCols, Use use)
        : type(type), nRows(nRows), nCols(nCols), use(use)
    {}

    // Compare function to enable use in e.g. std::set.
    bool operator<(const MatrixType &other) const;
    bool operator==(const MatrixType &other) const;

    // Return true if this type is convertible to 'other'.  That is, whether
    // destination type and sourcee type have the same number of Rows, number of
    // Columns, and Use.
    bool isConvertibleTo(const MatrixType &other) const;

    const cl_device_cooperative_matrix_component_type_khr type;
    const cl_uint nRows;
    const cl_uint nCols;
    const Use use;
};

enum class Layout
{
    RowMajor = 0x0, // RowMajorKHR
    ColumnMajor = 0x1 // ColumnMajorKHR
};

uint32_t elementsPerStride(const Layout layout, const uint32_t rows,
                           const uint32_t cols);
uint32_t numberOfStrides(const Layout layout, const uint32_t rows,
                         const uint32_t cols);
uint32_t elementSizeOf(const cl_device_cooperative_matrix_component_type_khr t);
const char *
showScalarType(const cl_device_cooperative_matrix_component_type_khr t);
std::string
describeBufferKind(const cl_device_cooperative_matrix_component_type_khr kind);
bool isFloatType(const cl_device_cooperative_matrix_component_type_khr t);
bool isSignedType(const cl_device_cooperative_matrix_component_type_khr t);

struct BufferDescriptor
{
    // Elements per stride.
    const uint32_t stride;

    // Number of strides.
    const uint32_t strideCount;

    // Element type.
    const BufferElementType elementType;

    // Padding after each stride. Needed to satisfy
    // CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR.
    const uint32_t stridePadding;

    static BufferDescriptor makeBufferDescriptor(
        const cl_device_cooperative_matrix_component_type_khr matElementType,
        const uint32_t stride, const uint32_t strideCount,
        const BufferElementType elementType);

private:
    explicit BufferDescriptor(const uint32_t stride,
                              const uint32_t stride_count,
                              const BufferElementType elementType,
                              const uint32_t stridePadding)
        : stride(stride), strideCount(stride_count), elementType(elementType),
          stridePadding(stridePadding)
    {}
};

uint32_t bufferStrideSizeOf(const BufferDescriptor &d);
uint32_t bufferSizeOf(const BufferDescriptor &d);
uint32_t bufferElementStride(const BufferDescriptor &d);

// Semantic buffer. Used together with cooperativeMatrixLoad and
// cooperativeMatrixStore. Respects the same padding requirements as the real CL
// buffer, but we do not care about the alignment. In short, its purpose is to
// simulate a buffer in the semantic domain.
struct SemBuffer
{
    const BufferDescriptor descriptor;
    std::vector<uint8_t> data;

    SemBuffer(const BufferDescriptor descriptor, std::vector<uint8_t> data)
        : descriptor(std::move(descriptor)), data(std::move(data))
    {}
};

struct Matrix
{
    explicit Matrix(const cl_device_cooperative_matrix_component_type_khr t,
                    cl_uint nRows, cl_uint nCols)
        : elementType(t), nRows(nRows), nCols(nCols)
    {
        const size_t num =
            static_cast<size_t>(nRows) * nCols * elementSizeOf(elementType);
        data.resize(num);
    }
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;
    Matrix(Matrix &&other)
        : elementType(other.elementType), nRows(other.nRows),
          nCols(other.nCols), data(std::move(other.data))
    {}
    Matrix &operator=(Matrix &&) = delete;

    ~Matrix() = default;

    // Number of elements in the matrix.
    uint32_t elementCount() const { return nRows * nCols; }

    // Flatten a (row,col) index into a linear index.
    unsigned getIndex(unsigned row, unsigned col) const;

    // Retrieve the i-th element and convert it to T.
    template <typename T> T get(size_t offset) const
    {
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, HalfFP>)
            return getF64(offset);
        if constexpr (std::is_signed_v<T>) return getS64(offset);
        if constexpr (std::is_unsigned_v<T>) return getU64(offset);
        assert(false && "Unhandled type");
        std::abort();
    }

    // Set the i-th element from a double value.  The value is converted to the
    // matrix element type before storing.
    template <typename T> void set(size_t offset, T value)
    {
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, HalfFP>)
            setF64(offset, value);
        else if constexpr (std::is_signed_v<T>)
            setS64(offset, value);
        else if constexpr (std::is_unsigned_v<T>)
            setU64(offset, value);
        else
            assert(false && "Unhandled type");
    }

    // Fill matrix with test data.
    // seed is used to make sure the matrix for non-unary tests are different.
    // canBeSigned is used to stop a signed number being generated for
    // conversion tests. canBeNonFinite is used to indicate if floats can be
    // positive or negative infinity or nan. bounds is used to stop overflows in
    // conversion and multiply adds.
    void fill(int8_t seed, std::optional<Bounds> bounds = std::nullopt);

    // Deep copy from another matrix with the same shape and type.
    void copyFrom(const Matrix &other);

    std::string showElementAtIndex(size_t index) const;

    void print(const char *title) const;

    const cl_device_cooperative_matrix_component_type_khr elementType;
    const cl_uint nRows;
    const cl_uint nCols;

    static Matrix cooperativeMatrixLoad(
        cl_device_cooperative_matrix_component_type_khr elementType,
        uint32_t rows, uint32_t cols, SemBuffer &buf, Layout layout,
        uint32_t stride);
    static void cooperativeMatrixStore(SemBuffer &buf, const Matrix &mat,
                                       Layout layout, uint32_t stride);

private:
    // Reads the matrix and upcasts to uint64
    uint64_t getU64(size_t offset) const;
    // Reads the matrix and upcasts to int64
    int64_t getS64(size_t offset) const;
    // Reads the matrix and upcasts to double
    double getF64(size_t offset) const;

    /// Sets the element at the given offset after conversion
    /// to the matrix's element type.
    void setU64(size_t i, uint64_t value);
    /// Sets the element at the given offset after conversion
    /// to the matrix's element type.
    void setS64(size_t i, int64_t value);
    /// Sets the element at the given offset after conversion
    /// to the matrix's element type.
    void setF64(size_t i, double value);

    // Data contained in the matrix. Always row-major without any element or
    // stride padding.
    std::vector<uint8_t> data;
};

// Variant to test. Holds the storage for the input, output, and reference
// output matrices.
class Variant {
public:
    enum class OperandOrder
    {
        OpA, // result = op(A)
        OpB, // result = op(B)
        OpC, // result = op(C)
        OpAA, // result = op(A, A)
        OpBB, // result = op(B, B)
        OpCC, // result = op(C, C)
        OpABC, // result = op(A, B, C)
    };

    // Construct a Variant for a single matrix type.
    Variant(cl_device_cooperative_matrix_component_type_khr type, cl_uint nRows,
            cl_uint nCols, OperandOrder o, Layout inputLayout,
            Layout outputLayout)
        : order(o), inputA(type, nRows, nCols), inputB(type, nRows, nCols),
          inputC(type, nRows, nCols), output(type, nRows, nCols),
          outputRef(type, nRows, nCols),
          inputADesc(BufferDescriptor::makeBufferDescriptor(
              type, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(type))),
          inputBDesc(BufferDescriptor::makeBufferDescriptor(
              type, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(type))),
          inputCDesc(BufferDescriptor::makeBufferDescriptor(
              type, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(type))),
          outputDesc(BufferDescriptor::makeBufferDescriptor(
              type, elementsPerStride(outputLayout, nRows, nCols),
              numberOfStrides(outputLayout, nRows, nCols),
              IndexedBufferElementType<1>(type))),
          layoutA(inputLayout), layoutB(inputLayout), layoutC(inputLayout),
          layoutRes(outputLayout), strideA(bufferElementStride(inputADesc)),
          strideB(bufferElementStride(inputBDesc)),
          strideC(bufferElementStride(inputCDesc)),
          strideRes(bufferElementStride(outputDesc)), isConversion(false),
          isSaturating(false), isMulticomponent(false)
    {
        assert(o != OperandOrder::OpABC
               && "constructor must not be used for ternary operations");
    }

    // Construct a Variant for a single matrix type, but with a different
    // output type; primarily intended for the length operation.
    Variant(cl_device_cooperative_matrix_component_type_khr inputType,
            cl_device_cooperative_matrix_component_type_khr outputType,
            cl_uint nRows, cl_uint nCols, OperandOrder o, Layout inputLayout,
            Layout outputLayout)
        : order(o), inputA(inputType, nRows, nCols),
          inputB(inputType, nRows, nCols), inputC(inputType, nRows, nCols),
          output(outputType, nRows, nCols), outputRef(outputType, nRows, nCols),
          inputADesc(BufferDescriptor::makeBufferDescriptor(
              inputType, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(inputType))),
          inputBDesc(BufferDescriptor::makeBufferDescriptor(
              inputType, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(inputType))),
          inputCDesc(BufferDescriptor::makeBufferDescriptor(
              inputType, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(inputType))),
          outputDesc(BufferDescriptor::makeBufferDescriptor(
              outputType, elementsPerStride(outputLayout, nRows, nCols),
              numberOfStrides(outputLayout, nRows, nCols),
              IndexedBufferElementType<1>(outputType))),
          layoutA(inputLayout), layoutB(inputLayout), layoutC(inputLayout),
          layoutRes(outputLayout), strideA(bufferElementStride(inputADesc)),
          strideB(bufferElementStride(inputBDesc)),
          strideC(bufferElementStride(inputCDesc)),
          strideRes(bufferElementStride(outputDesc)), isConversion(false),
          isSaturating(false), isMulticomponent(false)
    {
        assert((o == OperandOrder::OpA || o == OperandOrder::OpB
                || o == OperandOrder::OpC)
               && "constructor must not be used for non-unary operations");
    }

    // Construct a Variant for a conversion.
    Variant(cl_device_cooperative_matrix_component_type_khr srcType,
            cl_device_cooperative_matrix_component_type_khr dstType,
            cl_uint nRows, cl_uint nCols, MatrixType::Use u, OperandOrder o,
            Layout inputLayout, Layout outputLayout)
        : order(o), inputA(srcType, nRows, nCols),
          inputB(srcType, nRows, nCols), inputC(srcType, nRows, nCols),
          output(dstType, nRows, nCols), outputRef(dstType, nRows, nCols),
          inputADesc(BufferDescriptor::makeBufferDescriptor(
              srcType, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(srcType))),
          inputBDesc(BufferDescriptor::makeBufferDescriptor(
              srcType, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(srcType))),
          inputCDesc(BufferDescriptor::makeBufferDescriptor(
              srcType, elementsPerStride(inputLayout, nRows, nCols),
              numberOfStrides(inputLayout, nRows, nCols),
              IndexedBufferElementType<1>(srcType))),
          outputDesc(BufferDescriptor::makeBufferDescriptor(
              dstType, elementsPerStride(outputLayout, nRows, nCols),
              numberOfStrides(outputLayout, nRows, nCols),
              IndexedBufferElementType<1>(dstType))),
          layoutA(inputLayout), layoutB(inputLayout), layoutC(inputLayout),
          layoutRes(outputLayout), strideA(bufferElementStride(inputADesc)),
          strideB(bufferElementStride(inputBDesc)),
          strideC(bufferElementStride(inputCDesc)),
          strideRes(bufferElementStride(outputDesc)), isConversion(true),
          isSaturating(false), isMulticomponent(false)
    {
        assert((o == OperandOrder::OpA || o == OperandOrder::OpB
                || o == OperandOrder::OpC)
               && "constructor must not be used for non-unary operations");
    }

    // Construct a Variant for a multicomponent_load/store.
    Variant(cl_device_cooperative_matrix_component_type_khr type, cl_uint nRows,
            cl_uint nCols, OperandOrder o, const BufferDescriptor inputBuffer,
            const BufferDescriptor outputBuffer, Layout inputLayout,
            Layout outputLayout)
        : order(o), inputA(type, nRows, nCols), inputB(type, nRows, nCols),
          inputC(type, nRows, nCols), output(type, nRows, nCols),
          outputRef(type, nRows, nCols), inputADesc(inputBuffer),
          inputBDesc(inputBuffer), inputCDesc(inputBuffer),
          outputDesc(outputBuffer), layoutA(inputLayout), layoutB(inputLayout),
          layoutC(inputLayout), layoutRes(outputLayout),
          strideA(bufferElementStride(inputADesc)),
          strideB(bufferElementStride(inputBDesc)),
          strideC(bufferElementStride(inputCDesc)),
          strideRes(bufferElementStride(outputDesc)), isConversion(false),
          isSaturating(false), isMulticomponent(true)
    {}

    // Construct a Variant from config data returned by device query.
    Variant(const cl_device_cooperative_matrix_variant_khr &v, OperandOrder o,
            Layout inputLayout, Layout outputLayout)
        : order(o), inputA(v.a_type, v.m_size, v.k_size),
          inputB(v.b_type, v.k_size, v.n_size),
          inputC(v.c_type, v.m_size, v.n_size),
          output(v.result_type, v.m_size, v.n_size),
          outputRef(v.result_type, v.m_size, v.n_size),
          inputADesc(BufferDescriptor::makeBufferDescriptor(
              v.a_type, elementsPerStride(inputLayout, v.m_size, v.k_size),
              numberOfStrides(inputLayout, v.m_size, v.k_size),
              IndexedBufferElementType<1>(v.a_type))),
          inputBDesc(BufferDescriptor::makeBufferDescriptor(
              v.b_type, elementsPerStride(inputLayout, v.k_size, v.n_size),
              numberOfStrides(inputLayout, v.k_size, v.n_size),
              IndexedBufferElementType<1>(v.b_type))),
          inputCDesc(BufferDescriptor::makeBufferDescriptor(
              v.c_type, elementsPerStride(inputLayout, v.m_size, v.n_size),
              numberOfStrides(inputLayout, v.m_size, v.n_size),
              IndexedBufferElementType<1>(v.c_type))),
          outputDesc(BufferDescriptor::makeBufferDescriptor(
              v.result_type,
              elementsPerStride(outputLayout, v.m_size, v.n_size),
              numberOfStrides(outputLayout, v.m_size, v.n_size),
              IndexedBufferElementType<1>(v.result_type))),
          layoutA(inputLayout), layoutB(inputLayout), layoutC(inputLayout),
          layoutRes(outputLayout), strideA(bufferElementStride(inputADesc)),
          strideB(bufferElementStride(inputBDesc)),
          strideC(bufferElementStride(inputCDesc)),
          strideRes(bufferElementStride(outputDesc)), isConversion(false),
          isSaturating(v.saturating_accumulation), isMulticomponent(false)
    {
        assert(o == OperandOrder::OpABC
               && "constructor must only be used for ternary operations");
    }

    // Return a string that identifies the variant.
    std::string describe() const;

    // Accessors for matrices and buffer descriptors by index (A=0, B=1, C=2,
    // Res=3).
    const Matrix &getMatrix(uint8_t matrixID) const;
    const BufferDescriptor &getBufferDescriptor(uint8_t bufferID) const;

    // Return a vector with the input matrices for the OperandOrder of this
    // Variant.
    void getInputsForOperation(std::vector<const Matrix *> &inputs) const;

public:
    const OperandOrder order;
    Matrix inputA;
    Matrix inputB;
    Matrix inputC;
    Matrix output;
    Matrix outputRef;
    const BufferDescriptor inputADesc;
    const BufferDescriptor inputBDesc;
    const BufferDescriptor inputCDesc;
    const BufferDescriptor outputDesc;
    const Layout layoutA;
    const Layout layoutB;
    const Layout layoutC;
    const Layout layoutRes;
    uint32_t strideA;
    uint32_t strideB;
    uint32_t strideC;
    uint32_t strideRes;
    bool isConversion;
    bool isSaturating;
    bool isMulticomponent;
    size_t globalSize;
};

// Global test data.
struct TestContext
{
    // The result of the CL_DEVICE_ADDRESS_BITS device query.
    std::string addrWidth;

    // Name of variant if only one variant is to be tested; empty if all
    // variants are to be tested.
    std::string runSingleVariant;

    // Runtime-reported supported matrix variants.
    std::vector<cl_device_cooperative_matrix_variant_khr> variants;

    // Runtime-reported supported matrix types.
    std::set<MatrixType> types;

    // Runtime-reported support for double floating point extension.
    bool supportFP64 = false;

    // Only build programs; do not execute or verify output.
    bool linkCheckOnly = false;

    // Some devices have special restrictions on the pointer and stride
    // arguments to OpCooperativeMatrixLoadKHR and OpCooperativeMatrixStoreKHR.
    // These need to be queried by
    // CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR and
    // CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR, respectively.
    uint32_t devicePointerAlignment;
    uint32_t deviceStrideMultiple;
};

#endif // COOPERATIVE_MATRIX_HPP
