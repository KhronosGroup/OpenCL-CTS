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

#include "program.hpp"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string_view>

#include <spirv-tools/libspirv.hpp>

#include "harness/errorHelpers.h"

#include "cooperative_matrix.hpp"

extern const TestContext *gTestContext;

static std::string
bufferKindString(cl_device_cooperative_matrix_component_type_khr kind)
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

static std::string bufferTypeString(const BufferDescriptor &d)
{
    std::string res;
    if (d.elementType.vectorLength > 1)
    {
        res += "vec" + std::to_string(d.elementType.vectorLength) + "_";
    }
    res += bufferKindString(d.elementType.scalarType);
    return res;
}

std::ostream &operator<<(std::ostream &out, MatrixType::Use use)
{
    out << static_cast<unsigned>(use);
    return out;
}

std::ostream &operator<<(std::ostream &out, Layout layout)
{
    out << static_cast<unsigned>(layout);
    return out;
}

std::ostream &operator<<(std::ostream &out, Variant::OperandOrder order)
{
    switch (order)
    {
        case Variant::OperandOrder::OpA:
        case Variant::OperandOrder::OpAA: return out << "A";
        case Variant::OperandOrder::OpB:
        case Variant::OperandOrder::OpBB: return out << "B";
        case Variant::OperandOrder::OpC:
        case Variant::OperandOrder::OpCC: return out << "C";
        default:
            assert(false && "unexpected operand order to print");
            std::abort();
    }
    return out;
}

void ProgramGenerator::genTypeDecls()
{
    // Determine the scalar types to declare.  A SPIR-V module cannot contain
    // duplicate non-aggregate type declarations.
    std::set<std::string> types;

    // Always declare i32 for e.g. matrix sizes or invocation IDs.
    types.insert("i32");

    // Collect types for all input and output matrices.
    for (int i = 0; i < 4; i++)
    {
        types.insert(showScalarType(variant.getMatrix(i).elementType));
        types.insert(describeBufferKind(
            variant.getBufferDescriptor(i).elementType.scalarType));
    }

    // Emit the type declarations for the matrix types.
    for (const auto &t : types)
    {
        // Declare the scalar type.
        if (t[0] == 'f')
        {
            spirv_text << "    %" << t << " = OpTypeFloat " << t.substr(1)
                       << "\n";
        }
        else if (t[0] == 'i')
        {
            spirv_text << "    %" << t << " = OpTypeInt " << t.substr(1)
                       << " 0\n";
        }

        // Declare the cross-workgroup pointer type.
        spirv_text << "    %ptr_" << t << " = OpTypePointer CrossWorkgroup %"
                   << t << "\n";
    }

    // Declare the input pointer type.
    spirv_text << "    %iptr_i32 = OpTypePointer Input %i32\n";

    // Then, emit the vector declarations for the input/output buffer (if any).
    std::set<BufferElementType> bufElementTypes;
    std::array<const BufferDescriptor, 4> descs = { variant.inputADesc,
                                                    variant.inputBDesc,
                                                    variant.inputCDesc,
                                                    variant.outputDesc };
    for (const BufferDescriptor desc : descs)
    {
        if (desc.elementType.vectorLength > 1)
        {
            bufElementTypes.insert(desc.elementType);
        }
    }
    for (const BufferElementType bufElementType : bufElementTypes)
    {
        const BufferDescriptor desc = BufferDescriptor::makeBufferDescriptor(
            bufElementType.scalarType, 1, 1, bufElementType);
        const std::string typeStr = bufferTypeString(desc);
        std::string kindStr = bufferKindString(bufElementType.scalarType);
        spirv_text << "    %" << typeStr << " = OpTypeVector %" << kindStr
                   << " " << std::to_string(bufElementType.vectorLength)
                   << "\n";

        // Declare the cross-workgroup pointer type for input/output buffer
        // pointers.
        spirv_text << "    %ptr_" << typeStr
                   << " = OpTypePointer CrossWorkgroup %" << typeStr << "\n";
    }


    // Declare the kernel function type.
    spirv_text << "    %FnTy = OpTypeFunction %void";
    spirv_text << " %ptr_" << bufferTypeString(variant.inputADesc);
    spirv_text << " %ptr_" << bufferTypeString(variant.inputBDesc);
    spirv_text << " %ptr_" << bufferTypeString(variant.inputCDesc);
    spirv_text << " %ptr_" << bufferTypeString(variant.outputDesc);

    spirv_text << "\n";
}

void ProgramGenerator::genConstants()
{
    auto getResTy = [this]() {
        std::ostringstream resTyOS;
        resTyOS << "%mat" << variant.order << "ty";
        return resTyOS.str();
    };
    switch (op)
    {
        case CoopMatOp::length:
        case CoopMatOp::copy:
        case CoopMatOp::copy_stride0:
        case CoopMatOp::convert:
        case CoopMatOp::negate:
        case CoopMatOp::add:
        case CoopMatOp::sub:
        case CoopMatOp::mul:
        case CoopMatOp::div:
        case CoopMatOp::multicomponent_load:
        case CoopMatOp::multicomponent_store:
            // Nothing to do.
            break;
        case CoopMatOp::copy_workgroup:
            spirv_text << R"(
    %bool = OpTypeBool

    %numElems = OpConstant %i32 )"
                       << variant.output.elementCount() << R"(
    %arrayTy = OpTypeArray %)"
                       << showScalarType(variant.output.elementType)
                       << R"( %numElems
    %arrayPtr = OpTypePointer Function %arrayTy
    %workgroupPtr = OpTypePointer Workgroup %arrayTy
    %sharedBufferLoad = OpVariable %workgroupPtr Workgroup
    %sharedBufferStore = OpVariable %workgroupPtr Workgroup
    %zero = OpConstant %i32 0
    %scopeWorkgroup = OpConstant %i32 2
    %workAcquire  = OpConstant %i32 0x108
    %workGroupPtrFloat = OpTypePointer Workgroup %)"
                       << showScalarType(variant.output.elementType) << "\n";
            break;
        case CoopMatOp::constant:
        case CoopMatOp::composite:
            spirv_text << R"(
    %fillValue = OpConstant %)"
                       << showScalarType(variant.inputA.elementType)
                       << " 123\n";
            if (op == CoopMatOp::constant)
                spirv_text << R"(
    %result = OpConstantComposite %matAty %fillValue
            )";
            break;
        case CoopMatOp::composite_array:
            spirv_text << R"(
    %bool = OpTypeBool
    %zero = OpConstant %i32 0
    %one = OpConstant %i32 1
    %two = OpConstant %i32 2
    %fillValue = OpConstant %)"
                       << showScalarType(variant.output.elementType) << R"( 121
    %fillValue2 = OpConstant %)"
                       << showScalarType(variant.output.elementType) << R"( 122
    %fillValue3 = OpConstant %)"
                       << showScalarType(variant.output.elementType) << R"( 123
    %matArrty = OpTypeArray %mat)"
                       << variant.order << R"(ty %two
    %filled = OpConstantComposite %mat)"
                       << variant.order << R"(ty %fillValue
    %matOArrty = OpTypePointer Function %matArrty
    %ptr = OpTypePointer Function %)"
                       << showScalarType(variant.output.elementType) << R"(
    %ity = OpTypePointer Function %i32
    %matptr = OpTypePointer Function %mat)"
                       << variant.order << R"(ty
)";
            break;
        case CoopMatOp::composite_rvalue:
            spirv_text << R"(
    %bool = OpTypeBool
    %zero = OpConstant %i32 0
    %zeroO = OpConstant %)"
                       << showScalarType(variant.output.elementType) << R"( 0
    %ptr = OpTypePointer Function %)"
                       << showScalarType(variant.output.elementType) << R"(
    %ity = OpTypePointer Function %i32
    %matptr = OpTypePointer Function )"
                       << getResTy() << R"(

    %fillValue = OpConstant %)"
                       << showScalarType(variant.output.elementType) << R"( 123
    %filled = OpConstantComposite %mat)"
                       << variant.order << R"(ty %fillValue
)";
            break;
        case CoopMatOp::matrixmuladd_array:
            spirv_text << R"(
    %two = OpConstant %i32 2
    %fillValueA = OpConstant %)"
                       << showScalarType(variant.inputA.elementType) << R"( 123
    %fillValueB = OpConstant %)"
                       << showScalarType(variant.inputB.elementType) << R"( 123
    %fillValueC = OpConstant %)"
                       << showScalarType(variant.inputC.elementType) << R"( 123
    %matAArrty = OpTypeArray %matAty %two
    %matBArrty = OpTypeArray %matBty %two
    %matCArrty = OpTypeArray %matCty %two
    %filledA = OpConstantComposite %matAty %fillValueA
    %filledB = OpConstantComposite %matBty %fillValueB
    %filledC = OpConstantComposite %matCty %fillValueC
)";
        case CoopMatOp::matrixmuladd:
        case CoopMatOp::matrixmuladd_wrapping:
        case CoopMatOp::matrixmuladd_stride0:
        case CoopMatOp::matrixmuladd_saturating: {
            std::ostringstream tmpStringStream("");
            const auto add = [&](bool cond, const char *name) {
                if (cond)
                {
                    tmpStringStream << "|" << name;
                }
            };

            add(isSignedType(variant.inputA.elementType)
                    && !isFloatType(variant.inputA.elementType),
                "MatrixASignedComponentsKHR");
            add(isSignedType(variant.inputB.elementType)
                    && !isFloatType(variant.inputB.elementType),
                "MatrixBSignedComponentsKHR");
            add(isSignedType(variant.inputC.elementType)
                    && !isFloatType(variant.inputC.elementType),
                "MatrixCSignedComponentsKHR");
            add(isSignedType(variant.output.elementType)
                    && !isFloatType(variant.output.elementType),
                "MatrixResultSignedComponentsKHR");
            add(variant.isSaturating, "SaturatingAccumulationKHR");

            muladdOperand = tmpStringStream.str();
            // Replace the leading '|' with a space
            if (!muladdOperand.empty())
            {
                muladdOperand[0] = ' ';
            }
            break;
        }
        case CoopMatOp::func:
            spirv_text << R"(
    %matptr = OpTypePointer Function   )"
                       << getResTy() << R"(
    %fnty = OpTypeFunction )"
                       << getResTy() << R"( %matptr
    %identity = OpFunction )"
                       << getResTy() << R"( None %fnty
    %m = OpFunctionParameter %matptr
    %entry = OpLabel
    %arg = OpLoad )" << getResTy()
                       << R"( %m
    OpReturnValue %arg
    OpFunctionEnd)";
            break;
    }
}


void ProgramGenerator::genVariables()
{
    switch (op)
    {

        case CoopMatOp::length:
        case CoopMatOp::constant:
        case CoopMatOp::copy:
        case CoopMatOp::copy_stride0:
        case CoopMatOp::convert:
        case CoopMatOp::composite:
        case CoopMatOp::negate:
        case CoopMatOp::add:
        case CoopMatOp::sub:
        case CoopMatOp::mul:
        case CoopMatOp::div:
        case CoopMatOp::matrixmuladd:
        case CoopMatOp::matrixmuladd_array:
        case CoopMatOp::matrixmuladd_wrapping:
        case CoopMatOp::matrixmuladd_stride0:
        case CoopMatOp::copy_workgroup:
        case CoopMatOp::matrixmuladd_saturating:
        case CoopMatOp::multicomponent_load:
        case CoopMatOp::multicomponent_store: break;
        case CoopMatOp::composite_array:
            spirv_text << R"(
    %matArrVar = OpVariable %matOArrty Function
    %matOArr = OpVariable %matOArrty Function
    %i = OpVariable %ity Function)";
            break;
        case CoopMatOp::composite_rvalue:
            spirv_text << R"(
    %matOVar = OpVariable %matptr Function
    %tempVar = OpVariable %matptr Function)";
            break;

        case CoopMatOp::func:
            spirv_text << R"(
    %matVar = OpVariable %matptr Function)";
    }
}

void ProgramGenerator::genBody()
{
    auto getResTy = [this]() {
        std::ostringstream resTyOS;
        resTyOS << "%mat" << variant.order << "ty";
        return resTyOS.str();
    };

    // clang-format off
    switch (op)
    {
        case CoopMatOp::length:
            spirv_text << R"(
    %1 = OpCooperativeMatrixLengthKHR %i32 )" << getResTy() << R"(
    OpStore %out_slid_offset %1 Aligned 32)";
            break;

        case CoopMatOp::constant:
            spirv_text << R"(
    OpCooperativeMatrixStoreKHR %out %result %layoutRes %strideRes)";
            break;

        case CoopMatOp::copy:
        case CoopMatOp::copy_stride0:
        case CoopMatOp::multicomponent_load:
        case CoopMatOp::multicomponent_store:
            spirv_text << R"(
    %matSrc = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order << R"(
    OpCooperativeMatrixStoreKHR %out %matSrc %layoutRes %strideRes)";
            break;
        case CoopMatOp::copy_workgroup:

            spirv_text << R"(
    %scalarPtrLoad = OpAccessChain %workGroupPtrFloat %sharedBufferLoad %zero
    %scalarPtrStore = OpAccessChain %workGroupPtrFloat %sharedBufferStore %zero

    %matSrc = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order<< R"(
    OpCooperativeMatrixStoreKHR %scalarPtrLoad %matSrc)" << R"( %layout)" << variant.order << R"( %stride)" << variant.order << R"(

    %slIsZero = OpIEqual %bool %zero %slid
    %sgIsZero = OpIEqual %bool %zero %sgid
    %branch   = OpLogicalAnd %bool %sgIsZero %slIsZero
    OpSelectionMerge %falseLabel None
    OpBranchConditional %branch %trueLabel %falseLabel
    %trueLabel = OpLabel

    OpCopyMemory %sharedBufferStore %sharedBufferLoad

    OpBranch %falseLabel
    %falseLabel = OpLabel
    OpControlBarrier %scopeWorkgroup %scopeWorkgroup %workAcquire
    %matStored = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %scalarPtrStore)" << R"( %layout)" << variant.order << R"( %stride)" << variant.order<< R"(
    OpCooperativeMatrixStoreKHR %out %matStored %layoutRes %strideRes)";
            break;

        case CoopMatOp::composite:
            spirv_text << R"(
    %result = OpCompositeConstruct )" << getResTy() << R"( %fillValue
    OpCooperativeMatrixStoreKHR %out %result %layoutRes %strideRes)";
            break;
        case CoopMatOp::composite_array:
            spirv_text << R"(
    %mat = OpCooperativeMatrixLoadKHR %mat)" << variant.order << R"(ty %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order << R"(
    %matArr = OpCompositeConstruct %matArrty %filled %mat
    %matFilled = OpCompositeConstruct %matArrty %filled %filled
    OpStore %matArrVar %matArr
    OpStore %matOArr %matFilled
    OpStore %i %zero

    OpBranch %forLoop
    %forLoop = OpLabel
    %iVal = OpLoad %i32 %i
    %len = OpCooperativeMatrixLengthKHR %i32 %mat)" << variant.order << R"(ty
    %cond = OpSLessThan %bool %iVal %len
    OpBranchConditional %cond %body %exit

    %body = OpLabel
    %val = OpLoad %i32 %i
    %src_ptr = OpAccessChain %ptr %matArrVar %one %val
    %dest_ptr = OpAccessChain %ptr %matOArr %one %val
    %element = OpLoad %)" << showScalarType(variant.getMatrix(3).elementType) << R"( %src_ptr
    OpStore %dest_ptr %element

    %inc = OpIAdd %i32 %val %one
    OpStore %i %inc
    OpBranch %forLoop

    %exit = OpLabel

    %matOPtr = OpAccessChain %matptr %matOArr %one
    %matO = OpLoad %mat)" << variant.order << R"(ty %matOPtr
    OpCooperativeMatrixStoreKHR %out %matO %layoutRes %strideRes)";
            break;
        case CoopMatOp::composite_rvalue:
            spirv_text << R"(
    %mat = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order << R"(

    %matIns = OpCompositeInsert )" << getResTy() << R"( %zeroO %mat 0
    OpStore %tempVar %filled
    %length = OpCooperativeMatrixLengthKHR %i32 )" << getResTy() << R"(
    %nonEmpty = OpSGreaterThan %bool %length %zero
    OpSelectionMerge %endIf None
    OpBranchConditional %nonEmpty %then %else

    %then = OpLabel
    OpStore %tempVar %mat
    %tempMat = OpLoad )" << getResTy() << R"( %tempVar
    %val = OpCompositeExtract %)" << showScalarType(variant.getMatrix(3).elementType) << R"( %tempMat 0
    %matReIns = OpCompositeInsert )" << getResTy() << R"( %val %mat 0
    OpStore %matOVar %matReIns
    OpBranch %endIf

    %else = OpLabel
    OpStore %matOVar %matIns
    OpBranch %endIf

    %endIf = OpLabel
    %matO = OpLoad %mat)" << variant.order << R"(ty %matOVar
    OpCooperativeMatrixStoreKHR %out %matO %layoutRes %strideRes)";
            break;

        case CoopMatOp::negate: {
            std::ostringstream unaryOp;
            unaryOp << "Op"
                    << (isFloatType(variant.output.elementType) ? "F" : "S")
                    << "Negate";
            spirv_text << R"(
    %matA = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order << R"(
    %result = )" << unaryOp.str() << R"( )" << getResTy() << R"( %matA
    OpCooperativeMatrixStoreKHR %out %result %layoutRes %strideRes)";
            break;
        }

        case CoopMatOp::add:
        case CoopMatOp::sub:
        case CoopMatOp::mul:
        case CoopMatOp::div: {
            std::ostringstream binOp;
            binOp << "Op"
                  << (isFloatType(variant.output.elementType) ? "F" :
                      op != CoopMatOp::div ? "I" :
                      isSignedType(variant.output.elementType) ? "S" : "U")
                  << (op == CoopMatOp::negate ? "Negate" :
                      op == CoopMatOp::add ? "Add" :
                      op == CoopMatOp::sub ? "Sub" :
                      op == CoopMatOp::mul ? "Mul" :
                      op == CoopMatOp::div ? "Div" : "???");
            spirv_text << R"(
    %matA = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %inA %layoutA %strideA
    %matB = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %inB %layoutB %strideB
    %result = )" << binOp.str() << R"( )" << getResTy() << R"( %matA %matB
    OpCooperativeMatrixStoreKHR %out %result %layoutRes %strideRes)";
            break;
        }
        case CoopMatOp::matrixmuladd:
        case CoopMatOp::matrixmuladd_wrapping:
        case CoopMatOp::matrixmuladd_stride0:
        case CoopMatOp::matrixmuladd_saturating:
            spirv_text << R"(
    %matA = OpCooperativeMatrixLoadKHR %matAty %inA %layoutA %strideA
    %matB = OpCooperativeMatrixLoadKHR %matBty %inB %layoutB %strideB
    %matC = OpCooperativeMatrixLoadKHR %matCty %inC %layoutC %strideC
    %result = OpCooperativeMatrixMulAddKHR %matCty %matA %matB %matC)" << muladdOperand << R"(
    OpCooperativeMatrixStoreKHR %out %result %layoutRes %strideRes)";
            break;
        case CoopMatOp::matrixmuladd_array:
            spirv_text << R"(
    %matA = OpCooperativeMatrixLoadKHR %matAty %inA %layoutA %strideA
    %matB = OpCooperativeMatrixLoadKHR %matBty %inB %layoutB %strideB
    %matC = OpCooperativeMatrixLoadKHR %matCty %inC %layoutC %strideC
    %matAArr = OpCompositeConstruct %matAArrty %filledA %matA
    %matBArr = OpCompositeConstruct %matBArrty %filledB %matB
    %matCArr = OpCompositeConstruct %matCArrty %filledC %matC
    %matAArr1 = OpCompositeExtract %matAty %matAArr 1
    %matBArr1 = OpCompositeExtract %matBty %matBArr 1
    %matCArr1 = OpCompositeExtract %matCty %matCArr 1

    %result = OpCooperativeMatrixMulAddKHR %matCty %matA %matB %matC)" << muladdOperand << R"(
    OpCooperativeMatrixStoreKHR %out %result %layoutRes %strideRes)";
            break;

        case CoopMatOp::func:
            spirv_text << R"(
    %mat = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order << R"(
    OpStore %matVar %mat
    %matO = OpFunctionCall )" << getResTy() << R"( %identity %matVar
    OpCooperativeMatrixStoreKHR %out %matO %layoutRes %strideRes)";
            break;
        case CoopMatOp::convert:
            spirv_text << R"(
    %matSrc = OpCooperativeMatrixLoadKHR )" << getResTy() << R"( %in)" << variant.order << R"( %layout)" << variant.order << R"( %stride)" << variant.order;
            std::string resultVar = emitConversion();
            spirv_text << R"(
    OpCooperativeMatrixStoreKHR %out )" << resultVar << R"( %layoutRes %strideRes)";
            break;
    }
    // clang-format on
}

void ProgramGenerator::emitConversionTypes()
{
    if (op == CoopMatOp::convert)
    {
        std::vector<const Matrix *> inputs;
        variant.getInputsForOperation(inputs);
        assert(inputs.size() == 1 && "conversions must be unary");
        const Matrix *input = inputs.front();

        const cl_device_cooperative_matrix_component_type_khr inputElementType =
            input->elementType;
        const cl_device_cooperative_matrix_component_type_khr
            outputElementType = variant.output.elementType;

        const bool needsConvert = isFloatType(inputElementType)
            || isFloatType(outputElementType)
            || elementSizeOf(inputElementType)
                != elementSizeOf(outputElementType);
        if (needsConvert)
        {
            const char *elementType =
                showScalarType(variant.output.elementType);
            const char *numRows;
            const char *numCols;

            switch (variant.order)
            {
                case Variant::OperandOrder::OpA:
                    numRows = "%sizeM";
                    numCols = "%sizeK";
                    break;
                case Variant::OperandOrder::OpB:
                    numRows = "%sizeK";
                    numCols = "%sizeN";
                    break;
                case Variant::OperandOrder::OpC:
                    numRows = "%sizeM";
                    numCols = "%sizeN";
                    break;
                default:
                    assert(false && "Conversions should be unary ops");
                    std::abort();
            }
            spirv_text << R"(
    %matOutty = OpTypeCooperativeMatrixKHR %)"
                       << elementType << " %i32_subgroup " << numRows << " "
                       << numCols << " %i32_use" << variant.order << R"(
            )";
        }
    }
}

std::string ProgramGenerator::emitConversion()
{
    std::vector<const Matrix *> inputs;
    variant.getInputsForOperation(inputs);
    assert(inputs.size() == 1 && "conversions must be unary");
    const Matrix *input = inputs.front();

    const cl_device_cooperative_matrix_component_type_khr inputElementType =
        inputs.front()->elementType;
    const cl_device_cooperative_matrix_component_type_khr outputElementType =
        variant.output.elementType;

    // The OpenCL SPIR-V Environment Specification requires Signedness to be
    // equal to zero. When testing for variants that require a conversion
    // between integers of the same bit width, no code needs to be generated.
    const bool needsConvert = isFloatType(inputElementType)
        || isFloatType(outputElementType)
        || elementSizeOf(inputElementType) != elementSizeOf(outputElementType);

    // no conversion happened
    if (!needsConvert) return "%matSrc";

    std::string_view convOp;

    if (isFloatType(input->elementType))
    {
        if (isFloatType(variant.output.elementType))
        {
            // it must be a bitwidth conversion
            convOp = "OpFConvert";
        }
        else
        {
            if (isSignedType(outputElementType))
            {
                convOp = "OpConvertFToS";
            }
            else
            {
                convOp = "OpConvertFToU";
            }
        }
    }
    else
    {
        if (isFloatType(outputElementType))
        {
            if (isSignedType(input->elementType))
            {
                convOp = "OpConvertSToF";
            }
            else
            {
                convOp = "OpConvertUToF";
            }
        }
        else
        {
            // it must be a bitwidth conversion
            convOp = "OpUConvert";
        }
    }

    spirv_text << R"(
    %result = )"
               << convOp << " %matOutty %matSrc\n";
    return "%result";
}

bool ProgramGenerator::generateSpirv(Program *prog_out)
{
    // Capabilities that depend on the supported variants.
    std::set<std::string> capabilities;

    // Gather kernel argument information.
    for (size_t i = 0; i < Program::numKernelArgs; i++)
    {
        if (op == CoopMatOp::length && i == 3)
        {
            // length is different from the other operations: the result is an
            // i32 regardless of the input/output matrix types.
            prog_out->argType[i] = "i32";
        }
        else
        {
            prog_out->argType[i] =
                showScalarType(variant.getMatrix(i).elementType);
            if (prog_out->argType[i] == "i8")
            {
                capabilities.insert("Int8");
            }
            else if (prog_out->argType[i] == "i64")
            {
                capabilities.insert("Int64");
            }
            else if (prog_out->argType[i] == "f16")
            {
                capabilities.insert("Float16");
            }
            else if (prog_out->argType[i] == "f64")
            {
                capabilities.insert("Float64");
            }
        }
        prog_out->bufferSize[i] = bufferSizeOf(variant.getBufferDescriptor(i));
    }

    // Add capability for buffer element pointers (used by
    // multicomponent_load/store)
    auto checkBufferTypeCapabilities =
        [&capabilities](const BufferDescriptor &d) {
            if (d.elementType.vectorLength >= 8)
            {
                capabilities.insert("Vector16");
            }
            switch (d.elementType.scalarType)
            {
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT8_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT8_KHR:
                    capabilities.insert("Int8");
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT16_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT16_KHR:
                    capabilities.insert("Int16");
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_SINT64_KHR:
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR:
                    capabilities.insert("Int64");
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR:
                    capabilities.insert("Float16");
                    break;
                case CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP64_KHR:
                    capabilities.insert("Float64");
                    break;
                default: break;
            }
        };
    checkBufferTypeCapabilities(variant.inputADesc);
    checkBufferTypeCapabilities(variant.inputBDesc);
    checkBufferTypeCapabilities(variant.inputCDesc);
    checkBufferTypeCapabilities(variant.outputDesc);

    capabilities.insert("Addresses");
    capabilities.insert("Kernel");
    capabilities.insert("Linkage");
    capabilities.insert("CooperativeMatrixKHR");

    for (const auto &c : capabilities)
    {
        spirv_text << "    OpCapability " << c << "\n";
    }

    // clang-format off
    spirv_text << R"(
    OpExtension "SPV_KHR_cooperative_matrix"
    OpMemoryModel Physical)" << gTestContext->addrWidth << R"( OpenCL
    OpEntryPoint Kernel %fnDef "testCoopMat" %builtin_slid %builtin_sgid

    OpDecorate %builtin_slid LinkageAttributes "builtin_slid" Import
    OpDecorate %builtin_slid Constant
    OpDecorate %builtin_slid BuiltIn SubgroupLocalInvocationId

    OpDecorate %builtin_sgid LinkageAttributes "builtin_sgid" Import
    OpDecorate %builtin_sgid Constant
    OpDecorate %builtin_sgid BuiltIn SubgroupId

    %void = OpTypeVoid
)";

    genTypeDecls();

    const std::string sizetType("%i" + gTestContext->addrWidth);
    spirv_text << R"(
    %builtin_slid = OpVariable %iptr_i32 Input
    %builtin_sgid = OpVariable %iptr_i32 Input

    %sizeM = OpConstant %i32 )" << variant.inputA.nRows << R"(
    %sizeK = OpConstant %i32 )" << variant.inputA.nCols << R"(
    %sizeN = OpConstant %i32 )" << variant.inputB.nCols << R"(

    %layoutA = OpConstant %i32 )" << variant.layoutA << R"(
    %layoutB = OpConstant %i32 )" << variant.layoutB << R"(
    %layoutC = OpConstant %i32 )" << variant.layoutC << R"(
    %layoutRes = OpConstant %i32 )" << variant.layoutRes;

    spirv_text << R"(

    %strideA = OpConstant %i32 )" << variant.strideA << R"(
    %strideB = OpConstant %i32 )" << variant.strideB << R"(
    %strideC = OpConstant %i32 )" << variant.strideC << R"(
    %strideRes = OpConstant %i32 )" << variant.strideRes << R"(

    %i32_subgroup = OpConstant %i32 3
    %i32_useA = OpConstant %i32 )" << static_cast<int>(MatrixType::Use::A) << R"(
    %i32_useB = OpConstant %i32 )" << static_cast<int>(MatrixType::Use::B) << R"(
    %i32_useC = OpConstant %i32 )" << static_cast<int>(MatrixType::Use::Acc) << R"(

    %matAty = OpTypeCooperativeMatrixKHR %)" << showScalarType(variant.inputA.elementType) << R"( %i32_subgroup %sizeM %sizeK %i32_useA
    %matBty = OpTypeCooperativeMatrixKHR %)" << showScalarType(variant.inputB.elementType) << R"( %i32_subgroup %sizeK %sizeN %i32_useB
    %matCty = OpTypeCooperativeMatrixKHR %)" << showScalarType(variant.inputC.elementType) << R"( %i32_subgroup %sizeM %sizeN %i32_useC
)";

    emitConversionTypes();
    genConstants();

    spirv_text << R"(
    %fnDef = OpFunction %void None %FnTy
    %inA = OpFunctionParameter %ptr_)" << bufferTypeString(variant.inputADesc) << R"(
    %inB = OpFunctionParameter %ptr_)" << bufferTypeString(variant.inputBDesc) << R"(
    %inC = OpFunctionParameter %ptr_)" << bufferTypeString(variant.inputCDesc) << R"(
    %out = OpFunctionParameter %ptr_)" << bufferTypeString(variant.outputDesc) << R"(
    %fnEntryLabel = OpLabel)";

    genVariables();

    spirv_text << R"(
    %slid = OpLoad %i32 %builtin_slid Aligned 32
    %sgid = OpLoad %i32 %builtin_sgid Aligned 32
    %out_slid_offset = OpInBoundsPtrAccessChain %ptr_)" << bufferTypeString(variant.outputDesc) << R"( %out %slid
)";

    genBody();

    spirv_text << R"(

    OpReturn
    OpFunctionEnd)";

    // clang-format on

    auto DisMessagePrinter =
        [](spv_message_level_t, const char *, const spv_position_t &,
           const char *message) -> void { log_info("error: %s\n", message); };

    spvtools::SpirvTools SpvTool(SPV_ENV_OPENCL_2_0);
    SpvTool.SetMessageConsumer(DisMessagePrinter);

    if (!SpvTool.Assemble(spirv_text.str(), &prog_out->spirvBinary))
    {
        return false;
    }

    if (!SpvTool.Validate(prog_out->spirvBinary.data(),
                          prog_out->spirvBinary.size()))
    {
        return false;
    }

    return true;
}
