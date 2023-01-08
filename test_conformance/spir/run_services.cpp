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
#include "harness/compat.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <assert.h>
#include <string>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>

#include "exceptions.h"
#include "datagen.h"
#include "run_services.h"

#define XSTR(A) STR(A)
#define STR(A) #A

/**
 Based on the folder and the input string build the cl file nanme
 */
void get_cl_file_path (const char *folder, const char *test_name, std::string &cl_file_path)
{
    assert(folder && "folder is empty");
    assert(test_name && "test_name is empty");

    cl_file_path.append(folder);
    cl_file_path.append("/");
    cl_file_path.append(test_name);
    cl_file_path.append(".cl");
}

/**
 Based on the folder and the input string build the bc file nanme
 */
void get_bc_file_path (const char *folder, const char *test_name, std::string &bc_file_path, cl_uint size_t_width)
{
    assert(folder && "folder is empty");
    assert(test_name && "test_name is empty");
    bc_file_path.append(folder);
    bc_file_path.append("/");
    bc_file_path.append(test_name);
    if (32 == size_t_width)
        bc_file_path.append(".bc32");
    else
        bc_file_path.append(".bc64");
}

/**
 Based on the folder and the input string build the h file nanme
 */
void get_h_file_path (const char *folder, const char *file_name, std::string &h_file_path)
{
    assert(folder && "folder is empty");
    assert(file_name && "file_name is empty");

    h_file_path.assign(folder);
    h_file_path.append("/");
    h_file_path.append(file_name);
}

/**
 Fetch the kernel nanme from the test name
 */
void get_kernel_name (const char *test_name, std::string &kernel_name)
{
    char *temp_str, *p;
    std::string temp;

    temp.assign(test_name);

    // Check if the test name includes '.' -
    // the convention is that the test's kernel name is embedded in the test name up to the first '.'
    temp_str = (char *)temp.c_str();
    p = strstr(temp_str, ".");
    if (p != NULL)
    {
        *p = '\0';
    }
    kernel_name.assign(temp_str);
}

void CL_CALLBACK notify_callback(const char* errInfo, const void* privateInfo,
                                 size_t cb, void* userData);

void create_context_and_queue(cl_device_id device, cl_context *out_context, cl_command_queue *out_queue)
{
    assert( out_context && "out_context arg must be a valid pointer");
    assert( out_queue && "out_queue arg must be a valid pointer");

    int error = CL_SUCCESS;

    *out_context = clCreateContext( NULL, 1, &device, notify_callback, NULL, &error );
    if( NULL == *out_context || error != CL_SUCCESS)
    {
        throw Exceptions::TestError("clCreateContext failed\n", error);
    }

    *out_queue = clCreateCommandQueue( *out_context, device, 0, &error );
    if( NULL == *out_queue || error )
    {
        throw Exceptions::TestError("clCreateCommandQueue failed\n", error);
    }
}

/**
 Loads the kernel text from the given text file
 */
std::string load_file_cl( const std::string& file_name)
{
    std::ifstream ifs(file_name.c_str());
    if( !ifs.good() )
        throw Exceptions::TestError("Can't load the cl File " + file_name, 1);
    std::string str( ( std::istreambuf_iterator<char>( ifs ) ), std::istreambuf_iterator<char>());
    return str;
}

/**
 Loads the kernel IR from the given binary file in SPIR BC format
 */
void* load_file_bc( const std::string& file_name, size_t *binary_size)
{
    assert(binary_size && "binary_size arg should be valid");

    std::ifstream file(file_name.c_str(), std::ios::binary);

    if( !file.good() )
    {
        throw Exceptions::TestError("Can't load the bc File " + file_name, 1);
    }

    file.seekg(0, std::ios::end);
    *binary_size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);

    void* buffer = malloc(*binary_size);
    file.read((char*)buffer, *binary_size);
    file.close();

    return buffer;
}

/**
 Create program from the CL source file
 */
cl_program create_program_from_cl(cl_context context, const std::string& file_name)
{
    std::string text_file  = load_file_cl(file_name);
    const char* text_str = text_file.c_str();
    int error  = CL_SUCCESS;

    cl_program program = clCreateProgramWithSource( context, 1, &text_str, NULL, &error );
    if( program == NULL || error != CL_SUCCESS)
    {
        throw Exceptions::TestError("Error creating program\n", error);
    }

    return program;
}

/**
 Create program from the BC source file
 */
cl_program create_program_from_bc (cl_context context, const std::string& file_name)
{
    cl_int load_error = CL_SUCCESS;
    cl_int error;
    size_t binary_size;
    BufferOwningPtr<const unsigned char> binary(load_file_bc(file_name, &binary_size));
    const unsigned char* ptr = binary;

    cl_device_id device = get_context_device(context);
    cl_program program = clCreateProgramWithBinary( context, 1, &device, &binary_size, &ptr, &load_error, &error );


    if( program == NULL || error != CL_SUCCESS )
    {
        throw Exceptions::TestError("clCreateProgramWithBinary failed: Unable to load valid program binary\n", error);
    }

    if( load_error != CL_SUCCESS )
    {
         throw Exceptions::TestError("clCreateProgramWithBinary failed: Unable to load valid device binary into program\n", load_error);
    }

    return program;
}

/**
 Creates the kernel with the given name from the given program.
 */
cl_kernel create_kernel_helper( cl_program program, const std::string& kernel_name )
{
    int error = CL_SUCCESS;
    cl_kernel kernel = NULL;
    /* And create a kernel from it */
    kernel = clCreateKernel( program, kernel_name.c_str(), &error );
    if( kernel == NULL || error != CL_SUCCESS)
        throw Exceptions::TestError("Unable to create kernel\n", error);
    return kernel;
}

cl_device_id get_context_device (cl_context context)
{
    cl_device_id device[1];

    int error = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device), device, NULL);
    if( error != CL_SUCCESS )
    {
        throw Exceptions::TestError("clGetContextInfo failed\n", error);
    }

    return device[0];
}

cl_device_id get_program_device (cl_program program)
{
    cl_device_id device[1];

    int error = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(device), device, NULL);
    if( error != CL_SUCCESS )
    {
        throw Exceptions::TestError("clGetProgramInfo failed\n", error);
    }

    return device[0];
}

void generate_kernel_ws( cl_device_id device, cl_kernel kernel, WorkSizeInfo& ws)
{
    size_t compile_work_group_size[MAX_WORK_DIM];

    memset(&ws, 0, sizeof(WorkSizeInfo));
    ws.work_dim = 1;
    ws.global_work_size[0] = (GLOBAL_WORK_SIZE <= 32) ? GLOBAL_WORK_SIZE : 32;        // kernels limitations
    ws.local_work_size[0] = ((GLOBAL_WORK_SIZE % 4) == 0) ? (GLOBAL_WORK_SIZE / 4) : (GLOBAL_WORK_SIZE / 2);

    //Check if the kernel was compiled with specific work group size
    int error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(compile_work_group_size), &compile_work_group_size, NULL);
    if( error != CL_SUCCESS )
    {
        throw Exceptions::TestError("clGetKernelWorkGroupInfo failed\n", error);
    }

    // if compile_work_group_size[0] is not 0 - use the compiled values
    if ( 0 != compile_work_group_size[0] )
    {
        // the kernel compiled with __attribute__((reqd_work_group_size(X, Y, Z)))
        memcpy(ws.global_work_size, compile_work_group_size, sizeof(ws.global_work_size));

        // Now, check the correctness of the local work size and fix it if necessary
        for ( int i = 0; i < MAX_WORK_DIM; ++i )
        {
            if ( ws.local_work_size[i] > compile_work_group_size[i] )
            {
                ws.local_work_size[i] = compile_work_group_size[i];
            }
        }
    }
}

TestResult* TestResult::clone(cl_context ctx, const WorkSizeInfo& ws, const cl_kernel kernel, const cl_device_id device) const
{
    TestResult *cpy = new TestResult();

    for (size_t i=0; i<m_kernelArgs.getArgCount(); ++i)
        cpy->m_kernelArgs.addArg(m_kernelArgs.getArg(i)->clone(ctx, ws, kernel, device));

    return cpy;
}

/*
 * class DataRow
 */

const std::string& DataRow::operator[](int column)const
{
    assert((column > -1 && (size_t)column < m_row.size()) && "Index out of bound");
    return m_row[column];
}

std::string& DataRow::operator[](int column)
{
    assert((column > -1 && (size_t)column <= m_row.size())
           && "Index out of bound");
    if ((size_t)column == m_row.size()) m_row.push_back("");

    return m_row[column];
}

/*
 * class DataTable
 */

size_t DataTable::getNumRows() const
{
    return m_rows.size();
}

void DataTable::addTableRow(DataRow *dr)
{
    m_rows.push_back(dr);
}

const DataRow& DataTable::operator[](int index)const
{
    assert((index > -1 && (size_t)index < m_rows.size()) && "Index out of bound");
    return *m_rows[index];
}

DataRow& DataTable::operator[](int index)
{
    assert((index > -1 && (size_t)index < m_rows.size()) && "Index out of bound");
    return *m_rows[index];
}

/*
 * class OclExtensions
 */
OclExtensions OclExtensions::getDeviceCapabilities(cl_device_id devId)
{
    size_t size;
    size_t set_size;
    cl_int errcode = clGetDeviceInfo(devId, CL_DEVICE_EXTENSIONS, 0, NULL, &set_size);
    if (errcode)
        throw Exceptions::TestError("Device query failed");
    // Querying the device for its supported extensions
    std::vector<char> extensions(set_size);
    errcode = clGetDeviceInfo(devId,
                              CL_DEVICE_EXTENSIONS,
                              extensions.size(),
                              extensions.data(),
                              &size);

    if (errcode)
        throw Exceptions::TestError("Device query failed");

    char device_profile[1024] = {0};
    errcode = clGetDeviceInfo(devId,
                              CL_DEVICE_PROFILE,
                              sizeof(device_profile),
                              device_profile,
                              NULL);
    if (errcode)
        throw Exceptions::TestError("Device query failed");

    OclExtensions ret = OclExtensions::empty();
    assert(size == set_size);
    if (!size)
      return ret;

    // Iterate over the extensions, and convert them into the bit field.
    std::list<std::string> extVector;
    std::stringstream khrStream(extensions.data());
    std::copy(std::istream_iterator<std::string>(khrStream),
              std::istream_iterator<std::string>(),
              std::back_inserter(extVector));

    // full_profile devices supports embedded profile as core feature
    if ( std::string( device_profile ) == "FULL_PROFILE" ) {
        extVector.push_back("cles_khr_int64");
        extVector.push_back("cles_khr_2d_image_array_writes");
    }

    for(std::list<std::string>::const_iterator it = extVector.begin(),
                                               e = extVector.end(); it != e;
                                               it++)
    {
        ret = ret | OclExtensions::fromString(*it);
    }

    return ret;
}

OclExtensions OclExtensions::empty()
{
    return OclExtensions(0);
}

OclExtensions OclExtensions::fromString(const std::string& e)
{
    std::string s = "OclExtensions::has_" + e;
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_int64_base_atomics);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_int64_extended_atomics);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_3d_image_writes);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_fp16);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_gl_sharing);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_gl_event);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_d3d10_sharing);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_dx9_media_sharing);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_d3d11_sharing);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_depth_images);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_gl_depth_images);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_gl_msaa_sharing);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_image2d_from_buffer);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_initialize_memory);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_spir);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_fp64);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_global_int32_base_atomics);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_global_int32_extended_atomics);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_local_int32_base_atomics);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_local_int32_extended_atomics);
    RETURN_IF_ENUM(s, OclExtensions::has_cl_khr_byte_addressable_store);
    RETURN_IF_ENUM(s, OclExtensions::has_cles_khr_int64);
    RETURN_IF_ENUM(s, OclExtensions::has_cles_khr_2d_image_array_writes);
    // Unknown KHR string.
    return OclExtensions::empty();
}

std::string OclExtensions::toString()
{
#define APPEND_STR_IF_SUPPORTS(STR, E)                                         \
    if (this->supports(E))                                                     \
    {                                                                          \
        std::string ext_str(#E);                                               \
        std::string prefix = "OclExtensions::has_";                            \
        size_t pos = ext_str.find(prefix);                                     \
        if (pos != std::string::npos)                                          \
        {                                                                      \
            ext_str.replace(pos, prefix.length(), "");                         \
        }                                                                      \
        STR += ext_str;                                                        \
        STR += " ";                                                            \
    }

    std::string s = "";

    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_int64_base_atomics);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_int64_extended_atomics);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_3d_image_writes);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_fp16);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_gl_sharing);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_gl_event);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_d3d10_sharing);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_dx9_media_sharing);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_d3d11_sharing);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_depth_images);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_gl_depth_images);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_gl_msaa_sharing);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_image2d_from_buffer);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_initialize_memory);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_spir);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_fp64);
    APPEND_STR_IF_SUPPORTS(s,
                           OclExtensions::has_cl_khr_global_int32_base_atomics);
    APPEND_STR_IF_SUPPORTS(
        s, OclExtensions::has_cl_khr_global_int32_extended_atomics);
    APPEND_STR_IF_SUPPORTS(s,
                           OclExtensions::has_cl_khr_local_int32_base_atomics);
    APPEND_STR_IF_SUPPORTS(
        s, OclExtensions::has_cl_khr_local_int32_extended_atomics);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cl_khr_byte_addressable_store);
    APPEND_STR_IF_SUPPORTS(s, OclExtensions::has_cles_khr_int64);
    APPEND_STR_IF_SUPPORTS(s,
                           OclExtensions::has_cles_khr_2d_image_array_writes);

    return s;
}

std::ostream& operator<<(std::ostream& os, OclExtensions ext)
{
    return os << ext.toString();
}

OclExtensions OclExtensions::operator|(const OclExtensions& b) const
{
    return OclExtensions(m_extVector | b.m_extVector);
}

bool OclExtensions::supports(const OclExtensions& b) const
{
    return ((b.m_extVector & m_extVector) == b.m_extVector);
}

OclExtensions OclExtensions::get_missing(const OclExtensions& b) const
{
    return OclExtensions( b.m_extVector & ( ~ m_extVector ) );
}

/*
 * class KhrSupport
 */

KhrSupport *KhrSupport::m_instance = NULL;

const KhrSupport* KhrSupport::get(const std::string& path)
{
    if(m_instance)
        return m_instance;

    m_instance = new KhrSupport();
    // First invokation, parse the file into memory.
    std::fstream csv(path.c_str(), std::ios_base::in);
    if (!csv.is_open())
    {
        delete m_instance;
        std::string msg;
        msg.append("File ");
        msg.append(path);
        msg.append(" cannot be opened");
        throw Exceptions::TestError(msg.c_str());
    }

    m_instance->parseCSV(csv);
    csv.close();
    return m_instance;
}

void KhrSupport::parseCSV(std::fstream& f)
{
    assert(f.is_open() && "file is not in reading state.") ;
    char line[1024];
    while (!f.getline(line, sizeof(line)).eof())
    {
        DataRow *dr = parseLine(std::string(line));
        m_dt.addTableRow(dr);
    }
}

DataRow* KhrSupport::parseLine(const std::string& line)
{
    const char DELIM = ',';
    std::string token;
    DataRow *dr = new DataRow();
    int tIndex = 0;

    for(std::string::const_iterator it = line.begin(), e = line.end(); it != e;
        it++)
    {
        // Eat those characters away.
        if(isspace(*it) || '"' == *it)
            continue;

        // If that's a delimiter, we need to tokenize the collected value.
        if(*it == DELIM)
        {
            (*dr)[tIndex++] = token;
            token.clear();
            continue;
        }

        // Append to current token.
        token.append(1U, *it);
    }
    if (!token.empty())
        (*dr)[tIndex] = token;

    assert(tIndex && "empty data row??");
    return dr;
}

OclExtensions KhrSupport::getRequiredExtensions(const char* suite, const char* test) const
{
    OclExtensions ret = OclExtensions::empty();

    const std::string strSuite(suite), strTest(test);
    // Iterating on the DataTable, searching whether the row with th requested
    // row exists.
    for(size_t rowIndex = 0; rowIndex < m_dt.getNumRows(); rowIndex++)
    {
        const DataRow& dr = m_dt[rowIndex];
        const std::string csvSuite = dr[SUITE_INDEX], csvTest = dr[TEST_INDEX];
        bool sameSuite = (csvSuite == strSuite), sameTest = (csvTest == strTest)||(csvTest == "*");
        if (sameTest && sameSuite)
        {
            ret = ret | OclExtensions::fromString(dr[EXT_INDEX]);
        }
    }

    return ret;
}

cl_bool KhrSupport::isImagesRequired(const char* suite, const char* test) const
{
    cl_bool ret = CL_FALSE;
    const std::string strSuite(suite), strTest(test);

    // Iterating on the DataTable, searching whether the row with th requested
    // row exists.
    for(size_t rowIndex = 0; rowIndex < m_dt.getNumRows(); rowIndex++)
    {
        const DataRow& dr = m_dt[rowIndex];
        const std::string csvSuite = dr[SUITE_INDEX], csvTest = dr[TEST_INDEX];
        bool sameSuite = (csvSuite == strSuite), sameTest = (csvTest == strTest)||(csvTest == "*");
        if (sameTest && sameSuite)
        {
            ret = (dr[IMAGES_INDEX] == "CL_TRUE") ? CL_TRUE : CL_FALSE;
            break;
        }
    }

    return ret;
}

cl_bool KhrSupport::isImages3DRequired(const char* suite, const char* test) const
{
    cl_bool ret = CL_FALSE;
    const std::string strSuite(suite), strTest(test);

    // Iterating on the DataTable, searching whether the row with th requested
    // row exists.
    for(size_t rowIndex = 0; rowIndex < m_dt.getNumRows(); rowIndex++)
    {
        const DataRow& dr = m_dt[rowIndex];
        const std::string csvSuite = dr[SUITE_INDEX], csvTest = dr[TEST_INDEX];
        bool sameSuite = (csvSuite == strSuite), sameTest = (csvTest == strTest)||(csvTest == "*");
        if (sameTest && sameSuite)
        {
            ret = (dr[IMAGES_3D_INDEX] == "CL_TRUE") ? CL_TRUE : CL_FALSE;
            break;
        }
    }

    return ret;
}


static void generate_kernel_args(cl_context context, cl_kernel kernel, const WorkSizeInfo& ws, KernelArgs& cl_args, const cl_device_id device)
{
    int error = CL_SUCCESS;
    cl_uint num_args = 0;
    KernelArg* cl_arg = NULL;
    DataGenerator* dg = DataGenerator::getInstance();

    error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, sizeof( num_args ), &num_args, NULL );
    if( error != CL_SUCCESS )
    {
        throw Exceptions::TestError("Unable to get kernel arg count\n", error);
    }

    for ( cl_uint j = 0; j < num_args; ++j )
    {
        KernelArgInfo kernel_arg_info;
        size_t size;
        const int max_name_len = 512;
        char name[max_name_len];

        // Try to get the address qualifier of each argument.
        error = clGetKernelArgInfo( kernel, j, CL_KERNEL_ARG_ADDRESS_QUALIFIER, sizeof(cl_kernel_arg_address_qualifier), kernel_arg_info.getAddressQualifierRef(), &size);
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("Unable to get argument address qualifier\n", error);
        }

        // Try to get the access qualifier of each argument.
        error = clGetKernelArgInfo( kernel, j, CL_KERNEL_ARG_ACCESS_QUALIFIER, sizeof(cl_kernel_arg_access_qualifier), kernel_arg_info.getAccessQualifierRef(), &size );
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("Unable to get argument access qualifier\n", error);
        }

        // Try to get the type qualifier of each argument.
        error = clGetKernelArgInfo( kernel, j, CL_KERNEL_ARG_TYPE_QUALIFIER, sizeof(cl_kernel_arg_type_qualifier), kernel_arg_info.getTypeQualifierRef(), &size );
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("Unable to get argument type qualifier\n", error);
        }

        // Try to get the type of each argument.
        memset( name, 0, max_name_len );
        error = clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_TYPE_NAME, max_name_len, name, NULL );
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("Unable to get argument type name\n", error);
        }
        kernel_arg_info.setTypeName(name);

        // Try to get the name of each argument.
        memset( name, 0, max_name_len );
        error = clGetKernelArgInfo( kernel, j, CL_KERNEL_ARG_NAME, max_name_len, name, NULL );
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("Unable to get argument name\n", error);
        }
        kernel_arg_info.setName(name);

        cl_arg = dg->generateKernelArg(context, kernel_arg_info, ws, NULL, kernel, device);
        cl_args.addArg( cl_arg );
    }
}

void set_kernel_args( cl_kernel kernel, KernelArgs& args)
{
    int error = CL_SUCCESS;
    for( size_t i = 0;  i < args.getArgCount(); ++ i )
    {
        error = clSetKernelArg( kernel, i, args.getArg(i)->getArgSize(), args.getArg(i)->getArgValue());
        if( error != CL_SUCCESS )
        {
            throw Exceptions::TestError("clSetKernelArg failed\n", error);
        }
    }
}

/**
 Run the single kernel
 */
void generate_kernel_data ( cl_context context, cl_kernel kernel, WorkSizeInfo &ws, TestResult& results)
{
    cl_device_id device = get_context_device(context);
    generate_kernel_ws( device, kernel, ws);
    generate_kernel_args(context, kernel, ws, results.kernelArgs(), device);
}

/**
 Run the single kernel
 */
void run_kernel( cl_kernel kernel, cl_command_queue queue, WorkSizeInfo &ws, TestResult& result )
{
    clEventWrapper execute_event;

    set_kernel_args(kernel, result.kernelArgs());

    int error = clEnqueueNDRangeKernel( queue, kernel, ws.work_dim, ws.global_work_offset, ws.global_work_size, ws.local_work_size, 0, NULL, &execute_event );
    if( error != CL_SUCCESS )
    {
        throw Exceptions::TestError("clEnqueueNDRangeKernel failed\n", error);
    }

    error = clWaitForEvents( 1, &execute_event );
    if( error != CL_SUCCESS )
    {
        throw Exceptions::TestError("clWaitForEvents failed\n", error);
    }

    // read all the buffers back to host
    result.readToHost(queue);
}

/**
 Compare two test results
 */
bool compare_results( const TestResult& lhs, const TestResult& rhs, float ulps )
{
    if( lhs.kernelArgs().getArgCount() != rhs.kernelArgs().getArgCount() )
    {
        log_error("number of kernel parameters differ between SPIR and CL version of the kernel\n");
        return false;
    }

    for( size_t i = 0 ; i < lhs.kernelArgs().getArgCount(); ++i )
    {
        if( ! lhs.kernelArgs().getArg(i)->compare( *rhs.kernelArgs().getArg(i), ulps ) )
        {
            log_error("the kernel parameter (%d) is different between SPIR and CL version of the kernel\n", i);
            return false;
        }
    }
    return true;
}

