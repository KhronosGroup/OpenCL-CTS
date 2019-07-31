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
#include "../../test_common/harness/compat.h"

// Bug: Missing in spec: atomic_intptr_t is always supported if device is 32-bits.
// Bug: Missing in spec: CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE

#define FLUSH fflush(stdout)

#define MAX_STR 16*1024

#define ALIGNMENT 128

#define OPTIONS "-cl-std=CL2.0"

// NUM_ROUNDS must be at least 1.
// It determines how many sets of random data we push through the global
// variables.
#define NUM_ROUNDS 1

// This is a shared property of the writer and reader kernels.
#define NUM_TESTED_VALUES 5

// TODO: pointer-to-half (and its vectors)
// TODO: union of...

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include "../../test_common/harness/typeWrappers.h"
#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/mt19937.h"
#include "procs.h"


////////////////////
// Device capabilities
static int l_has_double = 0;
static int l_has_half = 0;
static int l_64bit_device = 0;
static int l_has_int64_atomics = 0;
static int l_has_intptr_atomics = 0;
static int l_has_cles_int64 = 0;

static int l_host_is_big_endian = 1;

static size_t l_max_global_id0 = 0;
static cl_bool l_linker_available = false;

#define check_error(errCode,msg,...) ((errCode != CL_SUCCESS) ? (log_error("ERROR: " msg "! (%s:%d)\n", ## __VA_ARGS__, __FILE__, __LINE__), 1) : 0)

////////////////////
// Info about types we can use for program scope variables.


class TypeInfo {

public:
    TypeInfo() :
        name(""),
        m_buf_elem_type(""),
        m_is_vecbase(false),
        m_is_atomic(false),
        m_is_like_size_t(false),
        m_is_bool(false),
        m_elem_type(0), m_num_elem(0),
        m_size(0),
        m_value_size(0)
        {}
    TypeInfo(const char* name_arg) :
        name(name_arg),
        m_buf_elem_type(name_arg),
        m_is_vecbase(false),
        m_is_atomic(false),
        m_is_like_size_t(false),
        m_is_bool(false),
        m_elem_type(0), m_num_elem(0),
        m_size(0),
        m_value_size(0)
        { }

    // Vectors
    TypeInfo( TypeInfo* elem_type, int num_elem ) :
        m_is_vecbase(false),
        m_is_atomic(false),
        m_is_like_size_t(false),
        m_is_bool(false),
        m_elem_type(elem_type),
        m_num_elem(num_elem)
    {
        char the_name[10]; // long enough for longest vector type name "double16"
        snprintf(the_name,sizeof(the_name),"%s%d",elem_type->get_name_c_str(),m_num_elem);
        this->name = std::string(the_name);
        this->m_buf_elem_type = std::string(the_name);
        this->m_value_size = num_elem * elem_type->get_size();
        if ( m_num_elem == 3 ) {
            this->m_size = 4 * elem_type->get_size();
        } else {
            this->m_size = num_elem * elem_type->get_size();
        }
    }
    const std::string& get_name(void) const { return name; }
    const char* get_name_c_str(void) const { return name.c_str(); }
    TypeInfo& set_vecbase(void) { this->m_is_vecbase = true; return *this; }
    TypeInfo& set_atomic(void) { this->m_is_atomic = true; return *this; }
    TypeInfo& set_like_size_t(void) {
        this->m_is_like_size_t = true;
        this->set_size( l_64bit_device ? 8 : 4 );
        this->m_buf_elem_type = l_64bit_device ? "ulong" : "uint";
        return *this;
    }
    TypeInfo& set_bool(void) { this->m_is_bool = true; return *this; }
    TypeInfo& set_size(size_t n) { this->m_value_size = this->m_size = n; return *this; }
    TypeInfo& set_buf_elem_type( const char* name ) { this->m_buf_elem_type = std::string(name); return *this; }

    const TypeInfo* elem_type(void) const { return m_elem_type; }
    int num_elem(void) const { return m_num_elem; }

    bool is_vecbase(void) const {return m_is_vecbase;}
    bool is_atomic(void) const {return m_is_atomic;}
    bool is_atomic_64bit(void) const {return m_is_atomic && m_size == 8;}
    bool is_like_size_t(void) const {return m_is_like_size_t;}
    bool is_bool(void) const {return m_is_bool;}
    size_t get_size(void) const {return m_size;}
    size_t get_value_size(void) const {return m_value_size;}

    // When passing values of this type to a kernel, what buffer type
    // should be used?
    const char* get_buf_elem_type(void) const { return m_buf_elem_type.c_str(); }

    std::string as_string(const cl_uchar* value_ptr) const {
        // This method would be shorter if I had a real handle to element
        // vector type.
        if ( this->is_bool() ) {
            std::string result( name );
            result += "<";
            result += (*value_ptr ? "true" : "false");
            result += ", ";
            char buf[10];
            sprintf(buf,"%02x",*value_ptr);
            result += buf;
            result += ">";
            return result;
        } else if ( this->num_elem() ) {
            std::string result( name );
            result += "<";
            for ( unsigned ielem = 0 ; ielem < this->num_elem() ; ielem++ ) {
                char buf[MAX_STR];
                if ( ielem ) result += ", ";
                for ( unsigned ibyte = 0; ibyte < this->m_elem_type->get_size() ; ibyte++ ) {
                    sprintf(buf + 2*ibyte,"%02x", value_ptr[ ielem * this->m_elem_type->get_size() + ibyte ] );
                }
                result += buf;
            }
            result += ">";
            return result;
        } else {
            std::string result( name );
            result += "<";
            char buf[MAX_STR];
            for ( unsigned ibyte = 0; ibyte < this->get_size() ; ibyte++ ) {
                sprintf(buf + 2*ibyte,"%02x", value_ptr[ ibyte ] );
            }
            result += buf;
            result += ">";
            return result;
        }
    }

    // Initialize the given buffer to a constant value initialized as if it
    // were from the INIT_VAR macro below.
    // Only needs to support values 0 and 1.
    void init( cl_uchar* buf, cl_uchar val) const {
        if ( this->num_elem() ) {
            for ( unsigned ielem = 0 ; ielem < this->num_elem() ; ielem++ ) {
                // Delegate!
                this->init_elem( buf + ielem * this->get_value_size()/this->num_elem(), val );
            }
        } else {
            init_elem( buf, val );
        }
    }

private:
    void init_elem( cl_uchar* buf, cl_uchar val ) const {
        size_t elem_size = this->num_elem() ? this->get_value_size()/this->num_elem() : this->get_size();
        memset(buf,0,elem_size);
        if ( val ) {
            if ( strstr( name.c_str(), "float" ) ) {
                *(float*)buf = (float)val;
                return;
            }
            if ( strstr( name.c_str(), "double" ) ) {
                *(double*)buf = (double)val;
                return;
            }
            if ( this->is_bool() ) { *buf = (bool)val; return; }

            // Write a single character value to the correct spot,
            // depending on host endianness.
            if ( l_host_is_big_endian ) *(buf + elem_size-1) = (cl_uchar)val;
            else *buf = (cl_uchar)val;
        }
    }
public:

    void dump(FILE* fp) const {
        fprintf(fp,"Type %s : <%d,%d,%s> ", name.c_str(),
                (int)m_size,
                (int)m_value_size,
                m_buf_elem_type.c_str() );
        if ( this->m_elem_type ) fprintf(fp, " vec(%s,%d)", this->m_elem_type->get_name_c_str(), this->num_elem() );
        if ( this->m_is_vecbase ) fprintf(fp, " vecbase");
        if ( this->m_is_bool ) fprintf(fp, " bool");
        if ( this->m_is_like_size_t ) fprintf(fp, " like-size_t");
        if ( this->m_is_atomic ) fprintf(fp, " atomic");
        fprintf(fp,"\n");
        fflush(fp);
    }

private:
    std::string name;
    TypeInfo* m_elem_type;
    int m_num_elem;
    bool m_is_vecbase;
    bool m_is_atomic;
    bool m_is_like_size_t;
    bool m_is_bool;
    size_t m_size; // Number of bytes of storage occupied by this type.
    size_t m_value_size; // Number of bytes of value significant for this type. Differs for vec3.

    // When passing values of this type to a kernel, what buffer type
    // should be used?
    // For most types, it's just itself.
    // Use a std::string so I don't have to make a copy constructor.
    std::string m_buf_elem_type;
};


#define NUM_SCALAR_TYPES (8+2) // signed and unsigned integral types, float and double
#define NUM_VECTOR_SIZES (5)   // 2,3,4,8,16
#define NUM_PLAIN_TYPES \
      5 /*boolean and size_t family */  \
    + NUM_SCALAR_TYPES \
    + NUM_SCALAR_TYPES*NUM_VECTOR_SIZES \
    + 10 /* atomic types */

// Need room for plain, array, pointer, struct
#define MAX_TYPES (4*NUM_PLAIN_TYPES)

static TypeInfo type_info[MAX_TYPES];
static int num_type_info = 0; // Number of valid entries in type_info[]




// A helper class to form kernel source arguments for clCreateProgramWithSource.
class StringTable {
public:
    StringTable() : m_c_strs(NULL), m_lengths(NULL), m_frozen(false), m_strings() {}
    ~StringTable() { release_frozen(); }

    void add(std::string s) { release_frozen(); m_strings.push_back(s); }

    const size_t num_str() { freeze(); return m_strings.size(); }
    const char** strs() { freeze(); return m_c_strs; }
    const size_t* lengths() { freeze(); return m_lengths; }

private:
    void freeze(void) {
        if ( !m_frozen ) {
            release_frozen();

            m_c_strs = (const char**) malloc(sizeof(const char*) * m_strings.size());
            m_lengths = (size_t*) malloc(sizeof(size_t) * m_strings.size());
            assert( m_c_strs );
            assert( m_lengths );

            for ( size_t i = 0; i < m_strings.size() ; i++ ) {
                m_c_strs[i] = m_strings[i].c_str();
                m_lengths[i] = strlen(m_c_strs[i]);
            }

            m_frozen = true;
        }
    }
    void release_frozen(void) {
        if ( m_c_strs ) { free(m_c_strs); m_c_strs = 0; }
        if ( m_lengths ) { free(m_lengths); m_lengths = 0; }
        m_frozen = false;
    }

    typedef std::vector<std::string> strlist_t;
    strlist_t m_strings;
    const char** m_c_strs;
    size_t* m_lengths;
    bool m_frozen;
};


////////////////////
// File scope function declarations

static void l_load_abilities(cl_device_id device);
static const char* l_get_fp64_pragma(void);
static const char* l_get_cles_int64_pragma(void);
static int l_build_type_table(cl_device_id device);

static int l_get_device_info(cl_device_id device, size_t* max_size_ret, size_t* pref_size_ret);

static void l_set_randomly( cl_uchar* buf, size_t buf_size, RandomSeed& rand_state );
static int l_compare( const cl_uchar* expected, const cl_uchar* received, unsigned num_values, const TypeInfo&ti );
static int l_copy( cl_uchar* dest, unsigned dest_idx, const cl_uchar* src, unsigned src_idx, const TypeInfo&ti );

static std::string conversion_functions(const TypeInfo& ti);
static std::string global_decls(const TypeInfo& ti, bool with_init);
static std::string writer_function(const TypeInfo& ti);
static std::string reader_function(const TypeInfo& ti);

static int l_write_read( cl_device_id device, cl_context context, cl_command_queue queue );
static int l_write_read_for_type( cl_device_id device, cl_context context, cl_command_queue queue, const TypeInfo& ti, RandomSeed& rand_state );

static int l_init_write_read( cl_device_id device, cl_context context, cl_command_queue queue );
static int l_init_write_read_for_type( cl_device_id device, cl_context context, cl_command_queue queue, const TypeInfo& ti, RandomSeed& rand_state );

static int l_capacity( cl_device_id device, cl_context context, cl_command_queue queue, size_t max_size );
static int l_user_type( cl_device_id device, cl_context context, cl_command_queue queue, size_t max_size, bool separate_compilation );



////////////////////
// File scope function definitions

static cl_int print_build_log(cl_program program, cl_uint num_devices, cl_device_id *device_list, cl_uint count, const char **strings, const size_t *lengths, const char* options)
{
    cl_uint i;
    cl_int error;
    BufferOwningPtr<cl_device_id> devices;

    if(num_devices == 0 || device_list == NULL)
    {
        error = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
        test_error(error, "clGetProgramInfo CL_PROGRAM_NUM_DEVICES failed");

        device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
        devices.reset(device_list);

        memset(device_list, 0, sizeof(cl_device_id) * num_devices);

        error = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * num_devices, device_list, NULL);
        test_error(error, "clGetProgramInfo CL_PROGRAM_DEVICES failed");
    }

    cl_uint z;
    bool sourcePrinted = false;

    for(z = 0; z < num_devices; z++)
    {
        char deviceName[4096] = "";
        error = clGetDeviceInfo(device_list[z], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        check_error(error, "Device \"%d\" failed to return a name. clGetDeviceInfo CL_DEVICE_NAME failed", z);

        cl_build_status buildStatus;
        error = clGetProgramBuildInfo(program, device_list[z], CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, NULL);
        check_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_STATUS failed");

        if(buildStatus != CL_BUILD_SUCCESS)
        {
            if(!sourcePrinted)
            {
                log_error("Build options: %s\n", options);
                if(count && strings)
                {
                    log_error("Original source is: ------------\n");
                    for(i = 0; i < count; i++) log_error("%s", strings[i]);
                }
                sourcePrinted = true;
            }

            char statusString[64] = "";
            if (buildStatus == (cl_build_status)CL_BUILD_SUCCESS)
              sprintf(statusString, "CL_BUILD_SUCCESS");
            else if (buildStatus == (cl_build_status)CL_BUILD_NONE)
              sprintf(statusString, "CL_BUILD_NONE");
            else if (buildStatus == (cl_build_status)CL_BUILD_ERROR)
              sprintf(statusString, "CL_BUILD_ERROR");
            else if (buildStatus == (cl_build_status)CL_BUILD_IN_PROGRESS)
              sprintf(statusString, "CL_BUILD_IN_PROGRESS");
            else
              sprintf(statusString, "UNKNOWN (%d)", buildStatus);

            log_error("Build not successful for device \"%s\", status: %s\n", deviceName, statusString);

            size_t paramSize = 0;
            error = clGetProgramBuildInfo(program, device_list[z], CL_PROGRAM_BUILD_LOG, 0, NULL, &paramSize);
            if(check_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed")) break;

            std::string log;
            log.resize(paramSize/sizeof(char));

            error = clGetProgramBuildInfo(program, device_list[z], CL_PROGRAM_BUILD_LOG, paramSize, &log[0], NULL);
            if(check_error(error, "Device %d (%s) failed to return a build log", z, deviceName)) break;
            if(log[0] == 0) log_error("clGetProgramBuildInfo returned an empty log.\n");
            else
            {
                log_error("Build log:\n", deviceName);
                log_error("%s\n", log.c_str());
            }
        }
    }
    return error;
}

static void l_load_abilities(cl_device_id device)
{
    l_has_half       = is_extension_available(device,"cl_khr_fp16");
    l_has_double     = is_extension_available(device,"cl_khr_fp64");
    l_has_cles_int64 = is_extension_available(device,"cles_khr_int64");

    l_has_int64_atomics
    =  is_extension_available(device,"cl_khr_int64_base_atomics")
    && is_extension_available(device,"cl_khr_int64_extended_atomics");

    {
        int status = CL_SUCCESS;
        cl_uint addr_bits = 32;
        status = clGetDeviceInfo(device,CL_DEVICE_ADDRESS_BITS,sizeof(addr_bits),&addr_bits,0);
        l_64bit_device = ( status == CL_SUCCESS && addr_bits == 64 );
    }

    // 32-bit devices always have intptr atomics.
    l_has_intptr_atomics = !l_64bit_device || l_has_int64_atomics;

    union { char c[4]; int i; } probe;
    probe.i = 1;
    l_host_is_big_endian = !probe.c[0];

    // Determine max global id.
    {
        int status = CL_SUCCESS;
        cl_uint max_dim = 0;
        status = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(max_dim),&max_dim,0);
        assert( status == CL_SUCCESS );
        assert( max_dim > 0 );
        size_t max_id[3];
        max_id[0] = 0;
    status = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_ITEM_SIZES,max_dim*sizeof(size_t),&max_id[0],0);
        assert( status == CL_SUCCESS );
        l_max_global_id0 = max_id[0];
    }

    { // Is separate compilation supported?
        int status = CL_SUCCESS;
        l_linker_available = false;
        status = clGetDeviceInfo(device,CL_DEVICE_LINKER_AVAILABLE,sizeof(l_linker_available),&l_linker_available,0);
        assert( status == CL_SUCCESS );
    }
}


static const char* l_get_fp64_pragma(void)
{
    return l_has_double ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" : "";
}

static const char* l_get_cles_int64_pragma(void)
{
    return l_has_cles_int64 ? "#pragma OPENCL EXTENSION cles_khr_int64 : enable\n" : "";
}

static const char* l_get_int64_atomic_pragma(void)
{
    return "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
           "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n";
}

static int l_build_type_table(cl_device_id device)
{
    int status = CL_SUCCESS;
    size_t iscalar = 0;
    size_t ivecsize = 0;
    int vecsizes[] = { 2, 3, 4, 8, 16 };
    const char* vecbase[] = {
        "uchar", "char",
        "ushort", "short",
        "uint", "int",
        "ulong", "long",
        "float",
        "double"
    };
    int vecbase_size[] = {
        1, 1,
        2, 2,
        4, 4,
        8, 8,
        4,
        8
    };
    const char* like_size_t[] = {
        "intptr_t",
        "uintptr_t",
        "size_t",
        "ptrdiff_t"
    };
    const char* atomics[] = {
        "atomic_int", "atomic_uint",
        "atomic_long", "atomic_ulong",
        "atomic_float",
        "atomic_double",
    };
    int atomics_size[] = {
        4, 4,
        8, 8,
        4,
        8
    };
    const char* intptr_atomics[] = {
        "atomic_intptr_t",
        "atomic_uintptr_t",
        "atomic_size_t",
        "atomic_ptrdiff_t"
    };

    l_load_abilities(device);
    num_type_info = 0;

    // Boolean.
    type_info[ num_type_info++ ] = TypeInfo( "bool" ).set_bool().set_size(1).set_buf_elem_type("uchar");

    // Vector types, and the related scalar element types.
    for ( iscalar=0; iscalar < sizeof(vecbase)/sizeof(vecbase[0]) ; ++iscalar ) {
        if ( !gHasLong && strstr(vecbase[iscalar],"long") ) continue;
        if ( !l_has_double && strstr(vecbase[iscalar],"double") ) continue;

        // Scalar
        TypeInfo* elem_type = type_info + num_type_info++;
        *elem_type = TypeInfo( vecbase[iscalar] ).set_vecbase().set_size( vecbase_size[iscalar] );

        // Vector
        for ( ivecsize=0; ivecsize < sizeof(vecsizes)/sizeof(vecsizes[0]) ; ivecsize++ ) {
            type_info[ num_type_info++ ] = TypeInfo( elem_type, vecsizes[ivecsize] );
        }
    }

    // Size_t-like types
    for ( iscalar=0; iscalar < sizeof(like_size_t)/sizeof(like_size_t[0]) ; ++iscalar ) {
        type_info[ num_type_info++ ] = TypeInfo( like_size_t[iscalar] ).set_like_size_t();
    }

    // Atomic types.
    for ( iscalar=0; iscalar < sizeof(atomics)/sizeof(atomics[0]) ; ++iscalar ) {
        if ( !l_has_int64_atomics && strstr(atomics[iscalar],"long") ) continue;
        if ( !(l_has_int64_atomics && l_has_double) && strstr(atomics[iscalar],"double") ) continue;

        // The +7 is used to skip over the "atomic_" prefix.
        const char* buf_type = atomics[iscalar] + 7;
        type_info[ num_type_info++ ] = TypeInfo( atomics[iscalar] ).set_atomic().set_size( atomics_size[iscalar] ).set_buf_elem_type( buf_type );
    }
    if ( l_has_intptr_atomics ) {
        for ( iscalar=0; iscalar < sizeof(intptr_atomics)/sizeof(intptr_atomics[0]) ; ++iscalar ) {
            type_info[ num_type_info++ ] = TypeInfo( intptr_atomics[iscalar] ).set_atomic().set_like_size_t();
        }
    }

    assert( num_type_info <= MAX_TYPES ); // or increase MAX_TYPES

#if 0
    for ( size_t i = 0 ; i < num_type_info ; i++ ) {
        type_info[ i ].dump(stdout);
    }
    exit(0);
#endif

    return status;
}

static const TypeInfo& l_find_type( const char* name )
{
    for ( size_t i = 0; i < num_type_info ; i++ ) {
        if ( 0 == strcmp( name, type_info[i].get_name_c_str() ) ) return type_info[i];
    }
    assert(0);
}



// Populate return parameters for max program variable size, preferred program variable size.

static int l_get_device_info(cl_device_id device, size_t* max_size_ret, size_t* pref_size_ret)
{
    int err = CL_SUCCESS;
    size_t return_size = 0;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, sizeof(*max_size_ret), max_size_ret, &return_size);
    if ( err != CL_SUCCESS ) {
        log_error("Error: Failed to get device info for CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE\n");
        return err;
    }
    if ( return_size != sizeof(size_t) ) {
        log_error("Error: Invalid size %d returned for CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE\n", (int)return_size );
        return 1;
    }
    if ( return_size != sizeof(size_t) ) {
        log_error("Error: Invalid size %d returned for CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE\n", (int)return_size );
        return 1;
    }

    return_size = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, sizeof(*pref_size_ret), pref_size_ret, &return_size);
    if ( err != CL_SUCCESS ) {
        log_error("Error: Failed to get device info for CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: %d\n",err);
        return err;
    }
    if ( return_size != sizeof(size_t) ) {
        log_error("Error: Invalid size %d returned for CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE\n", (int)return_size );
        return 1;
    }

    return CL_SUCCESS;
}


static void l_set_randomly( cl_uchar* buf, size_t buf_size, RandomSeed& rand_state )
{
    assert( 0 == (buf_size % sizeof(cl_uint) ) );
    for ( size_t i = 0; i < buf_size ; i += sizeof(cl_uint) ) {
        *( (cl_uint*)(buf + i) ) = genrand_int32( rand_state );
    }
#if 0
    for ( size_t i = 0; i < buf_size ; i++ ) {
        printf("%02x",buf[i]);
    }
    printf("\n");
#endif
}

// Return num_value values of the given type.
// Returns CL_SUCCESS if they compared as equal.
static int l_compare( const char* test_name, const cl_uchar* expected, const cl_uchar* received, size_t num_values, const TypeInfo&ti )
{
    // Compare only the valid returned bytes.
    for ( unsigned value_idx = 0; value_idx < num_values; value_idx++ ) {
        const cl_uchar* expv = expected + value_idx * ti.get_size();
        const cl_uchar* gotv = received + value_idx * ti.get_size();
        if ( memcmp( expv, gotv, ti.get_value_size() ) ) {
            std::string exp_str = ti.as_string( expv );
            std::string got_str = ti.as_string( gotv );
            log_error("Error: %s test for type %s, at index %d: Expected %s got %s\n",
                    test_name,
                    ti.get_name_c_str(), value_idx,
                    exp_str.c_str(),
                    got_str.c_str() );
            return 1;
        }
    }
    return CL_SUCCESS;
}

// Copy a target value from src[idx] to dest[idx]
static int l_copy( cl_uchar* dest, unsigned dest_idx, const cl_uchar* src, unsigned src_idx, const TypeInfo&ti )
{
    cl_uchar* raw_dest      = dest + dest_idx * ti.get_size();
    const cl_uchar* raw_src =  src +  src_idx * ti.get_size();
    memcpy( raw_dest, raw_src,  ti.get_value_size() );

    return 0;
}


static std::string conversion_functions(const TypeInfo& ti)
{
    std::string result;
    static char buf[MAX_STR];
    int num_printed = 0;
    // The atomic types just use the base type.
    if ( ti.is_atomic() || 0 == strcmp( ti.get_buf_elem_type(), ti.get_name_c_str() ) ) {
        // The type is represented in a buffer by itself.
        num_printed = snprintf(buf,MAX_STR,
                "%s from_buf(%s a) { return a; }\n"
                "%s to_buf(%s a) { return a; }\n",
                ti.get_buf_elem_type(), ti.get_buf_elem_type(),
                ti.get_buf_elem_type(), ti.get_buf_elem_type() );
    } else {
        // Just use C-style cast.
        num_printed = snprintf(buf,MAX_STR,
                "%s from_buf(%s a) { return (%s)a; }\n"
                "%s to_buf(%s a) { return (%s)a; }\n",
                ti.get_name_c_str(), ti.get_buf_elem_type(), ti.get_name_c_str(),
                ti.get_buf_elem_type(), ti.get_name_c_str(), ti.get_buf_elem_type() );
    }
    // Add initializations.
    if ( ti.is_atomic() ) {
        num_printed += snprintf( buf + num_printed, MAX_STR-num_printed,
                "#define INIT_VAR(a) ATOMIC_VAR_INIT(a)\n" );
    } else {
        // This cast works even if the target type is a vector type.
        num_printed += snprintf( buf + num_printed, MAX_STR-num_printed,
                "#define INIT_VAR(a) ((%s)(a))\n", ti.get_name_c_str());
    }
    assert( num_printed < MAX_STR ); // or increase MAX_STR
    result = buf;
    return result;
}

static std::string global_decls(const TypeInfo& ti, bool with_init )
{
    const char* tn = ti.get_name_c_str();
    const char* vol = (ti.is_atomic() ? " volatile " : " ");
    static char decls[MAX_STR];
    int num_printed = 0;
    if ( with_init ) {
        const char *decls_template_with_init =
            "%s %s var = INIT_VAR(0);\n"
            "global %s %s g_var = INIT_VAR(1);\n"
            "%s %s a_var[2] = { INIT_VAR(1), INIT_VAR(1) };\n"
            "volatile global %s %s* p_var = &a_var[1];\n\n";
        num_printed = snprintf(decls,sizeof(decls),decls_template_with_init,
                vol,tn,vol,tn,vol,tn,vol,tn);
    } else {
        const char *decls_template_no_init =
            "%s %s var;\n"
            "global %s %s g_var;\n"
            "%s %s a_var[2];\n"
            "global %s %s* p_var;\n\n";
        num_printed = snprintf(decls,sizeof(decls),decls_template_no_init,
             vol,tn,vol,tn,vol,tn,vol,tn);
    }
    assert( num_printed < sizeof(decls) );
    return std::string(decls);
}


// Return the source text for the writer function for the given type.
// For types that can't be passed as pointer-to-type as a kernel argument,
// use a substitute base type of the same size.
static std::string writer_function(const TypeInfo& ti)
{
    static char writer_src[MAX_STR];
    int num_printed = 0;
    if ( !ti.is_atomic() ) {
        const char* writer_template_normal =
            "kernel void writer( global %s* src, uint idx ) {\n"
            "  var = from_buf(src[0]);\n"
            "  g_var = from_buf(src[1]);\n"
            "  a_var[0] = from_buf(src[2]);\n"
            "  a_var[1] = from_buf(src[3]);\n"
            "  p_var = a_var + idx;\n"
            "}\n\n";
        num_printed = snprintf(writer_src,sizeof(writer_src),writer_template_normal,ti.get_buf_elem_type());
    } else {
        const char* writer_template_atomic =
            "kernel void writer( global %s* src, uint idx ) {\n"
            "  atomic_store( &var, from_buf(src[0]) );\n"
            "  atomic_store( &g_var, from_buf(src[1]) );\n"
            "  atomic_store( &a_var[0], from_buf(src[2]) );\n"
            "  atomic_store( &a_var[1], from_buf(src[3]) );\n"
            "  p_var = a_var + idx;\n"
            "}\n\n";
        num_printed = snprintf(writer_src,sizeof(writer_src),writer_template_atomic,ti.get_buf_elem_type());
    }
    assert( num_printed < sizeof(writer_src) );
    std::string result = writer_src;
    return result;
}


// Return source text for teh reader function for the given type.
// For types that can't be passed as pointer-to-type as a kernel argument,
// use a substitute base type of the same size.
static std::string reader_function(const TypeInfo& ti)
{
    static char reader_src[MAX_STR];
    int num_printed = 0;
    if ( !ti.is_atomic() ) {
        const char* reader_template_normal =
            "kernel void reader( global %s* dest, %s ptr_write_val ) {\n"
            "  *p_var = from_buf(ptr_write_val);\n"
            "  dest[0] = to_buf(var);\n"
            "  dest[1] = to_buf(g_var);\n"
            "  dest[2] = to_buf(a_var[0]);\n"
            "  dest[3] = to_buf(a_var[1]);\n"
            "}\n\n";
        num_printed = snprintf(reader_src,sizeof(reader_src),reader_template_normal,ti.get_buf_elem_type(),ti.get_buf_elem_type());
    } else {
        const char* reader_template_atomic =
            "kernel void reader( global %s* dest, %s ptr_write_val ) {\n"
            "  atomic_store( p_var, from_buf(ptr_write_val) );\n"
            "  dest[0] = to_buf( atomic_load( &var ) );\n"
            "  dest[1] = to_buf( atomic_load( &g_var ) );\n"
            "  dest[2] = to_buf( atomic_load( &a_var[0] ) );\n"
            "  dest[3] = to_buf( atomic_load( &a_var[1] ) );\n"
            "}\n\n";
        num_printed = snprintf(reader_src,sizeof(reader_src),reader_template_atomic,ti.get_buf_elem_type(),ti.get_buf_elem_type());
    }
    assert( num_printed < sizeof(reader_src) );
    std::string result = reader_src;
    return result;
}


// Check write-then-read.
static int l_write_read( cl_device_id device, cl_context context, cl_command_queue queue )
{
    int status = CL_SUCCESS;
    int itype;

    RandomSeed rand_state( gRandomSeed );

    for ( itype = 0; itype < num_type_info ; itype++ ) {
        status = status | l_write_read_for_type(device,context,queue,type_info[itype], rand_state );
        FLUSH;
    }

    return status;
}
static int l_write_read_for_type( cl_device_id device, cl_context context, cl_command_queue queue, const TypeInfo& ti, RandomSeed& rand_state )
{
    int err = CL_SUCCESS;
    std::string type_name( ti.get_name() );
    const char* tn = type_name.c_str();
    log_info("  %s ",tn);

    StringTable ksrc;
    ksrc.add( l_get_fp64_pragma() );
    ksrc.add( l_get_cles_int64_pragma() );
    if (ti.is_atomic_64bit())
      ksrc.add( l_get_int64_atomic_pragma() );
    ksrc.add( conversion_functions(ti) );
    ksrc.add( global_decls(ti,false) );
    ksrc.add( writer_function(ti) );
    ksrc.add( reader_function(ti) );

    int status = CL_SUCCESS;
    clProgramWrapper program;
    clKernelWrapper writer;

    status = create_single_kernel_helper_with_build_options(context, &program, &writer, ksrc.num_str(), ksrc.strs(), "writer", OPTIONS);
    test_error_ret(status,"Failed to create program for read-after-write test",status);

    clKernelWrapper reader( clCreateKernel( program, "reader", &status ) );
    test_error_ret(status,"Failed to create reader kernel for read-after-write test",status);

    // Check size query.
    size_t used_bytes = 0;
    status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, sizeof(used_bytes), &used_bytes, 0 );
    test_error_ret(status,"Failed to query global variable total size",status);
    size_t expected_used_bytes =
        (NUM_TESTED_VALUES-1)*ti.get_size() // Two regular variables and an array of 2 elements.
        + ( l_64bit_device ? 8 : 4 ); // The pointer
    if ( used_bytes < expected_used_bytes ) {
        log_error("Error program query for global variable total size query failed: Expected at least %llu but got %llu\n", (unsigned long long)expected_used_bytes, (unsigned long long)used_bytes );
        err |= 1;
    }

    // We need to create 5 random values of the given type,
    // and read 4 of them back.
    const size_t write_data_size = NUM_TESTED_VALUES * sizeof(cl_ulong16);
    const size_t read_data_size = (NUM_TESTED_VALUES - 1) * sizeof(cl_ulong16);
    cl_uchar* write_data = (cl_uchar*)align_malloc(write_data_size, ALIGNMENT);
    cl_uchar* read_data = (cl_uchar*)align_malloc(read_data_size, ALIGNMENT);

    clMemWrapper write_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, write_data_size, write_data, &status ) );
    test_error_ret(status,"Failed to allocate write buffer",status);
    clMemWrapper read_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, read_data_size, read_data, &status ) );
    test_error_ret(status,"Failed to allocate read buffer",status);

    status = clSetKernelArg(writer,0,sizeof(cl_mem),&write_mem); test_error_ret(status,"set arg",status);
    status = clSetKernelArg(reader,0,sizeof(cl_mem),&read_mem); test_error_ret(status,"set arg",status);

    // Boolean random data needs to be massaged a bit more.
    const int num_rounds = ti.is_bool() ? (1 << NUM_TESTED_VALUES ) : NUM_ROUNDS;
    unsigned bool_iter = 0;

    for ( int iround = 0; iround < num_rounds ; iround++ ) {
        for ( cl_uint iptr_idx = 0; iptr_idx < 2 ; iptr_idx++ ) { // Index into array, to write via pointer
            // Generate new random data to push through.
            // Generate 5 * 128 bytes all the time, even though the test for many types use less than all that.

            cl_uchar *write_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, write_mem, CL_TRUE, CL_MAP_WRITE, 0, write_data_size, 0, 0, 0, 0);

            if ( ti.is_bool() ) {
                // For boolean, random data cast to bool isn't very random.
                // So use the bottom bit of bool_value_iter to get true
                // diversity.
                for ( unsigned value_idx = 0; value_idx < NUM_TESTED_VALUES ; value_idx++ ) {
                    write_data[value_idx] = (1<<value_idx) & bool_iter;
                    //printf(" %s", (write_data[value_idx] ? "true" : "false" ));
                }
                bool_iter++;
            } else {
                l_set_randomly( write_data, write_data_size, rand_state );
            }
            status = clSetKernelArg(writer,1,sizeof(cl_uint),&iptr_idx); test_error_ret(status,"set arg",status);

            // The value to write via the pointer should be taken from the
            // 5th typed slot of the write_data.
            status = clSetKernelArg(reader,1,ti.get_size(),write_data + (NUM_TESTED_VALUES-1)*ti.get_size()); test_error_ret(status,"set arg",status);

            // Determine the expected values.
            cl_uchar expected[read_data_size];
            memset( expected, -1, sizeof(expected) );
            l_copy( expected, 0, write_data, 0, ti );
            l_copy( expected, 1, write_data, 1, ti );
            l_copy( expected, 2, write_data, 2, ti );
            l_copy( expected, 3, write_data, 3, ti );
            // But we need to take into account the value from the pointer write.
            // The 2 represents where the "a" array values begin in our read-back.
            l_copy( expected, 2 + iptr_idx, write_data, 4, ti );

            clEnqueueUnmapMemObject(queue, write_mem, write_ptr, 0, 0, 0);

            if ( ti.is_bool() ) {
                // Collapse down to one bit.
                for ( unsigned i = 0; i <  NUM_TESTED_VALUES-1 ; i++ ) expected[i] = (bool)expected[i];
            }

            cl_uchar *read_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, read_mem, CL_TRUE, CL_MAP_READ, 0, read_data_size, 0, 0, 0, 0);
            memset(read_data, -1, read_data_size);
            clEnqueueUnmapMemObject(queue, read_mem, read_ptr, 0, 0, 0);

            // Now run the kernel
            const size_t one = 1;
            status = clEnqueueNDRangeKernel(queue,writer,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue writer",status);
            status = clEnqueueNDRangeKernel(queue,reader,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue reader",status);
            status = clFinish(queue); test_error_ret(status,"finish",status);

            read_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, read_mem, CL_TRUE, CL_MAP_READ, 0, read_data_size, 0, 0, 0, 0);

            if ( ti.is_bool() ) {
                // Collapse down to one bit.
                for ( unsigned i = 0; i <  NUM_TESTED_VALUES-1 ; i++ ) read_data[i] = (bool)read_data[i];
            }

            // Compare only the valid returned bytes.
            int compare_result = l_compare( "read-after-write", expected, read_data, NUM_TESTED_VALUES-1, ti );
            // log_info("Compared %d values each of size %llu. Result %d\n", NUM_TESTED_VALUES-1, (unsigned long long)ti.get_value_size(), compare_result );
            err |= compare_result;

            clEnqueueUnmapMemObject(queue, read_mem, read_ptr, 0, 0, 0);

            if ( err ) break;
        }
    }

    if ( CL_SUCCESS == err ) { log_info("OK\n"); FLUSH; }
    align_free(write_data);
    align_free(read_data);
    return err;
}


// Check initialization, then, read, then write, then read.
static int l_init_write_read( cl_device_id device, cl_context context, cl_command_queue queue )
{
    int status = CL_SUCCESS;
    int itype;

    RandomSeed rand_state( gRandomSeed );

    for ( itype = 0; itype < num_type_info ; itype++ ) {
        status = status | l_init_write_read_for_type(device,context,queue,type_info[itype], rand_state );
    }
    return status;
}
static int l_init_write_read_for_type( cl_device_id device, cl_context context, cl_command_queue queue, const TypeInfo& ti, RandomSeed& rand_state )
{
    int err = CL_SUCCESS;
    std::string type_name( ti.get_name() );
    const char* tn = type_name.c_str();
    log_info("  %s ",tn);

    StringTable ksrc;
    ksrc.add( l_get_fp64_pragma() );
    ksrc.add( l_get_cles_int64_pragma() );
    if (ti.is_atomic_64bit())
      ksrc.add( l_get_int64_atomic_pragma() );
    ksrc.add( conversion_functions(ti) );
    ksrc.add( global_decls(ti,true) );
    ksrc.add( writer_function(ti) );
    ksrc.add( reader_function(ti) );

    int status = CL_SUCCESS;
    clProgramWrapper program;
    clKernelWrapper writer;

    status = create_single_kernel_helper_with_build_options(context, &program, &writer, ksrc.num_str(), ksrc.strs(), "writer", OPTIONS);
    test_error_ret(status,"Failed to create program for init-read-after-write test",status);

    clKernelWrapper reader( clCreateKernel( program, "reader", &status ) );
    test_error_ret(status,"Failed to create reader kernel for init-read-after-write test",status);

    // Check size query.
    size_t used_bytes = 0;
    status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, sizeof(used_bytes), &used_bytes, 0 );
    test_error_ret(status,"Failed to query global variable total size",status);
    size_t expected_used_bytes =
        (NUM_TESTED_VALUES-1)*ti.get_size() // Two regular variables and an array of 2 elements.
        + ( l_64bit_device ? 8 : 4 ); // The pointer
    if ( used_bytes < expected_used_bytes ) {
        log_error("Error: program query for global variable total size query failed: Expected at least %llu but got %llu\n", (unsigned long long)expected_used_bytes, (unsigned long long)used_bytes );
        err |= 1;
    }

    // We need to create 5 random values of the given type,
    // and read 4 of them back.
    const size_t write_data_size = NUM_TESTED_VALUES * sizeof(cl_ulong16);
    const size_t read_data_size = (NUM_TESTED_VALUES-1) * sizeof(cl_ulong16);

    cl_uchar* write_data = (cl_uchar*)align_malloc(write_data_size, ALIGNMENT);
    cl_uchar* read_data = (cl_uchar*)align_malloc(read_data_size, ALIGNMENT);
    clMemWrapper write_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, write_data_size, write_data, &status ) );
    test_error_ret(status,"Failed to allocate write buffer",status);
    clMemWrapper read_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, read_data_size, read_data, &status ) );
    test_error_ret(status,"Failed to allocate read buffer",status);

    status = clSetKernelArg(writer,0,sizeof(cl_mem),&write_mem); test_error_ret(status,"set arg",status);
    status = clSetKernelArg(reader,0,sizeof(cl_mem),&read_mem); test_error_ret(status,"set arg",status);

    // Boolean random data needs to be massaged a bit more.
    const int num_rounds = ti.is_bool() ? (1 << NUM_TESTED_VALUES ) : NUM_ROUNDS;
    unsigned bool_iter = 0;

    // We need to count iterations.  We do something *different on the
    // first iteration, to ensure we actually pick up the initialized
    // values.
    unsigned iteration = 0;

    for ( int iround = 0; iround < num_rounds ; iround++ ) {
        for ( cl_uint iptr_idx = 0; iptr_idx < 2 ; iptr_idx++ ) { // Index into array, to write via pointer
            // Generate new random data to push through.
            // Generate 5 * 128 bytes all the time, even though the test for many types use less than all that.

            cl_uchar *write_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, write_mem, CL_TRUE, CL_MAP_WRITE, 0, write_data_size, 0, 0, 0, 0);

            if ( ti.is_bool() ) {
                // For boolean, random data cast to bool isn't very random.
                // So use the bottom bit of bool_value_iter to get true
                // diversity.
                for ( unsigned value_idx = 0; value_idx < NUM_TESTED_VALUES ; value_idx++ ) {
                    write_data[value_idx] = (1<<value_idx) & bool_iter;
                    //printf(" %s", (write_data[value_idx] ? "true" : "false" ));
                }
                bool_iter++;
            } else {
                l_set_randomly( write_data, write_data_size, rand_state );
            }
            status = clSetKernelArg(writer,1,sizeof(cl_uint),&iptr_idx); test_error_ret(status,"set arg",status);

            if ( !iteration ) {
                // On first iteration, the value we write via the last arg
                // to the "reader" function is 0.
                // It's way easier to code the test this way.
                ti.init( write_data + (NUM_TESTED_VALUES-1)*ti.get_size(), 0 );
            }

            // The value to write via the pointer should be taken from the
            // 5th typed slot of the write_data.
            status = clSetKernelArg(reader,1,ti.get_size(),write_data + (NUM_TESTED_VALUES-1)*ti.get_size()); test_error_ret(status,"set arg",status);

            // Determine the expected values.
            cl_uchar expected[read_data_size];
            memset( expected, -1, sizeof(expected) );
            if ( iteration ) {
                l_copy( expected, 0, write_data, 0, ti );
                l_copy( expected, 1, write_data, 1, ti );
                l_copy( expected, 2, write_data, 2, ti );
                l_copy( expected, 3, write_data, 3, ti );
                // But we need to take into account the value from the pointer write.
                // The 2 represents where the "a" array values begin in our read-back.
                // But we need to take into account the value from the pointer write.
                l_copy( expected, 2 + iptr_idx, write_data, 4, ti );
            } else {
                // On first iteration, expect these initialized values!
                // See the decls_template_with_init above.
                ti.init( expected, 0 );
                ti.init( expected + ti.get_size(), 1 );
                ti.init( expected + 2*ti.get_size(), 1 );
                // Emulate the effect of the write via the pointer.
                // The value is 0, not 1 (see above).
                // The pointer is always initialized to the second element
                // of the array. So it goes into slot 3 of the "expected" array.
                ti.init( expected + 3*ti.get_size(), 0 );
            }

            if ( ti.is_bool() ) {
                // Collapse down to one bit.
                for ( unsigned i = 0; i <  NUM_TESTED_VALUES-1 ; i++ ) expected[i] = (bool)expected[i];
            }

            clEnqueueUnmapMemObject(queue, write_mem, write_ptr, 0, 0, 0);

            cl_uchar *read_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, read_mem, CL_TRUE, CL_MAP_READ, 0, read_data_size, 0, 0, 0, 0);
            memset( read_data, -1, read_data_size );
            clEnqueueUnmapMemObject(queue, read_mem, read_ptr, 0, 0, 0);

            // Now run the kernel
            const size_t one = 1;
            if ( iteration ) {
                status = clEnqueueNDRangeKernel(queue,writer,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue writer",status);
            } else {
                // On first iteration, we should be picking up the
                // initialized value. So don't enqueue the writer.
            }
            status = clEnqueueNDRangeKernel(queue,reader,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue reader",status);
            status = clFinish(queue); test_error_ret(status,"finish",status);

            read_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, read_mem, CL_TRUE, CL_MAP_READ, 0, read_data_size, 0, 0, 0, 0);

            if ( ti.is_bool() ) {
                // Collapse down to one bit.
                for ( unsigned i = 0; i <  NUM_TESTED_VALUES-1 ; i++ ) read_data[i] = (bool)read_data[i];
            }

            // Compare only the valid returned bytes.
            //log_info(" Round %d ptr_idx %u\n", iround, iptr_idx );
            int compare_result = l_compare( "init-write-read", expected, read_data, NUM_TESTED_VALUES-1, ti );
            //log_info("Compared %d values each of size %llu. Result %d\n", NUM_TESTED_VALUES-1, (unsigned long long)ti.get_value_size(), compare_result );
            err |= compare_result;

            clEnqueueUnmapMemObject(queue, read_mem, read_ptr, 0, 0, 0);

            if ( err ) break;

            iteration++;
        }
    }

    if ( CL_SUCCESS == err ) { log_info("OK\n"); FLUSH; }
    align_free(write_data);
    align_free(read_data);

    return err;
}


// Check that we can make at least one variable with size
// max_size which is returned from the device info property : CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE.
static int l_capacity( cl_device_id device, cl_context context, cl_command_queue queue, size_t max_size )
{
    int err = CL_SUCCESS;
    // Just test one type.
    const TypeInfo ti( l_find_type("uchar") );
    log_info(" l_capacity...");

    const char prog_src_template[] =
#if defined(_WIN32)
        "uchar var[%Iu];\n\n"
#else
        "uchar var[%zu];\n\n"
#endif
        "kernel void get_max_size( global ulong* size_ret ) {\n"
#if defined(_WIN32)
        "  *size_ret = (ulong)%Iu;\n"
#else
        "  *size_ret = (ulong)%zu;\n"
#endif
        "}\n\n"
        "kernel void writer( global uchar* src ) {\n"
        "  var[get_global_id(0)] = src[get_global_linear_id()];\n"
        "}\n\n"
        "kernel void reader( global uchar* dest ) {\n"
        "  dest[get_global_linear_id()] = var[get_global_id(0)];\n"
        "}\n\n";
    char prog_src[MAX_STR];
    int num_printed = snprintf(prog_src,sizeof(prog_src),prog_src_template,max_size, max_size);
    assert( num_printed < MAX_STR ); // or increase MAX_STR

    StringTable ksrc;
    ksrc.add( prog_src );

    int status = CL_SUCCESS;
    clProgramWrapper program;
    clKernelWrapper get_max_size;

    status = create_single_kernel_helper_with_build_options(context, &program, &get_max_size, ksrc.num_str(), ksrc.strs(), "get_max_size", OPTIONS);
    test_error_ret(status,"Failed to create program for capacity test",status);

    // Check size query.
    size_t used_bytes = 0;
    status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, sizeof(used_bytes), &used_bytes, 0 );
    test_error_ret(status,"Failed to query global variable total size",status);
    if ( used_bytes < max_size ) {
        log_error("Error: program query for global variable total size query failed: Expected at least %llu but got %llu\n", (unsigned long long)max_size, (unsigned long long)used_bytes );
        err |= 1;
    }

    // Prepare to execute
    clKernelWrapper writer( clCreateKernel( program, "writer", &status ) );
    test_error_ret(status,"Failed to create writer kernel for capacity test",status);
    clKernelWrapper reader( clCreateKernel( program, "reader", &status ) );
    test_error_ret(status,"Failed to create reader kernel for capacity test",status);

    cl_ulong max_size_ret = 0;
    const size_t arr_size = 10*1024*1024;
    cl_uchar* buffer = (cl_uchar*) align_malloc( arr_size, ALIGNMENT );

    if ( !buffer ) { log_error("Failed to allocate buffer\n"); return 1; }

    clMemWrapper max_size_ret_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, sizeof(max_size_ret), &max_size_ret, &status ) );
    test_error_ret(status,"Failed to allocate size query buffer",status);
    clMemWrapper buffer_mem( clCreateBuffer( context, CL_MEM_READ_WRITE, arr_size, 0, &status ) );
    test_error_ret(status,"Failed to allocate write buffer",status);

    status = clSetKernelArg(get_max_size,0,sizeof(cl_mem),&max_size_ret_mem); test_error_ret(status,"set arg",status);
    status = clSetKernelArg(writer,0,sizeof(cl_mem),&buffer_mem); test_error_ret(status,"set arg",status);
    status = clSetKernelArg(reader,0,sizeof(cl_mem),&buffer_mem); test_error_ret(status,"set arg",status);

    // Check the macro value of CL_DEVICE_MAX_GLOBAL_VARIABLE
    const size_t one = 1;
    status = clEnqueueNDRangeKernel(queue,get_max_size,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue size query",status);
    status = clFinish(queue); test_error_ret(status,"finish",status);

    cl_uchar *max_size_ret_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, max_size_ret_mem, CL_TRUE, CL_MAP_READ, 0, sizeof(max_size_ret), 0, 0, 0, 0);
    if ( max_size_ret != max_size ) {
        log_error("Error: preprocessor definition for CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE is %llu and does not match device query value %llu\n",
                (unsigned long long) max_size_ret,
                (unsigned long long) max_size );
        err |= 1;
    }
    clEnqueueUnmapMemObject(queue, max_size_ret_mem, max_size_ret_ptr, 0, 0, 0);

    RandomSeed rand_state_write( gRandomSeed );
    for ( size_t offset = 0; offset < max_size ; offset += arr_size ) {
        size_t curr_size = (max_size - offset) < arr_size ? (max_size - offset) : arr_size;
        l_set_randomly( buffer, curr_size, rand_state_write );
        status = clEnqueueWriteBuffer (queue, buffer_mem, CL_TRUE, 0, curr_size, buffer, 0, 0, 0);test_error_ret(status,"populate buffer_mem object",status);
        status = clEnqueueNDRangeKernel(queue,writer,1,&offset,&curr_size,0,0,0,0); test_error_ret(status,"enqueue writer",status);
    status = clFinish(queue); test_error_ret(status,"finish",status);
    }

    RandomSeed rand_state_read( gRandomSeed );
    for ( size_t offset = 0; offset < max_size ; offset += arr_size ) {
        size_t curr_size = (max_size - offset) < arr_size ? (max_size - offset) : arr_size;
        status = clEnqueueNDRangeKernel(queue,reader,1,&offset,&curr_size,0,0,0,0); test_error_ret(status,"enqueue reader",status);
        cl_uchar* read_mem_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, buffer_mem, CL_TRUE, CL_MAP_READ, 0, curr_size, 0, 0, 0, &status);test_error_ret(status,"map read data",status);
        l_set_randomly( buffer, curr_size, rand_state_read );
        err |= l_compare( "capacity", buffer, read_mem_ptr, curr_size, ti );
        clEnqueueUnmapMemObject(queue, buffer_mem, read_mem_ptr, 0, 0, 0);
    }

    if ( CL_SUCCESS == err ) { log_info("OK\n"); FLUSH; }
    align_free(buffer);

    return err;
}


// Check operation on a user type.
static int l_user_type( cl_device_id device, cl_context context, cl_command_queue queue, bool separate_compile )
{
    int err = CL_SUCCESS;
    // Just test one type.
    const TypeInfo ti( l_find_type("uchar") );
    log_info(" l_user_type %s...", separate_compile ? "separate compilation" : "single source compilation" );

    if ( separate_compile && ! l_linker_available ) {
        log_info("Separate compilation is not supported. Skipping test\n");
        return err;
    }

    const char type_src[] =
        "typedef struct { uchar c; uint i; } my_struct_t;\n\n";
    const char def_src[] =
        "my_struct_t var = { 'a', 42 };\n\n";
    const char decl_src[] =
        "extern my_struct_t var;\n\n";

    // Don't use a host struct. We can't guarantee that the host
    // compiler has the same structure layout as the device compiler.
    const char writer_src[] =
        "kernel void writer( uchar c, uint i ) {\n"
        "  var.c = c;\n"
        "  var.i = i;\n"
        "}\n\n";
    const char reader_src[] =
        "kernel void reader( global uchar* C, global uint* I ) {\n"
        "  *C = var.c;\n"
        "  *I = var.i;\n"
        "}\n\n";

    clProgramWrapper program;

    if ( separate_compile ) {
        // Separate compilation flow.
        StringTable wksrc;
        wksrc.add( type_src );
        wksrc.add( def_src );
        wksrc.add( writer_src );

        StringTable rksrc;
        rksrc.add( type_src );
        rksrc.add( decl_src );
        rksrc.add( reader_src );

        int status = CL_SUCCESS;
        clProgramWrapper writer_program( clCreateProgramWithSource( context, wksrc.num_str(), wksrc.strs(), wksrc.lengths(), &status ) );
        test_error_ret(status,"Failed to create writer program for user type test",status);

        status = clCompileProgram( writer_program, 1, &device, OPTIONS, 0, 0, 0, 0, 0 );
        if(check_error(status, "Failed to compile writer program for user type test (%s)", IGetErrorString(status)))
        {
            print_build_log(writer_program, 1, &device, wksrc.num_str(), wksrc.strs(), wksrc.lengths(), OPTIONS);
            return status;
        }

        clProgramWrapper reader_program( clCreateProgramWithSource( context, rksrc.num_str(), rksrc.strs(), rksrc.lengths(), &status ) );
        test_error_ret(status,"Failed to create reader program for user type test",status);

        status = clCompileProgram( reader_program, 1, &device, OPTIONS, 0, 0, 0, 0, 0 );
        if(check_error(status, "Failed to compile reader program for user type test (%s)", IGetErrorString(status)))
        {
            print_build_log(reader_program, 1, &device, rksrc.num_str(), rksrc.strs(), rksrc.lengths(), OPTIONS);
            return status;
        }

        cl_program progs[2];
        progs[0] = writer_program;
        progs[1] = reader_program;

        program = clLinkProgram( context, 1, &device, "", 2, progs, 0, 0, &status );
        if(check_error(status, "Failed to link program for user type test (%s)", IGetErrorString(status)))
        {
            print_build_log(program, 1, &device, 0, NULL, NULL, "");
            return status;
        }
    } else {
        // Single compilation flow.
        StringTable ksrc;
        ksrc.add( type_src );
        ksrc.add( def_src );
        ksrc.add( writer_src );
        ksrc.add( reader_src );

        int status = CL_SUCCESS;

        status = create_single_kernel_helper_create_program(context, &program, ksrc.num_str(), ksrc.strs(), OPTIONS);
        if(check_error(status, "Failed to build program for user type test (%s)", IGetErrorString(status)))
        {
            print_build_log(program, 1, &device, ksrc.num_str(), ksrc.strs(), ksrc.lengths(), OPTIONS);
            return status;
        }

        status = clBuildProgram(program, 1, &device, OPTIONS, 0, 0);
        if(check_error(status, "Failed to compile program for user type test (%s)", IGetErrorString(status)))
        {
            print_build_log(program, 1, &device, ksrc.num_str(), ksrc.strs(), ksrc.lengths(), OPTIONS);
            return status;
        }
    }


    // Check size query.
    size_t used_bytes = 0;
    int status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, sizeof(used_bytes), &used_bytes, 0 );
    test_error_ret(status,"Failed to query global variable total size",status);
    size_t expected_size = sizeof(cl_uchar) + sizeof(cl_uint);
    if ( used_bytes < expected_size ) {
        log_error("Error: program query for global variable total size query failed: Expected at least %llu but got %llu\n", (unsigned long long)expected_size, (unsigned long long)used_bytes );
        err |= 1;
    }

    // Prepare to execute
    clKernelWrapper writer( clCreateKernel( program, "writer", &status ) );
    test_error_ret(status,"Failed to create writer kernel for user type test",status);
    clKernelWrapper reader( clCreateKernel( program, "reader", &status ) );
    test_error_ret(status,"Failed to create reader kernel for user type test",status);

    // Set up data.
    cl_uchar* uchar_data = (cl_uchar*)align_malloc(sizeof(cl_uchar), ALIGNMENT);
    cl_uint* uint_data = (cl_uint*)align_malloc(sizeof(cl_uint), ALIGNMENT);

    clMemWrapper uchar_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, sizeof(cl_uchar), uchar_data, &status ) );
    test_error_ret(status,"Failed to allocate uchar buffer",status);
    clMemWrapper uint_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, sizeof(cl_uint), uint_data, &status ) );
    test_error_ret(status,"Failed to allocate uint buffer",status);

    status = clSetKernelArg(reader,0,sizeof(cl_mem),&uchar_mem); test_error_ret(status,"set arg",status);
    status = clSetKernelArg(reader,1,sizeof(cl_mem),&uint_mem); test_error_ret(status,"set arg",status);

    cl_uchar expected_uchar = 'a';
    cl_uint expected_uint = 42;
    for ( unsigned iter = 0; iter < 5 ; iter++ ) { // Must go around at least twice
        // Read back data
        *uchar_data = -1;
        *uint_data = -1;
        const size_t one = 1;
        status = clEnqueueNDRangeKernel(queue,reader,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue reader",status);
        status = clFinish(queue); test_error_ret(status,"finish",status);

        cl_uchar *uint_data_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, uint_mem, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint), 0, 0, 0, 0);
        cl_uchar *uchar_data_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, uchar_mem, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uchar), 0, 0, 0, 0);

        if ( expected_uchar != *uchar_data || expected_uint != *uint_data ) {
            log_error("FAILED: Iteration %d Got (0x%2x,%d) but expected (0x%2x,%d)\n",
                    iter, (int)*uchar_data, *uint_data, (int)expected_uchar, expected_uint );
            err |= 1;
        }

        clEnqueueUnmapMemObject(queue, uint_mem, uint_data_ptr, 0, 0, 0);
        clEnqueueUnmapMemObject(queue, uchar_mem, uchar_data_ptr, 0, 0, 0);

        // Mutate the data.
        expected_uchar++;
        expected_uint++;

        // Write the new values into persistent store.
        *uchar_data = expected_uchar;
        *uint_data = expected_uint;
        status = clSetKernelArg(writer,0,sizeof(cl_uchar),uchar_data); test_error_ret(status,"set arg",status);
        status = clSetKernelArg(writer,1,sizeof(cl_uint),uint_data); test_error_ret(status,"set arg",status);
        status = clEnqueueNDRangeKernel(queue,writer,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue writer",status);
        status = clFinish(queue); test_error_ret(status,"finish",status);
    }

    if ( CL_SUCCESS == err ) { log_info("OK\n"); FLUSH; }
    align_free(uchar_data);
    align_free(uint_data);
    return err;
}


////////////////////
// Global functions


// Test support for variables at program scope. Miscellaneous
int test_progvar_prog_scope_misc(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t max_size = 0;
    size_t pref_size = 0;

    cl_int err = CL_SUCCESS;

    err = l_get_device_info( device, &max_size, &pref_size );
    err |= l_build_type_table( device );

    err |= l_capacity( device, context, queue, max_size );
    err |= l_user_type( device, context, queue, false );
    err |= l_user_type( device, context, queue, true );

    return err;
}


// Test support for variables at program scope. Unitialized data
int test_progvar_prog_scope_uninit(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t max_size = 0;
    size_t pref_size = 0;

    cl_int err = CL_SUCCESS;

    err = l_get_device_info( device, &max_size, &pref_size );
    err |= l_build_type_table( device );

    err |= l_write_read( device, context, queue );

    return err;
}

// Test support for variables at program scope. Initialized data.
int test_progvar_prog_scope_init(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t max_size = 0;
    size_t pref_size = 0;

    cl_int err = CL_SUCCESS;

    err = l_get_device_info( device, &max_size, &pref_size );
    err |= l_build_type_table( device );

    err |= l_init_write_read( device, context, queue );

    return err;
}


// A simple test for support of static variables inside a kernel.
int test_progvar_func_scope(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t max_size = 0;
    size_t pref_size = 0;

    cl_int err = CL_SUCCESS;

    // Deliberately have two variables with the same name but in different
    // scopes.
    // Also, use a large initialized structure in both cases.
    const char prog_src[] =
        "typedef struct { char c; int16 i; } mystruct_t;\n"
        "kernel void test_bump( global int* value, int which ) {\n"
        "  if ( which ) {\n"
        // Explicit address space.
        // Last element set to 0
        "     static global mystruct_t persistent = {'a',(int16)(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0) };\n"
        "     *value = persistent.i.sf++;\n"
        "  } else {\n"
        // Implicitly global
        // Last element set to 100
        "     static mystruct_t persistent = {'b',(int16)(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,100) };\n"
        "     *value = persistent.i.sf++;\n"
        "  }\n"
        "}\n";

    StringTable ksrc;
    ksrc.add( prog_src );

    int status = CL_SUCCESS;
    clProgramWrapper program;
    clKernelWrapper test_bump;

    status = create_single_kernel_helper_with_build_options(context, &program, &test_bump, ksrc.num_str(), ksrc.strs(), "test_bump", OPTIONS);
    test_error_ret(status, "Failed to create program for function static variable test", status);

    // Check size query.
    size_t used_bytes = 0;
    status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, sizeof(used_bytes), &used_bytes, 0 );
    test_error_ret(status,"Failed to query global variable total size",status);
    size_t expected_size = 2 * sizeof(cl_int); // Two ints.
    if ( used_bytes < expected_size ) {
        log_error("Error: program query for global variable total size query failed: Expected at least %llu but got %llu\n", (unsigned long long)expected_size, (unsigned long long)used_bytes );
        err |= 1;
    }

    // Prepare the data.
    cl_int counter_value = 0;
    clMemWrapper counter_value_mem( clCreateBuffer( context, CL_MEM_USE_HOST_PTR, sizeof(counter_value), &counter_value, &status ) );
    test_error_ret(status,"Failed to allocate counter query buffer",status);

    status = clSetKernelArg(test_bump,0,sizeof(cl_mem),&counter_value_mem); test_error_ret(status,"set arg",status);

    // Go a few rounds, alternating between the two counters in the kernel.

    // Same as initial values in kernel.
    // But "true" which increments the 0-based counter, and "false" which
    // increments the 100-based counter.
    cl_int expected_counter[2] = { 100, 0 };

    const size_t one = 1;
    for ( int iround = 0; iround < 5 ; iround++ ) { // Must go at least twice around
        for ( int iwhich = 0; iwhich < 2 ; iwhich++ ) { // Cover both counters
            status = clSetKernelArg(test_bump,1,sizeof(iwhich),&iwhich); test_error_ret(status,"set arg",status);
            status = clEnqueueNDRangeKernel(queue,test_bump,1,0,&one,0,0,0,0); test_error_ret(status,"enqueue test_bump",status);
            status = clFinish(queue); test_error_ret(status,"finish",status);

            cl_uchar *counter_value_ptr = (cl_uchar *)clEnqueueMapBuffer(queue, counter_value_mem, CL_TRUE, CL_MAP_READ, 0, sizeof(counter_value), 0, 0, 0, 0);

            if ( counter_value != expected_counter[iwhich] ) {
                log_error("Error: Round %d on counter %d: Expected %d but got %d\n",
                        iround, iwhich, expected_counter[iwhich], counter_value );
                err |= 1;
            }
            expected_counter[iwhich]++; // Emulate behaviour of the kernel.

            clEnqueueUnmapMemObject(queue, counter_value_mem, counter_value_ptr, 0, 0, 0);
        }
    }

    if ( CL_SUCCESS == err ) { log_info("OK\n"); FLUSH; }

    return err;
}
