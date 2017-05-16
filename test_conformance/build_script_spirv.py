# Script parameters:
# 1 - input file
# 2 - output file
# 3 - architecture: 32 or 64
# 4 - one of the strings: binary, source, spir_v
# 5 - OpenCL version: 12, 20
# 6 - build options

import os
import sys

if len(sys.argv)<5:
	print 'Usage: "build_script_spirv.py <input> <output> <arch> <output_type> <opencl_version> [build_options]"'
	exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
arch = sys.argv[3]
output_type = sys.argv[4]
ocl_version = sys.argv[5]
build_options = ''

if len(sys.argv) == 5:
	build_options = sys.argv[6]

if arch == '32':
	arch_string = ''
	spir_arch = '__i386__'
else:
	arch_string = '64'
	spir_arch = '__x86_64__'

if ocl_version == '20':
	oclc_version = '200'
	spir_version = '2.0'
else:
	oclc_version = '120'
	spir_version = '1.2'

command = '%LLVMPATH%\\bin\\clang.exe -cc1 -include headers\\opencl_SPIR-' + spir_version + '.h -cl-std=CL' + spir_version +' -D__OPENCL_C_VERSION__=' + oclc_version + ' -fno-validate-pch -D__OPENCL_VERSION__=' + oclc_version + ' -x cl -cl-kernel-arg-info -O0 -emit-llvm-bc -triple spir' + arch_string + '-unknown-unknown -D' + spir_arch + '  -Dcl_khr_3d_image_writes -Dcl_khr_byte_addressable_store -Dcl_khr_d3d10_sharing -Dcl_khr_d3d11_sharing -Dcl_khr_depth_images -Dcl_khr_dx9_media_sharing -Dcl_khr_fp64 -Dcl_khr_global_int32_base_atomics -Dcl_khr_global_int32_extended_atomics -Dcl_khr_gl_depth_images -Dcl_khr_gl_event -Dcl_khr_gl_msaa_sharing -Dcl_khr_gl_sharing -Dcl_khr_icd -Dcl_khr_image2d_from_buffer -Dcl_khr_local_int32_base_atomics -Dcl_khr_local_int32_extended_atomics -Dcl_khr_mipmap_image -Dcl_khr_mipmap_image_writes -Dcl_khr_fp16 ' + build_options + ' -Dcl_khr_spir ' + input_file + ' -o intermediate.spir'
os.system(command)
command = '%LLVMPATH%\\bin\\llvm-spirv.exe intermediate.spir -o ' + output_file
os.system(command)