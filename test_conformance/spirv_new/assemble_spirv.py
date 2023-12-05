#!/usr/bin/env python3

#####################################################################
# Copyright (c) 2020-2023 The Khronos Group Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################

"""Assembles the SPIR-V assembly files used by spirv_new into binaries,
   and validates them using spirv-val.  Either run this from the parent
   of the spirv_asm directory, or pass the --source-dir and --output-dir
   options to specify the locations of the assembly files and the
   binaries to be generated.
"""

import argparse
import glob
import os
import subprocess
import sys
from textwrap import wrap

# sub-directories for specific SPIR-V environments
spirv_envs = [
    '', # all files in the root directory are considered SPIR-V 1.0
    'spv1.1',
    'spv1.2',
    'spv1.3',
    'spv1.4',
    'spv1.5',
    'spv1.6',
]

def fatal(message):
    """Print an error message and exit with a non-zero status, to
       indicate failure.
    """
    print(message)
    sys.exit(1)


def assemble_spirv(asm_dir, bin_dir, spirv_as, spirv_env, verbose):
    """Assemble SPIR-V source into binaries."""

    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    assembly_failures = False

    for asm_file_path in glob.glob(os.path.join(asm_dir, '*.spvasm*')):
        asm_file = os.path.basename(asm_file_path)
        if os.path.isfile(asm_file_path):
            if verbose:
                print(' Assembling {}'.format(asm_file))

            asm_file_root, asm_file_ext = os.path.splitext(asm_file)
            bin_file = asm_file_root + asm_file_ext.replace('asm', '')
            bin_file_path = os.path.join(bin_dir, bin_file)

            command = '"{}" --target-env "{}" "{}" -o "{}"'.format(
                spirv_as, spirv_env, asm_file_path, bin_file_path)
            if subprocess.call(command, shell=True) != 0:
                assembly_failures = True
                print('ERROR: Failure assembling {}: '
                      'see above output.'.format(
                          asm_file))
                print()

    if assembly_failures:
        fatal('\n'.join(wrap(
            'ERROR: Assembly failure(s) occurred.  See above for error '
            'messages from the assembler, if any.')))


def validate_spirv(bin_dir, spirv_val, spirv_env, verbose):
    """Validates SPIR-V binaries.  Ignores known failures."""

    validation_failures = False

    for bin_file_path in glob.glob(os.path.join(bin_dir, '*.spv*')):
        bin_file = os.path.basename(bin_file_path)
        if os.path.isfile(bin_file_path):
            if verbose:
                print(' Validating {}'.format(bin_file))

            command = '"{}" --target-env "{}" "{}"'.format(
                spirv_val, spirv_env, bin_file_path)
            if subprocess.call(command, shell=True) != 0:
                print('ERROR: Failure validating {}: '
                      'see above output.'.format(
                          bin_file))
                validation_failures = True
                print()

    if validation_failures:
        fatal('ERROR: Validation failure(s) found.  '
              'See above for validation output.')


def parse_args():
    """Parse the command-line arguments."""

    argparse_kwargs = (
        {'allow_abbrev': False} if sys.version_info >= (3, 5) else {})
    argparse_kwargs['description'] = (
        '''Assembles the SPIR-V assembly files used by spirv_new into
           binaries, and validates them using spirv-val.  Either run this
           from the parent of the spirv_asm directory, or pass the
           --source-dir and --output-dir options to specify the locations
           the assembly files and the binaries to be generated.
        ''')
    parser = argparse.ArgumentParser(**argparse_kwargs)
    parser.add_argument('-s', '--source-dir', metavar='DIR',
                        default='spirv_asm',
                        help='''specifies the directory containing SPIR-V
                                assembly files''')
    parser.add_argument('-o', '--output-dir', metavar='DIR',
                        default='spirv_bin',
                        help='''specifies the directory in which to
                                output SPIR-V binary files''')
    parser.add_argument('-a', '--assembler', metavar='PROGRAM',
                        default='spirv-as',
                        help='''specifies the program to use for assembly
                                of SPIR-V, defaults to spirv-as''')
    parser.add_argument('-l', '--validator', metavar='PROGRAM',
                        default='spirv-val',
                        help='''specifies the program to use for validation
                                of SPIR-V, defaults to spirv-val''')
    parser.add_argument('-k', '--skip-validation', action='store_true',
                        default=False,
                        help='skips validation of the genareted SPIR-V')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='''enable verbose output (i.e. prints the
                                name of each SPIR-V assembly file or
                                binary as it is assembled or validated)
                             ''')
    return parser.parse_args()


def main():
    """Main function.  Assembles and validates SPIR-V."""

    args = parse_args()

    for subdir in spirv_envs:
        src_dir = os.path.join(args.source_dir, subdir)
        out_dir = os.path.join(args.output_dir, subdir)
        spirv_env = 'spv1.0' if subdir == '' else subdir
        print('Assembling SPIR-V source into binaries for target {}...'.
              format(spirv_env))
        assemble_spirv(src_dir, out_dir, args.assembler,
                    spirv_env, args.verbose)
        print('Finished assembling SPIR-V binaries.')
        print()

        if args.skip_validation:
            print('Skipping validation of SPIR-V binaries as requested.')
        else:
            print('Validating SPIR-V binaries for target {}...'.
                  format(spirv_env))
            validate_spirv(out_dir, args.validator,
                    spirv_env, args.verbose)
            print('All SPIR-V binaries validated successfully.')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
