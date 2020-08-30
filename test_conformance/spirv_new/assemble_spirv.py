#
# Copyright (c) 2017 The Khronos Group Inc.
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
#

import argparse
import os
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-srcdir', action='store', default='spirv_asm',
                        help='Source directory with .spvasm files')
    parser.add_argument('-dstdir', action='store', default='spirv_bin',
                        help='Destination directory for .spv files')

    args = parser.parse_args()

    if not os.path.exists(args.srcdir):
        print('error: directory ' + args.srcdir + ' does not exist!')
        sys.exit(-1)
    if not os.path.exists(args.dstdir):
        print('error: directory ' + args.srcdir + ' does not exist!')
        sys.exit(-1)

    print('Running "spirv-as" to assemble all files in: ' + args.srcdir + ':')

    commandtorun = 'spirv-as --target-env spv1.0'
    numberOfFiles = 0

    for fullfilename in os.listdir(args.srcdir):
        (file, ext) = os.path.splitext(fullfilename)
        #print('File is %s, ext is %s' % (file, ext))

        outext = ''
        if ext == '.spvasm32':
            outext = '.spv32'
        elif ext == '.spvasm64':
            outext = '.spv64'
        elif ext == '.spvasm':
            outext = '.spv'
        else:
            print('Unknown extension %s!  Skipping file %s.' % (ext, fullfilename))
            continue

        numberOfFiles = numberOfFiles + 1

        srcfilename = args.srcdir + '/' + fullfilename
        dstfilename = args.dstdir + '/' + file + outext

        print('Running: %s %s -o %s' % (commandtorun, srcfilename, dstfilename))
        subprocess.call(commandtorun.split() + [srcfilename, '-o', dstfilename])

    print('Assembled %d file(s).' % numberOfFiles)
