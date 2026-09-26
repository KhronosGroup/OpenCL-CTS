#!/usr/bin/env python3

#####################################################################
# Copyright (c) 2025 The Khronos Group Inc. All Rights Reserved.
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

"""
Generates a file describing the SPIR-V extension dependencies or SPIR-V version
dependencies for a SPIR-V capability. This can be used to ensure that if support
for a SPIR-V capability is reported, the necessary SPIR-V extensions or SPIR-V
version is also supported.
"""

import argparse
import json

header_text = """\
//
// Copyright (c) 2025 The Khronos Group Inc.
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

// This file is generated from the SPIR-V JSON grammar file.
// Please do not edit it directly!
"""

def main():
    parser = argparse.ArgumentParser(description='Generate SPIR-V extension and version dependencies for SPIR-V capabilities')

    parser.add_argument('--grammar', metavar='<path>',
                        type=str, required=True,
                        help='input JSON grammar file')
    parser.add_argument('--output', metavar='<path>',
                        type=str, required=False,
                        help='output file path (default: stdout)')
    args = parser.parse_args()

    dependencies = {}
    capabilities = []
    with open(args.grammar) as json_file:
        grammar_json = json.loads(json_file.read())
        for operand_kind in grammar_json['operand_kinds']:
            if operand_kind['kind'] == 'Capability':
                for cap in operand_kind['enumerants']:
                    capname = cap['enumerant']
                    capabilities.append(capname)
                    dependencies[capname] = {}
                    dependencies[capname]['extensions'] = cap['extensions'] if 'extensions' in cap else []
                    dependencies[capname]['version'] = ("SPIR-V_" + cap['version']) if 'version' in cap and cap['version'] != 'None' else ""

    capabilities.sort()

    output = []
    output.append(header_text)
    output.append("// clang-format off")
    if False:
        for cap in capabilities:
            deps = dependencies[cap]
            extensions_str = ', '.join(f'"{ext}"' for ext in deps['extensions'])
            
            output.append('SPIRV_CAPABILITY_DEPENDENCIES( {}, {{{}}}, "{}" )'.format(
                cap, extensions_str, deps['version']))
    else:
        for cap in capabilities:
            deps = dependencies[cap]
            if deps['version'] != "":
                output.append('SPIRV_CAPABILITY_VERSION_DEPENDENCY( {}, "{}" )'.format(cap, deps['version']))
            for ext in deps['extensions']:
                output.append('SPIRV_CAPABILITY_EXTENSION_DEPENDENCY( {}, "{}" )'.format(cap, ext))
    output.append("// clang-format on")

    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write('\n'.join(output))
    else:
        print('\n'.join(output))

if __name__ == '__main__':
    main()
