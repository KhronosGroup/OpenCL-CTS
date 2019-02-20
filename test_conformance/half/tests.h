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
#ifndef TESTS_H
#define TESTS_H


int Test_vload_half( void );
int Test_vloada_half( void );
int Test_vstore_half( void );
int Test_vstorea_half( void );
int Test_vstore_half_rte( void );
int Test_vstorea_half_rte( void );
int Test_vstore_half_rtz( void );
int Test_vstorea_half_rtz( void );
int Test_vstore_half_rtp( void );
int Test_vstorea_half_rtp( void );
int Test_vstore_half_rtn( void );
int Test_vstorea_half_rtn( void );
int Test_roundTrip( void );

typedef cl_ushort (*f2h)( float );
typedef cl_ushort (*d2h)( double );
int Test_vStoreHalf_private( f2h referenceFunc, d2h referenceDoubleFunc, const char *roundName );
int Test_vStoreaHalf_private( f2h referenceFunc, d2h referenceDoubleFunc, const char *roundName );

#endif /* TESTS_H */


