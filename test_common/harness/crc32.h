/******************************************************************
Copyright (c) 2016 The Khronos Group Inc.
All Rights Reserved.  This code is protected by copyright laws and
contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed
in whole or in part to third parties, and may not be reproduced, republished,
distributed, transmitted, displayed, broadcast or otherwise exploited in any
manner without the express prior written permission of Khronos Group.

The receipt or possession of this code does not convey any rights to
reproduce, disclose, or distribute its contents, or to
manufacture, use, or sell anything that it may describe, in whole
or in part other than under the terms of the Khronos Adopters
Agreement or Khronos Conformance Test Source License Agreement as
executed between Khronos and the recipient.
******************************************************************/

#ifndef CRC32_H_
#define CRC32_H_

#include <stdint.h>
#include <stddef.h>

uint32_t crc32(const void *buf, size_t size);

#endif
