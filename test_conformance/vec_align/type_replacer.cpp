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
#include <string.h>
#if !defined(_MSC_VER)
#include <stdint.h>
#endif // !_MSC_VER

size_t doReplace(char * dest, size_t destLength, const char * source,
          const char * stringToReplace1,  const char * replaceWith1,
          const char * stringToReplace2, const char * replaceWith2)
{
    size_t copyCount = 0;
    const char * sourcePtr = source;
    char * destPtr = dest;
    const char * ptr1;
    const char * ptr2;
    size_t nJump;
    size_t len1, len2;
    size_t lenReplace1, lenReplace2;
    len1 = strlen(stringToReplace1);
    len2 = strlen(stringToReplace2);
    lenReplace1 = strlen(replaceWith1);
    lenReplace2 = strlen(replaceWith2);
    for(;copyCount < destLength && *sourcePtr; )
    {
        ptr1 = strstr(sourcePtr, stringToReplace1);
        ptr2 = strstr(sourcePtr, stringToReplace2);
        if(ptr1 != NULL && (ptr2 == NULL || ptr2 > ptr1))
        {
            nJump = ptr1-sourcePtr;
            if(((uintptr_t)ptr1-(uintptr_t)sourcePtr) > destLength-copyCount) { return -1; }
            copyCount += nJump;
            strncpy(destPtr, sourcePtr, nJump);
            destPtr += nJump;
            sourcePtr += nJump + len1;
            strcpy(destPtr, replaceWith1);
            destPtr += lenReplace1;
        }
        else if(ptr2 != NULL && (ptr1 == NULL || ptr1 >= ptr2))
        {
            nJump = ptr2-sourcePtr;
            if(nJump > destLength-copyCount) { return -2; }
            copyCount += nJump;
            strncpy(destPtr, sourcePtr, nJump);
            destPtr += nJump;
            sourcePtr += nJump + len2;
            strcpy(destPtr, replaceWith2);
            destPtr += lenReplace2;
        }
        else
        {
            nJump = strlen(sourcePtr);
            if(nJump > destLength-copyCount) { return -3; }
            copyCount += nJump;
            strcpy(destPtr, sourcePtr);
            destPtr += nJump;
            sourcePtr += nJump;
        }
    }
    *destPtr = '\0';
    return copyCount;
}

size_t doSingleReplace(char * dest, size_t destLength, const char * source,
               const char * stringToReplace, const char * replaceWith)
{
    size_t copyCount = 0;
    const char * sourcePtr = source;
    char * destPtr = dest;
    const char * ptr;
    size_t nJump;
    size_t len;
    size_t lenReplace;
    len = strlen(stringToReplace);
    lenReplace = strlen(replaceWith);
    for(;copyCount < destLength && *sourcePtr; )
    {
        ptr = strstr(sourcePtr, stringToReplace);
        if(ptr != NULL)
        {
            nJump = ptr-sourcePtr;
            if(((uintptr_t)ptr-(uintptr_t)sourcePtr) > destLength-copyCount) { return -1; }
            copyCount += nJump;
            strncpy(destPtr, sourcePtr, nJump);
            destPtr += nJump;
            sourcePtr += nJump + len;
            strcpy(destPtr, replaceWith);
            destPtr += lenReplace;
        }
        else
        {
            nJump = strlen(sourcePtr);
            if(nJump > destLength-copyCount) { return -3; }
            copyCount += nJump;
            strcpy(destPtr, sourcePtr);
            destPtr += nJump;
            sourcePtr += nJump;
        }
    }
    *destPtr = '\0';
    return copyCount;
}
