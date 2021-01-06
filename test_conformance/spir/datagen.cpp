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
#include "exceptions.h"
#include "datagen.h"

RandomGenerator gRG;

size_t WorkSizeInfo::getGlobalWorkSize() const
{
    switch( work_dim )
    {
    case 1: return global_work_size[0];
    case 2: return global_work_size[0] * global_work_size[1];
    case 3: return global_work_size[0] * global_work_size[1] * global_work_size[2];
    default:
        throw Exceptions::TestError("wrong work dimention\n");
    }
}

/*
 * DataGenerator
 */

DataGenerator* DataGenerator::Instance = NULL;

DataGenerator* DataGenerator::getInstance()
{
    if (!Instance)
        Instance = new DataGenerator();

    return Instance;
}

DataGenerator::DataGenerator()
{
    #define TYPE_HNDL( type, isBuffer, base_element_size, vector_size, min_value, max_value, Generator) \
    assert(m_argGenerators.find(type) == m_argGenerators.end()) ; \
    m_argGenerators[type] = new Generator( isBuffer, vector_size, min_value, max_value);
    #include "typeinfo.h"
    #undef TYPE_HNDL
}

DataGenerator::~DataGenerator()
{
    ArgGeneratorsMap::iterator i = m_argGenerators.begin();
    ArgGeneratorsMap::iterator e = m_argGenerators.end();

    for(; i != e; ++i)
    {
        delete i->second;
    }
}

KernelArgGenerator* DataGenerator::getArgGenerator( const KernelArgInfo& argInfo )
{
    //try to match the full type first
    ArgGeneratorsMap::iterator i = m_argGenerators.find(argInfo.getTypeName());
    ArgGeneratorsMap::iterator e = m_argGenerators.end();

    if( i != e )
    {
        return i->second;
    }
    // search for the proper prefix of the type
    for(i = m_argGenerators.begin(); i != e; ++i)
    {
        if( 0 == argInfo.getTypeName().find(i->first))
        {
            return i->second;
        }
    }
    throw Exceptions::TestError(std::string("Can't find the generator for the type ")
      + argInfo.getTypeName() + " for argument " + argInfo.getName() + "\n");
}

void DataGenerator::setArgGenerator(const KernelArgInfo& argInfo,
                                    KernelArgGenerator* pGen)
{
    m_argGenerators[argInfo.getTypeName()] = pGen;
}

size_t get_random_int32(int low, int high, MTdata d)
{
  int v = genrand_int32(d);

  assert(low <= high && "Invalid random number range specified");
  size_t range = high - low;

  return (range) ? low + ((v - low) % range) : low;
}

/*
 * KernelArgGeneratorSampler
 */
KernelArgGeneratorSampler::KernelArgGeneratorSampler(bool isBuffer,
                                                     size_t vectorSize,
                                                     int minValue,
                                                     int maxValue) {
  initToDefaults();
}

void KernelArgGeneratorSampler::initToDefaults() {
  m_normalized = false;
  m_addressingMode = CL_ADDRESS_NONE;
  m_filterMode = CL_FILTER_NEAREST;
}

KernelArgGeneratorSampler::KernelArgGeneratorSampler() {
  initToDefaults();

}

void KernelArgGeneratorSampler::setNormalized(cl_bool isNormalized)
{
    m_normalized = isNormalized;
}

void KernelArgGeneratorSampler::setAddressingMode(cl_addressing_mode mode)
{
    m_addressingMode = mode;
}

void KernelArgGeneratorSampler::setFiterMode(cl_filter_mode mode)
{
    m_filterMode = mode;
}


/*
 * SamplerValuesGenerator.
 */

/*
 * Static fields initialization.
 */
cl_bool SamplerValuesGenerator::coordNormalizations[] = {CL_TRUE, CL_FALSE};

cl_filter_mode SamplerValuesGenerator::filterModes[]  = {
    CL_FILTER_NEAREST,
    CL_FILTER_LINEAR
};

cl_addressing_mode SamplerValuesGenerator::addressingModes[] = {
    CL_ADDRESS_NONE,
    CL_ADDRESS_CLAMP,
    CL_ADDRESS_CLAMP_TO_EDGE,
    CL_ADDRESS_REPEAT,
    CL_ADDRESS_MIRRORED_REPEAT
};

const size_t NUM_NORM_MODES =
    sizeof(SamplerValuesGenerator::coordNormalizations)/sizeof(cl_bool);

const size_t NUM_FILTER_MODES =
    sizeof(SamplerValuesGenerator::filterModes)/sizeof(cl_filter_mode);

const size_t NUM_ADDR_MODES =
    sizeof(SamplerValuesGenerator::addressingModes)/sizeof(cl_addressing_mode);

SamplerValuesGenerator::iterator SamplerValuesGenerator::end()
{
    return iterator(NUM_NORM_MODES-1, NUM_FILTER_MODES-1, NUM_ADDR_MODES-1);
}

/*
 * A constructor for generating an 'end iterator'.
 */
SamplerValuesGenerator::iterator::iterator(size_t norm, size_t filter,
                                           size_t addressing):
    m_normIndex(norm), m_filterIndex(filter), m_addressingModeIndex(addressing){}

/*
 * A constructor for generating a 'begin iterator'.
 */
SamplerValuesGenerator::iterator::iterator():
    m_normIndex(0), m_filterIndex(0), m_addressingModeIndex(0){}

SamplerValuesGenerator::iterator& SamplerValuesGenerator::iterator::operator ++()
{
    if (incrementIndex(m_normIndex, NUM_NORM_MODES)) return *this;
    if (incrementIndex(m_filterIndex, NUM_FILTER_MODES)) return *this;
    if (incrementIndex(m_addressingModeIndex, NUM_ADDR_MODES)) return *this;

    assert(false && "incrementing end iterator!");
    return *this;
}

bool SamplerValuesGenerator::iterator::incrementIndex(size_t &i,
                                                      const size_t limit)
{
    i = (i+1) % limit;
    return i != 0;
}

bool SamplerValuesGenerator::iterator::operator == (const iterator& other) const
{
    return m_normIndex == other.m_normIndex &&
         m_filterIndex == other.m_filterIndex &&
         m_addressingModeIndex == other.m_addressingModeIndex;
}

bool SamplerValuesGenerator::iterator::operator != (const iterator& other) const
{
    return !(*this == other);
}

cl_bool SamplerValuesGenerator::iterator::getNormalized() const
{
    assert(m_normIndex < NUM_NORM_MODES && "illegal index");
    return coordNormalizations[m_normIndex];
}

cl_filter_mode SamplerValuesGenerator::iterator::getFilterMode() const
{
    assert(m_filterIndex < NUM_FILTER_MODES && "illegal index");
    return filterModes[m_filterIndex];
}

cl_addressing_mode SamplerValuesGenerator::iterator::getAddressingMode() const
{
    assert(m_addressingModeIndex < NUM_ADDR_MODES && "illegal index");
    return addressingModes[m_addressingModeIndex];
}

unsigned SamplerValuesGenerator::iterator::toBitmap() const
{
    unsigned norm, filter, addressingModes;
    switch (getNormalized())
    {
    case CL_TRUE:
        norm = 8;
        break;
    case CL_FALSE:
        norm = 0;
        break;
    default:
    assert(0 && "invalid normalize value");
    }

    switch (getFilterMode())
    {
    case CL_FILTER_NEAREST:
        filter = 0;
        break;
    case CL_FILTER_LINEAR:
        filter = 16;
        break;
    default:
    assert(0 && "invalid filter value");
    }

    switch(getAddressingMode())
    {
    case CL_ADDRESS_NONE:
        addressingModes = 0;
        break;
    case CL_ADDRESS_CLAMP:
        addressingModes = 1;
        break;
    case CL_ADDRESS_CLAMP_TO_EDGE:
        addressingModes = 2;
        break;
    case CL_ADDRESS_REPEAT:
        addressingModes = 3;
        break;
    case CL_ADDRESS_MIRRORED_REPEAT:
        addressingModes = 4;
        break;
    default:
    assert(0 && "invalid filter value");
    }

    return norm | filter | addressingModes;
}

std::string SamplerValuesGenerator::iterator::toString() const
{
    std::string ret("(");

    switch (getNormalized())
    {
    case CL_TRUE:
        ret.append("Normalized | ");
        break;
    case CL_FALSE:
        ret.append("Not Normalized | ");
        break;
    default:
    assert(0 && "invalid normalize value");
    }

    switch (getFilterMode())
    {
    case CL_FILTER_NEAREST:
        ret.append("Filter Nearest | ");
        break;
    case CL_FILTER_LINEAR:
        ret.append("Filter Linear | ");
        break;
    default:
    assert(0 && "invalid filter value");
    }

    switch(getAddressingMode())
    {
    case CL_ADDRESS_NONE:
        ret.append("Address None");
        break;
    case CL_ADDRESS_CLAMP:
        ret.append("Address clamp");
        break;
    case CL_ADDRESS_CLAMP_TO_EDGE:
        ret.append("Address clamp to edge");
        break;
    case CL_ADDRESS_REPEAT:
        ret.append("Address repeat");
        break;
    case CL_ADDRESS_MIRRORED_REPEAT:
        ret.append("Address mirrored repeat");
        break;
    default:
    assert(0 && "invalid filter value");
    }

    ret.append(")");
    return ret;
}

/*
 * ImageValuesGenerator.
 */

/*
 * Static fields initialization.
 */
const char* ImageValuesGenerator::imageTypes[] = {
    "image1d_array_float",
    "image1d_array_int",
    "image1d_array_uint",
    "image1d_buffer_float",
    "image1d_buffer_int",
    "image1d_buffer_uint",
    "image1d_float",
    "image1d_int",
    "image1d_uint",
    "image2d_array_float",
    "image2d_array_int",
    "image2d_array_uint",
    "image2d_float",
    "image2d_int",
    "image2d_uint",
    "image3d_float",
    "image3d_int",
    "image3d_uint"
};

cl_channel_order ImageValuesGenerator::channelOrders[] = {
    CL_A,
    CL_R,
    CL_Rx,
    CL_RG,
    CL_RGx,
    CL_RA,
    CL_RGB,
    CL_RGBx,
    CL_RGBA,
    CL_ARGB,
    CL_BGRA,
    CL_INTENSITY,
    CL_LUMINANCE,
    CL_DEPTH,
    CL_DEPTH_STENCIL
};

const size_t NUM_CHANNEL_ORDERS = sizeof(ImageValuesGenerator::channelOrders)/sizeof(ImageValuesGenerator::channelOrders[0]);
const size_t NUM_IMG_TYS = sizeof(ImageValuesGenerator::imageTypes)/sizeof(ImageValuesGenerator::imageTypes[0]);

ImageValuesGenerator::iterator ImageValuesGenerator::begin()
{
    return ImageValuesGenerator::iterator(this);
}

ImageValuesGenerator::iterator ImageValuesGenerator::end()
{
    return ImageValuesGenerator::iterator(0);
}
/*
 * Class Iterator
 */
ImageValuesGenerator::iterator::iterator(ImageValuesGenerator *pParent):
    m_parent(pParent), m_channelIndex(0), m_imgTyIndex(0)
{
}

/*
 * Initializes an 'end' iterator.
 */
ImageValuesGenerator::iterator::iterator(int):
    m_parent(NULL),
    m_channelIndex(NUM_CHANNEL_ORDERS),
    m_imgTyIndex(NUM_IMG_TYS) {}

ImageValuesGenerator::iterator& ImageValuesGenerator::iterator::operator ++()
{
    assert(m_channelIndex < NUM_CHANNEL_ORDERS && m_imgTyIndex < NUM_IMG_TYS &&
           "Attempt to increment an end iterator");

    ImageValuesGenerator::iterator endIter = iterator(0);
    // Incrementing untill we find the next legal combination, or we reach the
    // end value.
    while (incrementIndex(m_channelIndex,NUM_CHANNEL_ORDERS))
        if (isLegalCombination())
            return *this;

     // We have reach to this line because last increment caused an 'oveflow'
     // in data channel order index.
     if (incrementIndex(m_imgTyIndex, NUM_IMG_TYS))
        // In case this combination is not legal, we go on to the next legal
        // combo.
        return isLegalCombination() ? *this : ++(*this);

    *this = endIter;
    return *this;
}

bool ImageValuesGenerator::iterator::operator == (
    const ImageValuesGenerator::iterator& o) const
{
    return m_channelIndex == o.m_channelIndex &&
           m_imgTyIndex == o.m_imgTyIndex;
}

bool ImageValuesGenerator::iterator::operator != (
    const ImageValuesGenerator::iterator& o) const
{
    return !(*this == o);
}

std::string ImageValuesGenerator::iterator::getDataTypeName() const
{
    assert(m_imgTyIndex < NUM_IMG_TYS && "image type index is out of bound");

    std::string tyName(imageTypes[m_imgTyIndex]);
    // Find the last '_' and remove it (the suffix is _<channel type>).
    size_t pos = tyName.find_last_of('_');
    assert (std::string::npos != pos && "no under score in type name?");
    tyName = tyName.erase(0, pos+1);
    return tyName;
}

int ImageValuesGenerator::iterator::getOpenCLChannelOrder() const
{
    assert(m_channelIndex < NUM_CHANNEL_ORDERS && "channel index out of bound");
    return channelOrders[m_channelIndex];
}

int ImageValuesGenerator::iterator::getSPIRChannelOrder() const
{
    return getOpenCLChannelOrder();
}

std::string ImageValuesGenerator::iterator::getImageTypeName() const
{
    assert(m_imgTyIndex < NUM_IMG_TYS && "image type index is out of bound");

    std::string tyName = imageTypes[m_imgTyIndex];
    // Find the last '_' and remove it (the suffix is _<channel type>).
    size_t pos = tyName.find_last_of('_');
    assert (std::string::npos != pos && "no under score in type name?");
    tyName = tyName.erase(pos, tyName.size() - pos);

    return tyName;
}

std::string ImageValuesGenerator::iterator::getImageGeneratorName() const
{
    assert(m_imgTyIndex < NUM_IMG_TYS && "image type index is out of bound");
    return imageTypes[m_imgTyIndex];
}

std::string ImageValuesGenerator::iterator::getBaseImageGeneratorName() const
{
    assert(m_imgTyIndex < NUM_IMG_TYS && "image type index is out of bound");
    std::string tyName = getImageTypeName();
    tyName.append("_t");
    return tyName;
}

int ImageValuesGenerator::iterator::getDataType() const
{
    assert(m_imgTyIndex < NUM_IMG_TYS && "image type index is out of bound");
    std::string tyName = getDataTypeName();

    if ("int" == tyName)
       return SPIR_CLK_SIGNED_INT32;
    if ("uint" == tyName)
        return SPIR_CLK_UNSIGNED_INT32;
    if ("float" == tyName)
        return SPIR_CLK_FLOAT;
    assert (false && "unkown image data type");
    return -1;
}

std::string ImageValuesGenerator::iterator::toString() const
{
    if (*this == m_parent->end())
        return "End iterator";

    // Sanity.
    assert(m_imgTyIndex < NUM_IMG_TYS && "image type index is out of bound");
    assert(m_channelIndex < NUM_CHANNEL_ORDERS && "channel index out of bound");

    std::string str = imageTypes[m_imgTyIndex];
    str.append("_");

    switch (channelOrders[m_channelIndex])
    {
    case CL_R:
        str.append("cl_r");
        break;
    case CL_A:
        str.append("cl_a");
        break;
    case CL_RG:
        str.append("cl_rg");
        break;
    case CL_RA:
        str.append("cl_ra");
        break;
    case CL_RGB:
        str.append("cl_rgb");
        break;
    case CL_RGBA:
        str.append("cl_rgba");
        break;
    case CL_BGRA:
        str.append("cl_bgra");
        break;
    case CL_ARGB:
        str.append("cl_argb");
        break;
    case CL_INTENSITY:
        str.append("cl_intensity");
        break;
    case CL_LUMINANCE:
        str.append("cl_luminace");
        break;
    case CL_Rx:
        str.append("cl_Rx");
        break;
    case CL_RGx:
        str.append("cl_RGx");
        break;
    case CL_RGBx:
        str.append("cl_RGBx");
        break;
    case CL_DEPTH:
        str.append("cl_depth");
        break;
    case CL_DEPTH_STENCIL:
        str.append( "cl_depth_stencil");
        break;
    default:
        assert(false && "Invalid channel order");
        str.append("<invalid channel order>");
        break;
    }

    return str;
}

bool ImageValuesGenerator::iterator::incrementIndex(size_t& index,
                                                    size_t arrSize)
{
    index = (index + 1) % arrSize;
    return index != 0;
}

bool ImageValuesGenerator::iterator::isLegalCombination() const
{
    cl_channel_order corder = channelOrders[m_channelIndex];
    std::string strImgTy(imageTypes[m_imgTyIndex]);

    if (corder == CL_INTENSITY || corder == CL_LUMINANCE)
    {
        return getDataTypeName() == std::string("float");
    }

    if (corder == CL_DEPTH)
        return false;

    if (corder == CL_RGBx || corder == CL_RGB // Can only be applied for int unorms.
        || corder == CL_ARGB || corder == CL_BGRA) // Can only be applied for int8.
        return false;

    return true;
}

