/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */




/*
 * Based on "Designing efficient sorting algorithms for manycore GPUs"
 * by Nadathur Satish, Mark Harris, and Michael Garland
 * http://mgarland.org/files/papers/gpusort-ipdps09.pdf
 *
 * Victor Podlozhnyuk 09/24/2009
 */



#include <assert.h>
#include <helper_cuda.h>
#include "mergeSort_common.h"



////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
static inline __host__ __device__ uint iDivUp(uint a, uint b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ uint getSampleCount(uint dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(uint) * 8)
static inline __device__ uint nextPowerOfTwo(uint x)
{
    /*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    */
    return 1U << (W - __clz(x - 1));
}

static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride, uint sortDir)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride, uint sortDir)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}



////////////////////////////////////////////////////////////////////////////////
// Bottom-level merge sort (binary search-based)
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> __global__ void mergeSortSharedKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength
)
{
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint stride = 1; stride < arrayLength; stride <<= 1)
    {
        uint     lPos = threadIdx.x & (stride - 1);
        uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
        uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint valA = baseVal[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive(keyA, baseKey + stride, stride, stride, sortDir) + lPos;
        uint posB = binarySearchInclusive(keyB, baseKey +      0, stride, stride, sortDir) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

static void mergeSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint sortDir
)
{
    if (arrayLength < 2)
    {
        return;
    }

    assert(SHARED_SIZE_LIMIT % arrayLength == 0);
    assert(((batchSize * arrayLength) % SHARED_SIZE_LIMIT) == 0);
    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    if (sortDir)
    {
        mergeSortSharedKernel<1U><<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
        getLastCudaError("mergeSortShared<1><<<>>> failed\n");
    }
    else
    {
        mergeSortSharedKernel<0U><<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
        getLastCudaError("mergeSortShared<0><<<>>> failed\n");
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> __global__ void generateSampleRanksKernel(
    uint *d_RanksA,
    uint *d_RanksB,
    uint *d_SrcKey,
    uint stride,
    uint N,
    uint threadCount
)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint  segmentSamplesA = getSampleCount(segmentElementsA);
    const uint  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive(
                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB), sortDir
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive(
                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA), sortDir
                                                 );
    }
}

static void generateSampleRanks(
    uint *d_RanksA,
    uint *d_RanksB,
    uint *d_SrcKey,
    uint stride,
    uint N,
    uint sortDir
)
{
    uint lastSegmentElements = N % (2 * stride);
    uint         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    if (sortDir)
    {
        generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        getLastCudaError("generateSampleRanksKernel<1U><<<>>> failed\n");
    }
    else
    {
        generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256>>>(d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
        getLastCudaError("generateSampleRanksKernel<0U><<<>>> failed\n");
    }
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
__global__ void mergeRanksAndIndicesKernel(
    uint *d_Limits,
    uint *d_Ranks,
    uint stride,
    uint N,
    uint threadCount
)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint  segmentSamplesA = getSampleCount(segmentElementsA);
    const uint  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        uint dstPos = binarySearchExclusive(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB), 1U) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }

    if (i < segmentSamplesB)
    {
        uint dstPos = binarySearchInclusive(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA), 1U) + i;
        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
    }
}

static void mergeRanksAndIndices(
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint *d_RanksA,
    uint *d_RanksB,
    uint stride,
    uint N
)
{
    uint lastSegmentElements = N % (2 * stride);
    uint         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsA,
        d_RanksA,
        stride,
        N,
        threadCount
    );
    getLastCudaError("mergeRanksAndIndicesKernel(A)<<<>>> failed\n");

    mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
        d_LimitsB,
        d_RanksB,
        stride,
        N,
        threadCount
    );
    getLastCudaError("mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> inline __device__ void merge(
    uint *dstKey,
    uint *dstVal,
    uint *srcAKey,
    uint *srcAVal,
    uint *srcBKey,
    uint *srcBVal,
    uint lenA,
    uint nPowTwoLenA,
    uint lenB,
    uint nPowTwoLenB
)
{
    uint keyA, valA, keyB, valB, dstPosA, dstPosB;

    if (threadIdx.x < lenA)
    {
        keyA = srcAKey[threadIdx.x];
        valA = srcAVal[threadIdx.x];
        dstPosA = binarySearchExclusive(keyA, srcBKey, lenB, nPowTwoLenB, sortDir) + threadIdx.x;
    }

    if (threadIdx.x < lenB)
    {
        keyB = srcBKey[threadIdx.x];
        valB = srcBVal[threadIdx.x];
        dstPosB = binarySearchInclusive(keyB, srcAKey, lenA, nPowTwoLenA, sortDir) + threadIdx.x;
    }

    __syncthreads();

    if (threadIdx.x < lenA)
    {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB)
    {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}

template<uint sortDir> __global__ void mergeElementaryIntervalsKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N
)
{
    __shared__ uint s_key[2 * SAMPLE_STRIDE];
    __shared__ uint s_val[2 * SAMPLE_STRIDE];

    const uint   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint  segmentSamplesA = getSampleCount(segmentElementsA);
        uint  segmentSamplesB = getSampleCount(segmentElementsB);
        uint   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    //Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
    merge<sortDir>(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE
    );

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}

static void mergeElementaryIntervals(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N,
    uint sortDir
)
{
    uint lastSegmentElements = N % (2 * stride);
    uint          mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

    if (sortDir)
    {
        mergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        getLastCudaError("mergeElementaryIntervalsKernel<1> failed\n");
    }
    else
    {
        mergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
            d_DstKey,
            d_DstVal,
            d_SrcKey,
            d_SrcVal,
            d_LimitsA,
            d_LimitsB,
            stride,
            N
        );
        getLastCudaError("mergeElementaryIntervalsKernel<0> failed\n");
    }
}



extern "C" void bitonicSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint sortDir
);

extern "C" void bitonicMergeElementaryIntervals(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N,
    uint sortDir
);



static uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
static const uint MAX_SAMPLE_COUNT = 32768 * 16; //changed by Tony

extern "C" void initMergeSort(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_RanksA,  MAX_SAMPLE_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_RanksB,  MAX_SAMPLE_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(uint)));
}

extern "C" void closeMergeSort(void)
{
    checkCudaErrors(cudaFree(d_RanksA));
    checkCudaErrors(cudaFree(d_RanksB));
    checkCudaErrors(cudaFree(d_LimitsB));
    checkCudaErrors(cudaFree(d_LimitsA));
}
__global__ void k(uint *d_DstKey,
                  uint *d_DstVal,
                  uint *d_SrcKey,
                  uint *d_SrcVal,
                  uint tileSize,
                  uint N,
                  uint sortDir
){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId > N)
        return;

    int chunkSize = tileSize / blockDim.x;
    uint blockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    uint firstElIndex = 2 * tileSize * blockIndex;

    /*
     * binarySearchInclusive(uint val, uint *data, uint L, uint stride)
     * binarySearchExclusive(uint val, uint *data, uint L, uint stride)
     */



    int i = 0;
    for(; i < chunkSize; ++i){
        int leftIndex  = i + chunkSize * threadIdx.x;
        int rightIndex = leftIndex + tileSize;
        uint leftElement  = d_SrcKey[leftIndex];
        uint rightElement = d_SrcKey[rightIndex];

        uint leftRank  = binarySearchInclusive(leftElement,  d_SrcKey + firstElIndex,              tileSize, tileSize, sortDir) + binarySearchExclusive(leftElement,  d_SrcKey + (firstElIndex + tileSize), tileSize, tileSize, sortDir);
        uint rightRank = binarySearchInclusive(rightElement, d_SrcKey + (firstElIndex + tileSize), tileSize, tileSize, sortDir) + binarySearchExclusive(rightElement, d_SrcKey + firstElIndex,              tileSize, tileSize, sortDir);

        d_DstKey[leftRank] = blockIndex;//leftElement;
        d_DstVal[leftRank] = d_SrcVal[leftIndex];
        d_DstKey[rightRank] = blockIndex;//rightElement;
        d_DstVal[rightRank] = d_SrcVal[rightIndex];
    }


    /*
    if(threadId == 0){
        uint i = 0;
        for(; i < N; ++i){
            d_DstKey[i] = d_SrcKey[i];
            d_DstVal[i] = d_SrcVal[i];
        }
    }
    */
}



extern "C" void mergeSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_BufKey,
    uint *d_BufVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint N,
    uint sortDir
) {

    //uint  blockCount = batchSize;
    //uint threadCount = SHARED_SIZE_LIMIT / 2;


    uint tileSize = SHARED_SIZE_LIMIT;

    uint * ikey, *ival, *okey, *oval;

    ikey = d_SrcKey;
    ival = d_SrcVal;
    okey = d_DstKey;
    oval = d_DstVal;
    int iterationNumer = 0;

    for (; tileSize < N; tileSize *= 2, ++iterationNumer) {
        k <<< tileSize,  N / (2 * tileSize) >>> (okey, oval, ikey, ival, tileSize, N, sortDir);

        uint *t;
        t = ikey;
        ikey = okey;
        okey = t;
        t = ival;
        ival = oval;
        oval = t;
    }

    // if the last iteration didn't store the results in the output arrays, fix it.
    if(okey != d_DstKey) {
        checkCudaErrors(cudaMemcpy(d_DstKey, ikey, N * sizeof(uint), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_DstVal, ival, N * sizeof(uint), cudaMemcpyDeviceToDevice));
    }
}


    /*
     *
     uint stageCount = 0;

    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);

    uint *ikey, *ival, *okey, *oval;

    if (stageCount & 1)
    {
        ikey = d_BufKey;
        ival = d_BufVal;
        okey = d_DstKey;
        oval = d_DstVal;
    }
    else
    {
        ikey = d_DstKey;
        ival = d_DstVal;
        okey = d_BufKey;
        oval = d_BufVal;
    }

    assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
    assert(N % SHARED_SIZE_LIMIT == 0);
    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir);

    for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1)
    {
        uint lastSegmentElements = N % (2 * stride);

        //Find sample ranks and prepare for limiters merge
        generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

        //Merge ranks and indices
        mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

        //Merge elementary intervals
        mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, N, sortDir);

        if (lastSegmentElements <= stride)
        {
            //Last merge segment consists of a single array which just needs to be passed through
            checkCudaErrors(cudaMemcpy(okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements), lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(oval + (N - lastSegmentElements), ival + (N - lastSegmentElements), lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
        }

        uint *t;
        t = ikey;
        ikey = okey;
        okey = t;
        t = ival;
        ival = oval;
        oval = t;
    }
*/


