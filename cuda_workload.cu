#include "cuda_workload.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
void handleCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Retained for the next step: the reader does not launch this kernel yet.
[[maybe_unused]] __global__ void dotProducts(const float *blockOfVectors, const float *query,
                            float *scores, unsigned int vectorCount)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectorCount)
        return;

    float score = 0.0f;
    for (unsigned int d = 0; d < CudaWorkload::dimensions; ++d)
        score += blockOfVectors[i * CudaWorkload::dimensions + d] * query[d];
    scores[i] = score;
}
} // namespace

CudaWorkload::CudaWorkload(std::size_t blockSize) : blockSize_(blockSize)
{
    if (blockSize_ == 0 || blockSize_ % (dimensions * sizeof(float)) != 0)
        throw std::runtime_error("GPU block must contain whole 128-float vectors.");

    // A failed constructor does not run this object's destructor, so clean up
    // any allocations already obtained before propagating the exception.
    try
    {
        int deviceCount = 0;
        handleCuda(cudaGetDeviceCount(&deviceCount), "CUDA device query");
        if (deviceCount == 0)
            throw std::runtime_error("No CUDA-capable GPU is available.");
        std::fprintf(stderr, "CUDA devices available: %d\n", deviceCount);

        handleCuda(cudaMalloc(&gpuBuffer_, blockSize_), "CUDA allocation");
        std::fprintf(stderr, "Allocated GPU buffer: %zu bytes.\n", blockSize_);

        // A fixed all-ones query makes the first computation easy to verify.
        const std::vector<float> query(dimensions, 1.0f);
        const std::size_t queryBytes = query.size() * sizeof(float);
        const std::size_t scoreBytes = (blockSize_ / (dimensions * sizeof(float))) * sizeof(float);
        handleCuda(cudaMalloc(&gpuQuery_, queryBytes), "CUDA query allocation");
        handleCuda(cudaMalloc(&gpuScores_, scoreBytes), "CUDA scores allocation");
        handleCuda(cudaMemcpy(gpuQuery_, query.data(), queryBytes, cudaMemcpyHostToDevice),
                   "CUDA query copy");
        // The future kernel will overwrite every valid score; no initialization needed.
        std::fprintf(stderr, "Allocated query: %zu bytes; scores: %zu bytes.\n",
                     queryBytes, scoreBytes);
    }
    catch (...)
    {
        release();
        throw;
    }
}

CudaWorkload::~CudaWorkload() noexcept
{
    release();
}

void CudaWorkload::copyBlock(const float *source)
{
    if (gpuBuffer_ == nullptr)
        throw std::runtime_error("Cannot copy a block after GPU buffers were released.");
    handleCuda(cudaMemcpy(gpuBuffer_, source, blockSize_, cudaMemcpyHostToDevice),
               "CUDA host-to-device copy");
}

bool CudaWorkload::release() noexcept
{
    bool success = true;
    for (float **slot : {&gpuScores_, &gpuQuery_, &gpuBuffer_})
    {
        // Forget the pointer before the attempt so the destructor never retries
        // a failed explicit release whose resource state might be uncertain.
        float *allocation = std::exchange(*slot, nullptr);
        if (allocation == nullptr)
            continue;
        const cudaError_t status = cudaFree(allocation);
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "CUDA deallocation: %s\n", cudaGetErrorString(status));
            success = false;
        }
        else
            std::fprintf(stderr, "Freed GPU buffer.\n");
    }
    return success;
}
