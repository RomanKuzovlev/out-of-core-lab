#pragma once

#include <cstddef>

// Owns the reusable GPU allocations. CUDA types stay in the implementation file.
class CudaWorkload
{
public:
    static constexpr unsigned int dimensions = 128;

    explicit CudaWorkload(std::size_t blockSize);
    ~CudaWorkload() noexcept;

    // Two owners of the same pointers would attempt to free them twice.
    CudaWorkload(const CudaWorkload &) = delete;
    CudaWorkload &operator=(const CudaWorkload &) = delete;

    void copyBlock(const float *source);

    // Attempts every release, reports failures, and never throws.
    // Repeated calls do nothing after the first attempt.
    bool release() noexcept;

private:
    std::size_t blockSize_;
    float *gpuBuffer_ = nullptr;
    float *gpuQuery_ = nullptr;
    float *gpuScores_ = nullptr;
};
