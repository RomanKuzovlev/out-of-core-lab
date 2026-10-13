#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>
#include <chrono>

constexpr unsigned int dimensions = 128;

__global__ void dotProducts(const float *blockOfVectors, const float *query,
                            float *scores, unsigned int vectorCount)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectorCount)
        return;

    float score = 0.0f;
    for (unsigned int d = 0; d < dimensions; ++d)
        score += blockOfVectors[i * dimensions + d] * query[d];
    scores[i] = score;
}

void handleCuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

int main(int argc, char *argv[])
{
    constexpr std::size_t blockSize = 4096 * 1024; // 4 MiB
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "The dataset requires 32-bit IEEE 754 floats.");
    static_assert(blockSize % (dimensions * sizeof(float)) == 0,
                  "Each block must contain whole vectors.");
    constexpr unsigned int vectorsPerBlock = blockSize / (dimensions * sizeof(float));

    if (argc > 2)
    {
        std::cerr << "Usage: " << argv[0] << " [FILE]\n"
                  << "FILE defaults to sample.bin; all blocks are read sequentially.\n";
        return 1;
    }

    // Allocate once and reuse for every block.
    std::vector<float> buffer(blockSize / sizeof(float));
    char *bytes = reinterpret_cast<char *>(buffer.data());

    const char *path = argc == 2 ? argv[1] : "sample.bin";
    const int fd = open(path, O_RDONLY);
    if (fd == -1)
    {
        std::perror("open");
        return 1;
    }

    int exitCode = 0;
    // Keep the pointer in scope for cleanup even if the scan throws.
    void *gpuBuffer = nullptr;
    float *gpuQuery = nullptr;
    float *gpuScores = nullptr;
    try
    {
        struct stat fileStat{};
        if (fstat(fd, &fileStat) == -1)
            throw std::system_error(errno, std::generic_category(), "fstat");
        if (!S_ISREG(fileStat.st_mode))
            throw std::runtime_error("Expected a regular dataset file.");
        if (fileStat.st_size <= 0)
            throw std::runtime_error("Dataset file is empty.");

        const auto fileSize = static_cast<std::uint64_t>(fileStat.st_size);
        if (fileSize % blockSize != 0)
            throw std::runtime_error("File size is not a multiple of the 4 MiB block size.");
        const std::uint64_t blocksToRead = fileSize / blockSize;

        // One latency per block, in seconds. Allocate storage before reading.
        std::vector<std::chrono::duration<double>> measurements;
        if (blocksToRead > measurements.max_size())
            throw std::runtime_error("Too many blocks to store latency measurements.");
        measurements.reserve(static_cast<std::size_t>(blocksToRead));

        // CUDA. Query once, before reading. The count is valid only on success.
        int deviceCount = 0;
        handleCuda(cudaGetDeviceCount(&deviceCount), "CUDA device query");
        if (deviceCount == 0)
            throw std::runtime_error("No CUDA-capable GPU is available.");
        std::cerr << "CUDA devices available: " << deviceCount << '\n';

        // Allocate one block in GPU memory before the scan.
        handleCuda(cudaMalloc(&gpuBuffer, blockSize), "CUDA allocation");
        std::cerr << "Allocated GPU buffer: " << blockSize << " bytes.\n";

        // A fixed all-ones query makes the first computation easy to verify.
        std::vector<float> query(dimensions, 1.0f);
        std::vector<float> scores(vectorsPerBlock); // RAM destination for future results.

        handleCuda(cudaMalloc(&gpuQuery, query.size() * sizeof(float)), "CUDA query allocation");
        handleCuda(cudaMalloc(&gpuScores, scores.size() * sizeof(float)), "CUDA scores allocation");

        handleCuda(cudaMemcpy(gpuQuery, query.data(), query.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "CUDA query copy");
        // GPU scores need no initialization: the kernel will overwrite every valid slot.
        std::cerr << "Allocated query: " << query.size() * sizeof(float)
                  << " bytes; scores: " << scores.size() * sizeof(float) << " bytes.\n";

        for (std::uint64_t blockIndex = 0; blockIndex < blocksToRead; ++blockIndex)
        {
            // The validated file size bounds every block offset within off_t.
            const off_t blockOffset = static_cast<off_t>(blockIndex * blockSize);
            std::size_t bytesRead = 0;
            const auto start{std::chrono::steady_clock::now()};
            while (bytesRead < blockSize)
            {
                const ssize_t result = pread(fd, bytes + bytesRead, blockSize - bytesRead,
                                             blockOffset + static_cast<off_t>(bytesRead));
                if (result < 0)
                {
                    if (errno == EINTR)
                        continue;
                    throw std::system_error(errno, std::generic_category(), "pread");
                }
                if (result == 0)
                    throw std::runtime_error("Unexpected end of file while reading a block.");

                bytesRead += static_cast<std::size_t>(result);
            }
            const auto finish{std::chrono::steady_clock::now()};
            // Record after stopping the timer; reserve() prevents reallocations.
            measurements.emplace_back(finish - start);

            // Copy the completed RAM block into the reusable GPU allocation.
            // This transfer is outside the disk-read latency measurement.
            handleCuda(cudaMemcpy(gpuBuffer, buffer.data(), blockSize,
                                  cudaMemcpyHostToDevice),
                       "CUDA host-to-device copy");
        }

        // Output only after the scan, preserving sequential request order
        // (an OS can pre-read the next block while we are printing, thus skewing the measurment).
        std::cout << "block_index,latency_seconds\n";
        std::cout.precision(9);
        for (std::size_t i = 0; i < measurements.size(); ++i)
            std::cout << i << ',' << measurements[i].count() << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        exitCode = 1;
    }

    // Both normal completion and exceptions reach this cleanup.
    for (void *allocation : {static_cast<void *>(gpuScores), static_cast<void *>(gpuQuery), gpuBuffer})
    {
        if (allocation == nullptr)
            continue;
        const cudaError_t freeStatus = cudaFree(allocation);
        if (freeStatus != cudaSuccess)
        {
            std::cerr << "CUDA deallocation: " << cudaGetErrorString(freeStatus) << '\n';
            exitCode = 1;
        }
        else
            std::cerr << "Freed GPU buffer.\n";
    }
    if (close(fd) == -1)
    {
        std::perror("close");
        exitCode = 1;
    }

    return exitCode;
}
