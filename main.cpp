#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <unistd.h>
#include <vector>

int main(int argc, char *argv[])
{
    constexpr std::size_t blockSize = 4096 * 1024; // 4 MiB
    constexpr std::size_t dimensions = 128;
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "The dataset requires 32-bit IEEE 754 floats.");
    static_assert(blockSize % (dimensions * sizeof(float)) == 0,
                  "Each block must contain whole vectors.");
    constexpr std::uint64_t targetSize = std::uint64_t{32} * 1024 * 1024 * 1024;
    std::uint64_t blocksAmount = targetSize / blockSize;

    // An optional block count allows small development runs.
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " OUTPUT [BLOCK_COUNT]\n";
        return 1;
    }
    if (argc == 3)
    {
        const std::string_view count = argv[2];
        const auto result = std::from_chars(count.data(), count.data() + count.size(), blocksAmount);
        if (result.ec != std::errc{} || result.ptr != count.data() + count.size() || blocksAmount == 0)
        {
            std::cerr << "BLOCK_COUNT must be a positive integer.\n";
            return 1;
        }
    }

    // Headerless file: contiguous vectors of 128 floats, in native byte order.
    // Each 4 MiB block holds 8,192 vectors. Reuse this allocation throughout.
    std::vector<float> buffer(blockSize / sizeof(float));

    // Seed once; the sequence continues across blocks. Reproducibility assumes
    // the same generator and standard-library distribution implementation.
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    // Create a new file; fail if the path already exists.
    const int fd = open(argv[1], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd == -1)
    {
        std::perror("open");
        return 1;
    }

    bool failed = false;
    for (std::uint64_t i = 0; i < blocksAmount; ++i)
    {
        for (float &value : buffer)
            value = distribution(generator);

        // write() counts bytes, so partial-write offsets need a byte pointer.
        const char *bytes = reinterpret_cast<const char *>(buffer.data());

        std::size_t bytesWritten = 0;
        while (bytesWritten < blockSize)
        {
            const ssize_t result = write(fd, bytes + bytesWritten, blockSize - bytesWritten);
            if (result < 0)
            {
                // A signal interrupted the call before any bytes were written.
                if (errno == EINTR)
                    continue;
                std::perror("write");
                failed = true;
                break;
            }
            if (result == 0)
            {
                std::cerr << "write made no progress.\n";
                failed = true;
                break;
            }
            bytesWritten += static_cast<std::size_t>(result);
        }
        if (failed)
            break;
    }

    if (close(fd) == -1)
    {
        std::perror("close");
        failed = true;
    }
    if (failed)
    {
        std::cerr << "Generation failed; the output file may be incomplete.\n";
        return 1;
    }

    std::cout << "Wrote " << blocksAmount << " blocks of " << blockSize << " bytes.\n";
}
