#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

int main(int argc, char *argv[])
{
    constexpr std::size_t blockSize = 4096 * 1024; // 4 MiB
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

    // Allocate once and reuse for every block.
    std::vector<char> buffer(blockSize);

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
        const std::string value = std::to_string(i) + "yo";

        // Repeat the pattern, trimming the final copy to the remaining space.
        for (std::size_t offset = 0; offset < blockSize;)
        {
            const std::size_t amount = std::min(value.size(), blockSize - offset);
            std::memcpy(buffer.data() + offset, value.data(), amount);
            offset += amount;
        }

        std::size_t bytesWritten = 0;
        while (bytesWritten < blockSize)
        {
            const ssize_t result = write(fd, buffer.data() + bytesWritten, blockSize - bytesWritten);
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
