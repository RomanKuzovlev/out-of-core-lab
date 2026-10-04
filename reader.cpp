#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <string_view>
#include <unistd.h>
#include <vector>

int main(int argc, char *argv[])
{
    constexpr std::size_t blockSize = 4096 * 1024; // 4 MiB
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "The dataset requires 32-bit IEEE 754 floats.");

    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " BLOCK_INDEX [FILE]\n"
                  << "FILE defaults to sample.bin; indices start at zero.\n";
        return 1;
    }

    std::uint64_t blockIndex = 0;
    const std::string_view indexText = argv[1];
    const auto parsed = std::from_chars(indexText.data(), indexText.data() + indexText.size(), blockIndex);
    if (parsed.ec != std::errc{} || parsed.ptr != indexText.data() + indexText.size())
    {
        std::cerr << "BLOCK_INDEX must be a nonnegative integer.\n";
        return 1;
    }

    // Ensure both the starting offset and the full block fit in off_t.
    const auto maxOffset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (blockIndex > (maxOffset - blockSize) / blockSize)
    {
        std::cerr << "BLOCK_INDEX is too large for a file offset.\n";
        return 1;
    }
    const off_t blockOffset = static_cast<off_t>(blockIndex * blockSize);

    std::vector<float> buffer(blockSize / sizeof(float));
    char *bytes = reinterpret_cast<char *>(buffer.data());

    const char *path = argc == 3 ? argv[2] : "sample.bin";
    const int fd = open(path, O_RDONLY);
    if (fd == -1)
    {
        std::perror("open");
        return 1;
    }

    bool failed = false;
    std::size_t bytesRead = 0;
    while (bytesRead < blockSize)
    {
        // Advance in both the destination buffer and the file after a short read.
        const ssize_t result = pread(fd, bytes + bytesRead, blockSize - bytesRead,
                                     blockOffset + static_cast<off_t>(bytesRead));
        if (result < 0)
        {
            if (errno == EINTR)
                continue;
            std::perror("pread");
            failed = true;
            break;
        }
        if (result == 0)
        {
            std::cerr << "Reached end of file after " << bytesRead
                      << " bytes; requested block is missing or incomplete.\n";
            failed = true;
            break;
        }
        bytesRead += static_cast<std::size_t>(result);
    }

    if (close(fd) == -1)
    {
        std::perror("close");
        failed = true;
    }
    if (failed)
        return 1;

    std::cout << "Read block " << blockIndex << " at byte offset " << blockOffset
              << ": " << bytesRead << " bytes.\n";
    std::cout << "First float: " << buffer.front()
              << "; last float: " << buffer.back() << '\n';
}
