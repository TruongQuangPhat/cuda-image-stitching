// Authors: HUỲNH TẤN PHƯỚC (23120334), integration.
// PPM reader/writer for V0; no stb implementation is bundled.
// Compatibility with the course starter interface has not been verified.

#include "common.h"
#include <fstream>
#include <limits>
#include <stdexcept>

namespace stitching::io {
namespace {
std::string token(std::istream& stream) {
    std::string value;
    while (stream >> std::ws && stream.peek() == '#')
        stream.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if (!(stream >> value)) throw std::runtime_error("Incomplete PPM header");
    return value;
}

}  // namespace

warping::RgbImage readPpm(const std::string& path) {
    std::ifstream stream(path,std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open input PPM");
    if (token(stream) != "P6") throw std::runtime_error("Expected RGB P6 PPM");
    const int width = std::stoi(token(stream)), height = std::stoi(token(stream));
    if (width <= 0 || height <= 0 || static_cast<double>(width)*height > 16777216 ||
        token(stream) != "255") throw std::runtime_error("Invalid PPM dimensions/range");
    const int separator = stream.get();
    if (separator == '\r' && stream.peek() == '\n') stream.get();
    else if (separator != ' ' && separator != '\t' && separator != '\n' && separator != '\r')
        throw std::runtime_error("Invalid PPM separator");
    warping::RgbImage image{width,height,std::vector<std::uint8_t>(
        static_cast<std::size_t>(width)*height*3)};
    stream.read(reinterpret_cast<char*>(image.pixels.data()),
                static_cast<std::streamsize>(image.pixels.size()));
    if (!stream) throw std::runtime_error("Truncated PPM input");
    return image;
}

void writePpm(const std::string& path, const warping::RgbImage& input) {
    std::ofstream image(path,std::ios::binary);
    image << "P6\n" << input.width << ' ' << input.height << "\n255\n";
    image.write(reinterpret_cast<const char*>(input.pixels.data()),
                static_cast<std::streamsize>(input.pixels.size()));
    if (!image) throw std::runtime_error("Cannot write panorama");
}
}  // namespace stitching::io
