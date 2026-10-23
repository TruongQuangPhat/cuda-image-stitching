// Authors: HUỲNH TẤN PHƯỚC (23120334), integration.
// PPM reader/writer for V0; no stb implementation is bundled.
// Compatibility with the course starter interface has not been verified.

#include "common.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace stitching::io {
namespace {
bool whitespace(int ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
           ch == '\v' || ch == '\f';
}

std::string token(std::istream& stream) {
    std::string value;
    for (;;) {
        if (whitespace(stream.peek())) stream.get();
        else if (stream.peek() == '#') {
            int ch;
            do { ch = stream.get(); } while (ch != '\n' && ch != '\r' && ch != EOF);
        } else break;
    }
    while (stream.peek() != EOF && !whitespace(stream.peek()) && stream.peek() != '#')
        value.push_back(static_cast<char>(stream.get()));
    if (value.empty()) throw std::runtime_error("Incomplete PPM header");
    return value;
}

int decimal(const std::string& text) {
    int value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9' ||
            value > (std::numeric_limits<int>::max() - (ch - '0')) / 10)
            throw std::runtime_error("Invalid PPM numeric token");
        value = value * 10 + (ch - '0');
    }
    return value;
}

std::size_t rgbBytes(int width, int height) {
    if (width <= 0 || height <= 0) throw std::runtime_error("Invalid PPM dimensions");
    const auto w = static_cast<std::size_t>(width), h = static_cast<std::size_t>(height);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (w > limit / h || w * h > limit / 3)
        throw std::runtime_error("PPM dimensions overflow RGB buffer size");
    const auto bytes = w * h * 3;
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::runtime_error("PPM buffer size is not representable");
    return bytes;
}

}  // namespace

warping::RgbImage readPpm(const std::string& path) {
    std::ifstream stream(path,std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open input PPM");
    if (token(stream) != "P6") throw std::runtime_error("Expected RGB P6 PPM");
    const int width = decimal(token(stream)), height = decimal(token(stream));
    const auto bytes = rgbBytes(width, height);
    if (bytes / 3 > 16777216 || decimal(token(stream)) != 255)
        throw std::runtime_error("Invalid PPM dimensions/range");
    int separator = stream.get();
    if (separator == '#') {
        do { separator = stream.get(); }
        while (separator != '\n' && separator != '\r' && separator != EOF);
    }
    // Do not skip binary whitespace/comments after the one raster separator.
    if (!whitespace(separator))
        throw std::runtime_error("Invalid PPM separator");
    const bool possible_crlf = separator == '\r' && stream.peek() == '\n';
    warping::RgbImage image{width,height,std::vector<std::uint8_t>(bytes)};
    stream.read(reinterpret_cast<char*>(image.pixels.data()),
                static_cast<std::streamsize>(image.pixels.size()));
    if (!stream) throw std::runtime_error("Truncated PPM input");
    if (possible_crlf) {
        // Prefer the conventional CRLF ending when a complete raster follows.
        // With exactly 'bytes' left after CR, LF is instead the first pixel.
        if (stream.peek() != EOF) {
            std::move(image.pixels.begin()+1, image.pixels.end(), image.pixels.begin());
            image.pixels.back() = static_cast<std::uint8_t>(stream.get());
            if (!stream) throw std::runtime_error("Truncated PPM input");
        } else {
            if (stream.bad()) throw std::runtime_error("Cannot read PPM raster");
            stream.clear();  // EOF from lookahead is not an input/close failure.
        }
    }
    stream.close();
    if (!stream) throw std::runtime_error("Failed closing input PPM");
    return image;
}

void writePpm(const std::string& path, const warping::RgbImage& input) {
    const auto bytes = rgbBytes(input.width, input.height);
    if (input.pixels.size() != bytes)
        throw std::runtime_error("PPM image must contain width*height*3 RGB bytes");
    std::ofstream image(path,std::ios::binary);
    if (!image) throw std::runtime_error("Cannot open panorama output");
    image << "P6\n" << input.width << ' ' << input.height << "\n255\n";
    image.write(reinterpret_cast<const char*>(input.pixels.data()),
                static_cast<std::streamsize>(input.pixels.size()));
    if (!image) throw std::runtime_error("Cannot write panorama");
    image.flush();
    if (!image) throw std::runtime_error("Cannot flush panorama");
    image.close();
    if (!image) throw std::runtime_error("Cannot close panorama");
}
}  // namespace stitching::io
