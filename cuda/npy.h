// npy.h
// =====
// Write a float64 array in NumPy's .npy format (version 1.0), so the Python
// side can np.load() the result with nothing else installed. Assumes a
// little-endian host, which covers x86 and every GPU node you are likely to
// meet.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

inline bool write_npy(const std::string& path, const std::vector<double>& data,
                      const std::vector<int>& shape) {
    std::string shp = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) shp += ", ";
        shp += std::to_string(shape[i]);
    }
    if (shape.size() == 1) shp += ",";
    shp += ")";

    std::string header =
        "{'descr': '<f8', 'fortran_order': False, 'shape': " + shp + ", }";
    // magic (6) + version (2) + header length (2) + header + newline must be
    // a multiple of 64 bytes
    const size_t used = 10 + header.size() + 1;
    header += std::string((64 - used % 64) % 64, ' ');
    header += "\n";

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const uint16_t hlen = static_cast<uint16_t>(header.size());
    const unsigned char hl[2] = {static_cast<unsigned char>(hlen & 0xff),
                                 static_cast<unsigned char>(hlen >> 8)};
    bool ok = std::fwrite(magic, 1, 8, fp) == 8 && std::fwrite(hl, 1, 2, fp) == 2 &&
              std::fwrite(header.data(), 1, header.size(), fp) == header.size() &&
              std::fwrite(data.data(), sizeof(double), data.size(), fp) == data.size();
    return std::fclose(fp) == 0 && ok;
}
