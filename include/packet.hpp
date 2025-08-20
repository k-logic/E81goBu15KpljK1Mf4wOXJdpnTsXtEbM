#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace packet {

// ========== 共通ヘッダー ==========
struct packet_header {
    uint16_t frame_id;
    uint16_t chunk_id;
    uint16_t chunk_total;
    uint16_t data_len; 
    //uint16_t spare;
};

// ========== データ付きパケット ==========
struct packet_u8 {
    packet_header header;
    std::vector<uint8_t> compressed;
};

struct packet_u16 {
    packet_header header;
    std::vector<uint16_t> compressed;
};

struct packet_u32 {
    packet_header header;
    std::vector<uint32_t> compressed;
};

// ========== ヘルパー関数 ==========
inline void write_u16_le(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

inline void write_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

inline uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// ========== make_packet_u8 ==========
inline std::vector<uint8_t> make_packet_u8(
    packet_header& header,
    const std::vector<uint8_t>& compressed
) {
    std::vector<uint8_t> packet;
    write_u16_le(packet, header.frame_id);
    write_u16_le(packet, header.chunk_id);
    write_u16_le(packet, header.chunk_total);
    write_u16_le(packet, header.data_len);
    packet.insert(packet.end(), compressed.begin(), compressed.end());
    return packet;
}

inline packet_u8 parse_packet_u8(const std::vector<uint8_t>& packet) {
    const size_t HEADER_SIZE = 8;
    if (packet.size() < HEADER_SIZE) throw std::runtime_error("Invalid packet size");

    const uint8_t* ptr = packet.data();
    packet_header header{
        read_u16_le(ptr),
        read_u16_le(ptr + 2),
        read_u16_le(ptr + 4),
        read_u16_le(ptr + 6),
    };

    std::vector<uint8_t> compressed(ptr + HEADER_SIZE, ptr + packet.size());
    return packet_u8{header, std::move(compressed)};
}

// ========== make_packet_u16 ==========
inline std::vector<uint8_t> make_packet_u16(
    const packet_header& header,
    const std::vector<uint16_t>& compressed
) {
    std::vector<uint8_t> packet;
    write_u16_le(packet, header.frame_id);
    write_u16_le(packet, header.chunk_id);
    write_u16_le(packet, header.chunk_total);
    write_u16_le(packet, header.data_len);
    for (uint16_t v : compressed) {
        write_u16_le(packet, v);
    }
    return packet;
}

inline packet_u16 parse_packet_u16(const std::vector<uint8_t>& packet) {
    const size_t HEADER_SIZE = 8;
    if (packet.size() < HEADER_SIZE || (packet.size() - HEADER_SIZE) % 2 != 0)
        throw std::runtime_error("Invalid packet size for uint16_t");

    const uint8_t* ptr = packet.data();
    packet_header header{
        read_u16_le(ptr),
        read_u16_le(ptr + 2),
        read_u16_le(ptr + 4),
        read_u16_le(ptr + 6),
    };

    std::vector<uint16_t> compressed;
    size_t count = (packet.size() - HEADER_SIZE) / 2;
    compressed.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        compressed.push_back(read_u16_le(ptr + HEADER_SIZE + i * 2));
    }

    return packet_u16{header, std::move(compressed)};
}

// ========== make_packet_u32 ==========
inline std::vector<uint8_t> make_packet_u32(
    const packet_header& header,
    const std::vector<uint32_t>& compressed
) {
    std::vector<uint8_t> packet;
    write_u16_le(packet, header.frame_id);
    write_u16_le(packet, header.chunk_id);
    write_u16_le(packet, header.chunk_total);
    write_u16_le(packet, header.data_len);
    for (uint32_t v : compressed) {
        write_u32_le(packet, v);
    }
    return packet;
}

inline packet_u32 parse_packet_u32(const std::vector<uint8_t>& packet) {
    const size_t HEADER_SIZE = 8;
    if (packet.size() < HEADER_SIZE || (packet.size() - HEADER_SIZE) % 4 != 0)
        throw std::runtime_error("Invalid packet size for uint32_t");

    const uint8_t* ptr = packet.data();
    packet_header header{
        read_u16_le(ptr),
        read_u16_le(ptr + 2),
        read_u16_le(ptr + 4),
        read_u16_le(ptr + 6),
    };

    std::vector<uint32_t> compressed;
    size_t count = (packet.size() - HEADER_SIZE) / 4;
    compressed.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        compressed.push_back(read_u32_le(ptr + HEADER_SIZE + i * 4));
    }

    return packet_u32{header, std::move(compressed)};
}

}
