#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace packet {
struct packet_u8 {
    uint16_t frame_id;
    uint16_t chunk_id;
    uint16_t chunk_total;
    uint8_t spare;
    std::vector<uint8_t> compressed;
};

struct packet_u16 {
    uint16_t frame_id;
    uint16_t chunk_id;
    uint16_t chunk_total;
    uint8_t spare;
    std::vector<uint16_t> compressed;
};

struct packet_u32 {
    uint16_t frame_id;
    uint16_t chunk_id;
    uint16_t chunk_total;
    uint8_t spare;
    std::vector<uint32_t> compressed;
};

inline std::vector<uint8_t> make_packet_u8(
    uint16_t frame_id,
    uint16_t chunk_id,
    uint16_t chunk_total,
    uint8_t spare,
    const std::vector<uint8_t>& compressed
) {
    std::vector<uint8_t> packet;

    packet.push_back(frame_id & 0xFF);
    packet.push_back((frame_id >> 8) & 0xFF);

    packet.push_back(chunk_id & 0xFF);
    packet.push_back((chunk_id >> 8) & 0xFF);

    packet.push_back(chunk_total & 0xFF);
    packet.push_back((chunk_total >> 8) & 0xFF);

    packet.push_back(spare);

    packet.insert(packet.end(), compressed.begin(), compressed.end());

    return packet;
}

inline packet_u8 parse_packet_u8(const std::vector<uint8_t>& packet) {
    const size_t HEADER_SIZE = 7;

    if (packet.size() < HEADER_SIZE) throw std::runtime_error("Invalid packet size");

    const uint8_t* ptr = packet.data();
    uint16_t frame_id = ptr[0] | (ptr[1] << 8);
    uint16_t chunk_id = ptr[2] | (ptr[3] << 8);
    uint16_t chunk_total = ptr[4] | (ptr[5] << 8);
    uint8_t spare = ptr[6];

    std::vector<uint8_t> compressed(ptr + HEADER_SIZE, packet.data() + packet.size());

    return packet_u8{
        .frame_id = frame_id,
        .chunk_id = chunk_id,
        .chunk_total = chunk_total,
        .spare = spare,
        .compressed = std::move(compressed)
    };
}

inline std::vector<uint8_t> make_packet_u16(
    uint16_t frame_id,
    uint16_t chunk_id,
    uint16_t chunk_total,
    uint8_t spare,
    const std::vector<uint16_t>& compressed
) {
    std::vector<uint8_t> packet;

    packet.push_back(frame_id & 0xFF);
    packet.push_back((frame_id >> 8) & 0xFF);

    packet.push_back(chunk_id & 0xFF);
    packet.push_back((chunk_id >> 8) & 0xFF);

    packet.push_back(chunk_total & 0xFF);
    packet.push_back((chunk_total >> 8) & 0xFF);

    packet.push_back(spare);

    for (uint16_t v : compressed) {
        packet.push_back(v & 0xFF);
        packet.push_back((v >> 8) & 0xFF);
    }

    return packet;
}

inline packet_u16 parse_packet_u16(const std::vector<uint8_t>& packet) {
    const size_t HEADER_SIZE = 7;
    if (packet.size() < HEADER_SIZE || (packet.size() - HEADER_SIZE) % 2 != 0)
        throw std::runtime_error("Invalid packet size for float16");

    const uint8_t* ptr = packet.data();
    uint16_t frame_id = ptr[0] | (ptr[1] << 8);
    uint16_t chunk_id = ptr[2] | (ptr[3] << 8);
    uint16_t chunk_total = ptr[4] | (ptr[5] << 8);
    uint8_t spare = ptr[6];

    std::vector<uint16_t> compressed;
    size_t count = (packet.size() - HEADER_SIZE) / 2;
    compressed.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        uint8_t lo = packet[HEADER_SIZE + 2 * i];
        uint8_t hi = packet[HEADER_SIZE + 2 * i + 1];
        compressed.push_back(static_cast<uint16_t>(lo | (hi << 8)));
    }

    return packet_u16{
        .frame_id = frame_id,
        .chunk_id = chunk_id,
        .chunk_total = chunk_total,
        .spare = spare,
        .compressed = std::move(compressed)
    };
}

inline std::vector<uint8_t> make_packet_u32(
    uint16_t frame_id,
    uint16_t chunk_id,
    uint16_t chunk_total,
    uint8_t spare,
    const std::vector<uint32_t>& compressed
) {
    std::vector<uint8_t> packet;

    packet.push_back(frame_id & 0xFF);
    packet.push_back((frame_id >> 8) & 0xFF);

    packet.push_back(chunk_id & 0xFF);
    packet.push_back((chunk_id >> 8) & 0xFF);

    packet.push_back(chunk_total & 0xFF);
    packet.push_back((chunk_total >> 8) & 0xFF);

    packet.push_back(spare);

    for (uint32_t v : compressed) {
        packet.push_back(v & 0xFF);
        packet.push_back((v >> 8) & 0xFF);
        packet.push_back((v >> 16) & 0xFF);
        packet.push_back((v >> 24) & 0xFF);
    }

    return packet;
}

inline packet_u32 parse_packet_u32(const std::vector<uint8_t>& packet) {
    const size_t HEADER_SIZE = 7;
    if (packet.size() < HEADER_SIZE || (packet.size() - HEADER_SIZE) % 4 != 0)
        throw std::runtime_error("Invalid packet size for uint32_t");

    const uint8_t* ptr = packet.data();
    uint16_t frame_id = ptr[0] | (ptr[1] << 8);
    uint16_t chunk_id = ptr[2] | (ptr[3] << 8);
    uint16_t chunk_total = ptr[4] | (ptr[5] << 8);
    uint8_t spare = ptr[6];

    std::vector<uint32_t> compressed;
    size_t count = (packet.size() - HEADER_SIZE) / 4;
    compressed.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        size_t offset = HEADER_SIZE + i * 4;
        uint32_t val =
            packet[offset] |
            packet[offset + 1] << 8 |
            packet[offset + 2] << 16 |
            packet[offset + 3] << 24;
        compressed.push_back(val);
    }

    return packet_u32{
        .frame_id = frame_id,
        .chunk_id = chunk_id,
        .chunk_total = chunk_total,
        .spare = spare,
        .compressed = std::move(compressed)
    };
}

}
