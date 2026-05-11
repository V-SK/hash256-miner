#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct ParamsHost {
    uint64_t challenge[4];   // Keccak absorb lanes for bytes32 challenge.
    uint64_t target[4];      // Effective scan target, big-endian uint256 words.
    uint32_t challenge32[8]; // Little-endian uint32 pairs for the uint2 kernel.
    uint32_t target32[8];    // Effective scan target, big-endian uint256 words.
    uint64_t start;
    uint64_t total;
    uint64_t iters;
    uint64_t stride;
};

struct ResultHost {
    uint32_t found;
    uint32_t pad;
    uint64_t nonce;
    uint64_t digest[4];      // Big-endian uint256 words.
};

static const char *kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong challenge[4];
    ulong target[4];
    uint challenge32[8];
    uint target32[8];
    ulong start;
    ulong total;
    ulong iters;
    ulong stride;
};

struct Result {
    atomic_uint found;
    uint pad;
    ulong nonce;
    ulong digest[4];
};

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL
};

constant ushort RHO[25] = {
    0, 1, 62, 28, 27,
    36, 44, 6, 55, 20,
    3, 10, 43, 25, 39,
    41, 45, 15, 21, 8,
    18, 2, 61, 56, 14
};

constant uchar ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36,
    45, 55, 2, 14, 27, 41, 56, 8,
    25, 43, 62, 18, 39, 61, 20, 44
};

constant uchar PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16,
    8, 21, 24, 4, 15, 23, 19, 13,
    12, 2, 20, 14, 22, 9, 6, 1
};

inline ulong rotl64(ulong x, ushort n) {
    return n == 0 ? x : ((x << n) | (x >> (64 - n)));
}

inline ulong bswap64(ulong x) {
    x = ((x & 0x00ff00ff00ff00ffUL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffUL);
    x = ((x & 0x0000ffff0000ffffUL) << 16) | ((x >> 16) & 0x0000ffff0000ffffUL);
    return (x << 32) | (x >> 32);
}

inline void keccakf(thread ulong A[25]) {
    for (uint round = 0; round < 24; round++) {
        ulong C[5];
        ulong D[5];
        ulong B[25];

        for (uint x = 0; x < 5; x++) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        for (uint x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                A[x + 5 * y] ^= D[x];
            }
        }

        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                uint src = x + 5 * y;
                uint dst = y + 5 * ((2 * x + 3 * y) % 5);
                B[dst] = rotl64(A[src], RHO[src]);
            }
        }

        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                A[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);
            }
        }

        A[0] ^= RC[round];
    }
}

inline void keccakf_compact(thread ulong A[25]) {
    for (uint round = 0; round < 24; round++) {
        ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D0 = C4 ^ rotl64(C1, 1);
        ulong D1 = C0 ^ rotl64(C2, 1);
        ulong D2 = C1 ^ rotl64(C3, 1);
        ulong D3 = C2 ^ rotl64(C4, 1);
        ulong D4 = C3 ^ rotl64(C0, 1);

        A[0] ^= D0; A[5] ^= D0; A[10] ^= D0; A[15] ^= D0; A[20] ^= D0;
        A[1] ^= D1; A[6] ^= D1; A[11] ^= D1; A[16] ^= D1; A[21] ^= D1;
        A[2] ^= D2; A[7] ^= D2; A[12] ^= D2; A[17] ^= D2; A[22] ^= D2;
        A[3] ^= D3; A[8] ^= D3; A[13] ^= D3; A[18] ^= D3; A[23] ^= D3;
        A[4] ^= D4; A[9] ^= D4; A[14] ^= D4; A[19] ^= D4; A[24] ^= D4;

        ulong t = A[1];
        for (uint i = 0; i < 24; i++) {
            uint j = PILN[i];
            ulong tmp = A[j];
            A[j] = rotl64(t, ROTC[i]);
            t = tmp;
        }

        for (uint y = 0; y < 25; y += 5) {
            ulong a0 = A[y + 0];
            ulong a1 = A[y + 1];
            ulong a2 = A[y + 2];
            ulong a3 = A[y + 3];
            ulong a4 = A[y + 4];
            A[y + 0] = a0 ^ ((~a1) & a2);
            A[y + 1] = a1 ^ ((~a2) & a3);
            A[y + 2] = a2 ^ ((~a3) & a4);
            A[y + 3] = a3 ^ ((~a4) & a0);
            A[y + 4] = a4 ^ ((~a0) & a1);
        }

        A[0] ^= RC[round];
    }
}

inline void absorb_and_permute(device const Params& p, ulong nonce, thread ulong A[25]) {
    for (uint i = 0; i < 25; i++) A[i] = 0;

    // abi.encode(bytes32 challenge, uint256 nonce) is exactly 64 bytes.
    A[0] ^= p.challenge[0];
    A[1] ^= p.challenge[1];
    A[2] ^= p.challenge[2];
    A[3] ^= p.challenge[3];
    A[4] ^= 0UL;
    A[5] ^= 0UL;
    A[6] ^= 0UL;
    A[7] ^= bswap64(nonce);  // Low 64 bits of uint256, ABI big-endian.

    // Ethereum Keccak padding: 0x01 then final rate byte OR 0x80.
    A[8]  ^= 0x01UL;
    A[16] ^= 0x8000000000000000UL;

    keccakf(A);
}

inline ulong hash_nonce_top(device const Params& p, ulong nonce) {
    ulong A[25];
    absorb_and_permute(p, nonce, A);
    return bswap64(A[0]);
}

inline ulong hash_nonce_top_compact(device const Params& p, ulong nonce) {
    ulong A[25];
    for (uint i = 0; i < 25; i++) A[i] = 0;
    A[0] ^= p.challenge[0];
    A[1] ^= p.challenge[1];
    A[2] ^= p.challenge[2];
    A[3] ^= p.challenge[3];
    A[7] ^= bswap64(nonce);
    A[8] ^= 0x01UL;
    A[16] ^= 0x8000000000000000UL;
    keccakf_compact(A);
    return bswap64(A[0]);
}

inline void hash_nonce_full_compact(device const Params& p, ulong nonce, thread ulong digest_be[4]) {
    ulong A[25];
    for (uint i = 0; i < 25; i++) A[i] = 0;
    A[0] ^= p.challenge[0];
    A[1] ^= p.challenge[1];
    A[2] ^= p.challenge[2];
    A[3] ^= p.challenge[3];
    A[7] ^= bswap64(nonce);
    A[8] ^= 0x01UL;
    A[16] ^= 0x8000000000000000UL;
    keccakf_compact(A);
    digest_be[0] = bswap64(A[0]);
    digest_be[1] = bswap64(A[1]);
    digest_be[2] = bswap64(A[2]);
    digest_be[3] = bswap64(A[3]);
}

inline void hash_nonce_scalar_digest(device const Params& p, ulong nonce, thread ulong digest_be[4]) {
    ulong a0 = p.challenge[0], a1 = p.challenge[1], a2 = p.challenge[2], a3 = p.challenge[3], a4 = 0;
    ulong a5 = 0, a6 = 0, a7 = bswap64(nonce), a8 = 0x01UL, a9 = 0;
    ulong a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0;
    ulong a15 = 0, a16 = 0x8000000000000000UL, a17 = 0, a18 = 0, a19 = 0;
    ulong a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

    for (uint round = 0; round < 24; round++) {
        ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;
        ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;
        ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;
        ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;
        ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;
        ulong d0 = c4 ^ rotl64(c1, 1);
        ulong d1 = c0 ^ rotl64(c2, 1);
        ulong d2 = c1 ^ rotl64(c3, 1);
        ulong d3 = c2 ^ rotl64(c4, 1);
        ulong d4 = c3 ^ rotl64(c0, 1);

        a0 ^= d0; a5 ^= d0; a10 ^= d0; a15 ^= d0; a20 ^= d0;
        a1 ^= d1; a6 ^= d1; a11 ^= d1; a16 ^= d1; a21 ^= d1;
        a2 ^= d2; a7 ^= d2; a12 ^= d2; a17 ^= d2; a22 ^= d2;
        a3 ^= d3; a8 ^= d3; a13 ^= d3; a18 ^= d3; a23 ^= d3;
        a4 ^= d4; a9 ^= d4; a14 ^= d4; a19 ^= d4; a24 ^= d4;

        ulong t = a1;
        ulong tmp = a10; a10 = rotl64(t, 1);  t = tmp;
        tmp = a7;       a7  = rotl64(t, 3);  t = tmp;
        tmp = a11;      a11 = rotl64(t, 6);  t = tmp;
        tmp = a17;      a17 = rotl64(t, 10); t = tmp;
        tmp = a18;      a18 = rotl64(t, 15); t = tmp;
        tmp = a3;       a3  = rotl64(t, 21); t = tmp;
        tmp = a5;       a5  = rotl64(t, 28); t = tmp;
        tmp = a16;      a16 = rotl64(t, 36); t = tmp;
        tmp = a8;       a8  = rotl64(t, 45); t = tmp;
        tmp = a21;      a21 = rotl64(t, 55); t = tmp;
        tmp = a24;      a24 = rotl64(t, 2);  t = tmp;
        tmp = a4;       a4  = rotl64(t, 14); t = tmp;
        tmp = a15;      a15 = rotl64(t, 27); t = tmp;
        tmp = a23;      a23 = rotl64(t, 41); t = tmp;
        tmp = a19;      a19 = rotl64(t, 56); t = tmp;
        tmp = a13;      a13 = rotl64(t, 8);  t = tmp;
        tmp = a12;      a12 = rotl64(t, 25); t = tmp;
        tmp = a2;       a2  = rotl64(t, 43); t = tmp;
        tmp = a20;      a20 = rotl64(t, 62); t = tmp;
        tmp = a14;      a14 = rotl64(t, 18); t = tmp;
        tmp = a22;      a22 = rotl64(t, 39); t = tmp;
        tmp = a9;       a9  = rotl64(t, 61); t = tmp;
        tmp = a6;       a6  = rotl64(t, 20); t = tmp;
        tmp = a1;       a1  = rotl64(t, 44);

        ulong b0 = a0, b1 = a1, b2 = a2, b3 = a3, b4 = a4;
        a0 = b0 ^ ((~b1) & b2);
        a1 = b1 ^ ((~b2) & b3);
        a2 = b2 ^ ((~b3) & b4);
        a3 = b3 ^ ((~b4) & b0);
        a4 = b4 ^ ((~b0) & b1);

        b0 = a5; b1 = a6; b2 = a7; b3 = a8; b4 = a9;
        a5 = b0 ^ ((~b1) & b2);
        a6 = b1 ^ ((~b2) & b3);
        a7 = b2 ^ ((~b3) & b4);
        a8 = b3 ^ ((~b4) & b0);
        a9 = b4 ^ ((~b0) & b1);

        b0 = a10; b1 = a11; b2 = a12; b3 = a13; b4 = a14;
        a10 = b0 ^ ((~b1) & b2);
        a11 = b1 ^ ((~b2) & b3);
        a12 = b2 ^ ((~b3) & b4);
        a13 = b3 ^ ((~b4) & b0);
        a14 = b4 ^ ((~b0) & b1);

        b0 = a15; b1 = a16; b2 = a17; b3 = a18; b4 = a19;
        a15 = b0 ^ ((~b1) & b2);
        a16 = b1 ^ ((~b2) & b3);
        a17 = b2 ^ ((~b3) & b4);
        a18 = b3 ^ ((~b4) & b0);
        a19 = b4 ^ ((~b0) & b1);

        b0 = a20; b1 = a21; b2 = a22; b3 = a23; b4 = a24;
        a20 = b0 ^ ((~b1) & b2);
        a21 = b1 ^ ((~b2) & b3);
        a22 = b2 ^ ((~b3) & b4);
        a23 = b3 ^ ((~b4) & b0);
        a24 = b4 ^ ((~b0) & b1);

        a0 ^= RC[round];
    }

    digest_be[0] = bswap64(a0);
    digest_be[1] = bswap64(a1);
    digest_be[2] = bswap64(a2);
    digest_be[3] = bswap64(a3);
}

inline void hash_nonce_full(device const Params& p, ulong nonce, thread ulong digest_be[4]) {
    ulong A[25];
    absorb_and_permute(p, nonce, A);

    digest_be[0] = bswap64(A[0]);
    digest_be[1] = bswap64(A[1]);
    digest_be[2] = bswap64(A[2]);
    digest_be[3] = bswap64(A[3]);
}

kernel void scan_hash(device const Params& p [[buffer(0)]],
                      device Result& r [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    ulong baseIndex = ((ulong)gid) * p.iters;

    for (ulong i = 0; i < p.iters; i++) {
        ulong index = baseIndex + i;
        if (index >= p.total) return;
        ulong nonce = p.start + index * p.stride;

        ulong top = hash_nonce_top(p, nonce);
        bool hit = top < p.target[0];

        if (top == p.target[0]) {
            ulong digest[4];
            hash_nonce_full(p, nonce, digest);
            hit = false;
            for (uint w = 1; w < 4; w++) {
                if (digest[w] < p.target[w]) {
                    hit = true;
                    break;
                }
                if (digest[w] > p.target[w]) {
                    break;
                }
            }
        }

        if (hit) {
            if (atomic_exchange_explicit(&r.found, 1, memory_order_relaxed) == 0) {
                ulong digest[4];
                hash_nonce_full(p, nonce, digest);
                r.nonce = nonce;
                r.digest[0] = digest[0];
                r.digest[1] = digest[1];
                r.digest[2] = digest[2];
                r.digest[3] = digest[3];
            }
            return;
        }
    }
}

kernel void scan_hash_compact(device const Params& p [[buffer(0)]],
                              device Result& r [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
    ulong baseIndex = ((ulong)gid) * p.iters;

    for (ulong i = 0; i < p.iters; i++) {
        ulong index = baseIndex + i;
        if (index >= p.total) return;
        ulong nonce = p.start + index * p.stride;

        ulong top = hash_nonce_top_compact(p, nonce);
        bool hit = top < p.target[0];

        if (top == p.target[0]) {
            ulong digest[4];
            hash_nonce_full_compact(p, nonce, digest);
            hit = false;
            for (uint w = 1; w < 4; w++) {
                if (digest[w] < p.target[w]) {
                    hit = true;
                    break;
                }
                if (digest[w] > p.target[w]) {
                    break;
                }
            }
        }

        if (hit) {
            if (atomic_exchange_explicit(&r.found, 1, memory_order_relaxed) == 0) {
                ulong digest[4];
                hash_nonce_full_compact(p, nonce, digest);
                r.nonce = nonce;
                r.digest[0] = digest[0];
                r.digest[1] = digest[1];
                r.digest[2] = digest[2];
                r.digest[3] = digest[3];
            }
            return;
        }
    }
}

kernel void scan_hash_scalar(device const Params& p [[buffer(0)]],
                             device Result& r [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
    ulong baseIndex = ((ulong)gid) * p.iters;

    for (ulong i = 0; i < p.iters; i++) {
        ulong index = baseIndex + i;
        if (index >= p.total) return;
        ulong nonce = p.start + index * p.stride;

        ulong digest[4];
        hash_nonce_scalar_digest(p, nonce, digest);

        bool hit = digest[0] < p.target[0];
        if (digest[0] == p.target[0]) {
            for (uint w = 1; w < 4; w++) {
                if (digest[w] < p.target[w]) {
                    hit = true;
                    break;
                }
                if (digest[w] > p.target[w]) {
                    break;
                }
            }
        }

        if (hit) {
            if (atomic_exchange_explicit(&r.found, 1, memory_order_relaxed) == 0) {
                r.nonce = nonce;
                r.digest[0] = digest[0];
                r.digest[1] = digest[1];
                r.digest[2] = digest[2];
                r.digest[3] = digest[3];
            }
            return;
        }
    }
}

inline uint bswap32(uint x) {
    x = ((x & 0x00ff00ffU) << 8) | ((x >> 8) & 0x00ff00ffU);
    return (x << 16) | (x >> 16);
}

inline uint2 rotl64_32(uint2 v, ushort n) {
    if (n == 0) return v;
    if (n < 32) {
        return uint2((v.x << n) | (v.y >> (32 - n)),
                     (v.y << n) | (v.x >> (32 - n)));
    }
    if (n == 32) return uint2(v.y, v.x);
    ushort m = n - 32;
    return uint2((v.y << m) | (v.x >> (32 - m)),
                 (v.x << m) | (v.y >> (32 - m)));
}

inline void keccakf32(thread uint2 A[25]) {
    for (uint round = 0; round < 24; round++) {
        uint2 C[5];
        uint2 D[5];
        uint2 B[25];

        for (uint x = 0; x < 5; x++) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        for (uint x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64_32(C[(x + 1) % 5], 1);
        }
        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                A[x + 5 * y] ^= D[x];
            }
        }

        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                uint src = x + 5 * y;
                uint dst = y + 5 * ((2 * x + 3 * y) % 5);
                B[dst] = rotl64_32(A[src], RHO[src]);
            }
        }

        for (uint y = 0; y < 5; y++) {
            for (uint x = 0; x < 5; x++) {
                A[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);
            }
        }

        A[0].x ^= uint(RC[round]);
        A[0].y ^= uint(RC[round] >> 32);
    }
}

inline void absorb_and_permute32(device const Params& p, ulong nonce, thread uint2 A[25]) {
    for (uint i = 0; i < 25; i++) A[i] = uint2(0, 0);

    A[0] = uint2(p.challenge32[0], p.challenge32[1]);
    A[1] = uint2(p.challenge32[2], p.challenge32[3]);
    A[2] = uint2(p.challenge32[4], p.challenge32[5]);
    A[3] = uint2(p.challenge32[6], p.challenge32[7]);
    // ABI uint256 nonce is 24 zero bytes followed by the 8-byte big-endian nonce.
    // As Keccak lanes are little-endian, lane 7 becomes byteswap64(nonce).
    uint nonceLo = uint(nonce);
    uint nonceHi = uint(nonce >> 32);
    A[7] = uint2(bswap32(nonceHi), bswap32(nonceLo));

    A[8].x ^= 0x01U;
    A[16].y ^= 0x80000000U;

    keccakf32(A);
}

inline void digest_words32(thread uint2 A[25], thread uint out[8]) {
    out[0] = bswap32(A[0].x);
    out[1] = bswap32(A[0].y);
    out[2] = bswap32(A[1].x);
    out[3] = bswap32(A[1].y);
    out[4] = bswap32(A[2].x);
    out[5] = bswap32(A[2].y);
    out[6] = bswap32(A[3].x);
    out[7] = bswap32(A[3].y);
}

inline bool lt256_32(thread const uint d[8], device const uint t[8]) {
    for (uint i = 0; i < 8; i++) {
        if (d[i] < t[i]) return true;
        if (d[i] > t[i]) return false;
    }
    return false;
}

kernel void scan_hash32(device const Params& p [[buffer(0)]],
                        device Result& r [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    ulong baseIndex = ((ulong)gid) * p.iters;

    for (ulong i = 0; i < p.iters; i++) {
        ulong index = baseIndex + i;
        if (index >= p.total) return;
        ulong nonce = p.start + index * p.stride;

        uint2 A[25];
        absorb_and_permute32(p, nonce, A);

        uint d0 = bswap32(A[0].x);
        uint d1 = bswap32(A[0].y);
        bool hit = d0 < p.target32[0] || (d0 == p.target32[0] && d1 < p.target32[1]);

        if (!hit && d0 == p.target32[0] && d1 == p.target32[1]) {
            uint digest[8];
            digest_words32(A, digest);
            hit = lt256_32(digest, p.target32);
        }

        if (hit) {
            if (atomic_exchange_explicit(&r.found, 1, memory_order_relaxed) == 0) {
                uint digest[8];
                digest_words32(A, digest);
                r.nonce = nonce;
                r.digest[0] = (ulong(digest[0]) << 32) | ulong(digest[1]);
                r.digest[1] = (ulong(digest[2]) << 32) | ulong(digest[3]);
                r.digest[2] = (ulong(digest[4]) << 32) | ulong(digest[5]);
                r.digest[3] = (ulong(digest[6]) << 32) | ulong(digest[7]);
            }
            return;
        }
    }
}
)METAL";

static uint8_t hex_byte(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
    throw std::runtime_error("invalid hex");
}

static std::vector<uint8_t> parse_hex(std::string s, size_t want) {
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    if (s.size() != want * 2) {
        throw std::runtime_error("hex length mismatch");
    }
    std::vector<uint8_t> out(want);
    for (size_t i = 0; i < want; i++) {
        out[i] = static_cast<uint8_t>((hex_byte(s[2 * i]) << 4) | hex_byte(s[2 * i + 1]));
    }
    return out;
}

static uint64_t load_le64(const uint8_t *p) {
    uint64_t x = 0;
    for (int i = 7; i >= 0; i--) x = (x << 8) | p[i];
    return x;
}

static uint64_t load_be64(const uint8_t *p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x = (x << 8) | p[i];
    return x;
}

static uint32_t load_le32(const uint8_t *p) {
    uint32_t x = 0;
    for (int i = 3; i >= 0; i--) x = (x << 8) | p[i];
    return x;
}

static uint32_t load_be32(const uint8_t *p) {
    uint32_t x = 0;
    for (int i = 0; i < 4; i++) x = (x << 8) | p[i];
    return x;
}

static std::string hex64(uint64_t x) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(x));
    return std::string(buf);
}

static uint64_t parse_u64(const std::string &s) {
    size_t idx = 0;
    int base = 10;
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) base = 16;
    uint64_t v = std::stoull(s, &idx, base);
    if (idx != s.size()) throw std::runtime_error("invalid integer: " + s);
    return v;
}

static int cmp256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 0; i < 4; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static bool lt256(const uint64_t a[4], const uint64_t b[4]) {
    return cmp256(a, b) < 0;
}

static std::string arg_value(int &i, int argc, char **argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        std::string challenge_hex;
        std::string target_hex;
        std::string share_target_hex;
        uint64_t start = 0;
        uint64_t total = 0;
        uint64_t batch = 1ULL << 23;
        uint64_t iters = 8;
        uint64_t groupSize = 64;
        uint64_t stride = 1;
        uint64_t inflight = 2;
        double seconds = 5.0;
        std::string kernel = "compact";

        try {
            for (int i = 1; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--challenge") challenge_hex = arg_value(i, argc, argv);
                else if (a == "--target") target_hex = arg_value(i, argc, argv);
                else if (a == "--share-target") share_target_hex = arg_value(i, argc, argv);
                else if (a == "--start") start = parse_u64(arg_value(i, argc, argv));
                else if (a == "--total") total = parse_u64(arg_value(i, argc, argv));
                else if (a == "--batch") batch = parse_u64(arg_value(i, argc, argv));
                else if (a == "--iters") iters = parse_u64(arg_value(i, argc, argv));
                else if (a == "--group") groupSize = parse_u64(arg_value(i, argc, argv));
                else if (a == "--stride") stride = parse_u64(arg_value(i, argc, argv));
                else if (a == "--inflight") inflight = parse_u64(arg_value(i, argc, argv));
                else if (a == "--kernel") kernel = arg_value(i, argc, argv);
                else if (a == "--seconds") seconds = std::stod(arg_value(i, argc, argv));
                else if (a == "--help" || a == "-h") {
                    std::cout << "usage: hash_gpu_metal --challenge 0x<32b> --target 0x<32b> [--share-target 0x<32b>] [--seconds 5] [--start 0] [--total N]\n";
                    return 0;
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (challenge_hex.empty() || target_hex.empty()) {
                throw std::runtime_error("--challenge and --target are required");
            }
            if (iters == 0 || batch == 0 || groupSize == 0 || stride == 0 || inflight == 0) {
                throw std::runtime_error("--iters, --batch, --group, --stride, and --inflight must be positive");
            }

            auto challenge = parse_hex(challenge_hex, 32);
            auto target = parse_hex(target_hex, 32);
            std::vector<uint8_t> effective_target = target;
            uint64_t networkTarget[4]{};
            uint64_t shareTarget[4]{};
            bool haveShareTarget = !share_target_hex.empty();

            for (int i = 0; i < 4; i++) {
                networkTarget[i] = load_be64(target.data() + i * 8);
            }
            if (haveShareTarget) {
                auto share_target = parse_hex(share_target_hex, 32);
                for (int i = 0; i < 4; i++) {
                    shareTarget[i] = load_be64(share_target.data() + i * 8);
                }
                if (cmp256(networkTarget, shareTarget) < 0) {
                    effective_target = share_target;
                }
            }

            ParamsHost params{};
            for (int i = 0; i < 4; i++) {
                params.challenge[i] = load_le64(challenge.data() + i * 8);
                params.target[i] = load_be64(effective_target.data() + i * 8);
            }
            for (int i = 0; i < 8; i++) {
                params.challenge32[i] = load_le32(challenge.data() + i * 4);
                params.target32[i] = load_be32(effective_target.data() + i * 4);
            }
            params.iters = iters;
            params.stride = stride;

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("no Metal device");

            NSError *error = nil;
            NSString *src = [NSString stringWithUTF8String:kMetalSource];
            id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
            if (!library) {
                std::string msg = error ? [[error localizedDescription] UTF8String] : "unknown Metal compile error";
                throw std::runtime_error(msg);
            }
            NSString *kernelName = nil;
            if (kernel == "u64") {
                kernelName = @"scan_hash";
            } else if (kernel == "compact") {
                kernelName = @"scan_hash_compact";
            } else if (kernel == "scalar") {
                kernelName = @"scan_hash_scalar";
            } else if (kernel == "u32") {
                kernelName = @"scan_hash32";
            } else {
                throw std::runtime_error("--kernel must be u32, u64, compact, or scalar");
            }
            id<MTLFunction> fn = [library newFunctionWithName:kernelName];
            id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pipe) {
                std::string msg = error ? [[error localizedDescription] UTF8String] : "pipeline error";
                throw std::runtime_error(msg);
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];
            std::vector<id<MTLBuffer>> paramsBufs;
            std::vector<id<MTLBuffer>> resultBufs;
            for (uint64_t i = 0; i < inflight; i++) {
                paramsBufs.push_back([device newBufferWithLength:sizeof(ParamsHost) options:MTLResourceStorageModeShared]);
                resultBufs.push_back([device newBufferWithLength:sizeof(ResultHost) options:MTLResourceStorageModeShared]);
            }

            const auto t0 = std::chrono::steady_clock::now();
            uint64_t checked = 0;
            uint64_t cursor = start;
            bool found = false;
            ResultHost finalResult{};

            std::cout << "Metal device: " << [[device name] UTF8String] << "\n";

            struct ActiveBatch {
                id<MTLCommandBuffer> cmd;
                id<MTLBuffer> resultBuf;
                uint64_t count;
            };

            while (true) {
                std::vector<ActiveBatch> active;
                active.reserve(static_cast<size_t>(inflight));

                for (uint64_t slot = 0; slot < inflight; slot++) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - t0).count();
                    if (seconds > 0 && elapsed >= seconds) break;
                    if (total > 0 && checked >= total) break;

                    uint64_t remaining = total > 0 ? total - checked : batch;
                    uint64_t thisBatch = std::min(batch, remaining);
                    if (thisBatch == 0) break;

                    params.start = cursor;
                    params.total = thisBatch;
                    std::memcpy([paramsBufs[slot] contents], &params, sizeof(params));

                    ResultHost zero{};
                    std::memcpy([resultBufs[slot] contents], &zero, sizeof(zero));

                    uint64_t threads = (thisBatch + iters - 1) / iters;
                    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(threads), 1, 1);
                    NSUInteger w = static_cast<NSUInteger>(std::min<uint64_t>(groupSize, pipe.maxTotalThreadsPerThreadgroup));
                    MTLSize group = MTLSizeMake(w, 1, 1);

                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pipe];
                    [enc setBuffer:paramsBufs[slot] offset:0 atIndex:0];
                    [enc setBuffer:resultBufs[slot] offset:0 atIndex:1];
                    [enc dispatchThreads:grid threadsPerThreadgroup:group];
                    [enc endEncoding];
                    [cmd commit];

                    active.push_back({cmd, resultBufs[slot], thisBatch});
                    if (thisBatch > UINT64_MAX / stride) throw std::runtime_error("stride overflow");
                    cursor += thisBatch * stride;
                    checked += thisBatch;
                }

                if (active.empty()) break;

                for (const auto &batchInfo : active) {
                    [batchInfo.cmd waitUntilCompleted];
                    if (batchInfo.cmd.error) {
                        std::string msg = [[batchInfo.cmd.error localizedDescription] UTF8String];
                        throw std::runtime_error(msg);
                    }

                    auto *r = reinterpret_cast<ResultHost *>([batchInfo.resultBuf contents]);
                    if (r->found) {
                        finalResult = *r;
                        found = true;
                    }
                }
                if (found) break;
            }

            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double rate = elapsed > 0 ? static_cast<double>(checked) / elapsed : 0.0;
            std::cout << "checked: " << checked << "\n";
            std::cout << "elapsed: " << elapsed << "s\n";
            std::cout << "rate: " << static_cast<uint64_t>(rate) << " H/s\n";

            if (found) {
                bool isFound = lt256(finalResult.digest, networkTarget);
                bool isShare = haveShareTarget && lt256(finalResult.digest, shareTarget);
                if (!isFound && !isShare) {
                    std::cout << "not found\n";
                    return 1;
                }
                std::cout << (isFound ? "FOUND" : "SHARE") << "\n";
                std::cout << "nonce: " << finalResult.nonce << "\n";
                std::cout << "digest: 0x";
                for (int i = 0; i < 4; i++) std::cout << hex64(finalResult.digest[i]);
                std::cout << "\n";
                std::cout << "calldata: 0x4d474898";
                std::cout << std::string(48, '0') << hex64(finalResult.nonce) << "\n";
                return 0;
            }
            std::cout << "not found\n";
            return 1;
        } catch (const std::exception &e) {
            std::cerr << "error: " << e.what() << "\n";
            return 2;
        }
    }
}
