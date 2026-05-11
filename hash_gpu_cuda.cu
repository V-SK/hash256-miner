// HASH CUDA nonce scanner for NVIDIA GPUs (v2 — register-resident keccak).
//
// v2 changes vs v1:
//   * Keccak-f[1600] rewritten with 25 named scalars (a00..a44) and 25 named
//     temporaries (b00..b44). No array indexing, no __constant__ permutation
//     tables, no chain swap — every lane is a true register on sm_86.
//   * Theta + Rho + Pi fused into a single ROL64(a^D, rho) expression per
//     lane, then Chi + Iota merged on the first column of each row.
//   * 24 rounds emitted by an explicit macro chain so every RC value is a
//     literal in the SASS.
//   * scan_hash_cuda annotated with __launch_bounds__(256, 4) to drive the
//     register allocator toward the desired occupancy on Ampere.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_86 hash_gpu_cuda.cu -o hash_gpu_cuda
//
// CLI/output kept stable so pool wrappers can parse records identically.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct ParamsHost {
    uint64_t challenge[4]; // Keccak absorb lanes for bytes32 challenge.
    uint64_t target[4];    // Big-endian uint256 words.
    uint64_t start;
    uint64_t total;
    uint64_t iters;
    uint64_t stride;
};

struct ResultHost {
    uint32_t found;
    uint32_t pad;
    uint64_t nonce;
    uint64_t digest[4]; // Big-endian uint256 words.
};

__host__ __device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

// One Keccak-f[1600] round, operating on 25 named scalars a00..a44.
// Theta D is computed, then fused into a 25-way ROL of (a^D) producing
// b00..b44 with the Rho rotations and Pi permutation built in. Chi reads
// from b and writes back to a. The Iota constant is XOR'd into a00 in the
// same statement as its Chi result to avoid an extra round-trip.
#define KECCAK_ROUND(RC) do {                                          \
    uint64_t C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;                         \
    uint64_t C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;                         \
    uint64_t C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;                         \
    uint64_t C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;                         \
    uint64_t C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;                         \
    uint64_t D0 = C4 ^ ROL64(C1, 1);                                   \
    uint64_t D1 = C0 ^ ROL64(C2, 1);                                   \
    uint64_t D2 = C1 ^ ROL64(C3, 1);                                   \
    uint64_t D3 = C2 ^ ROL64(C4, 1);                                   \
    uint64_t D4 = C3 ^ ROL64(C0, 1);                                   \
    uint64_t b00 = (a00 ^ D0);                                         \
    uint64_t b10 = ROL64(a01 ^ D1,  1);                                \
    uint64_t b20 = ROL64(a02 ^ D2, 62);                                \
    uint64_t b05 = ROL64(a03 ^ D3, 28);                                \
    uint64_t b15 = ROL64(a04 ^ D4, 27);                                \
    uint64_t b16 = ROL64(a05 ^ D0, 36);                                \
    uint64_t b01 = ROL64(a06 ^ D1, 44);                                \
    uint64_t b11 = ROL64(a07 ^ D2,  6);                                \
    uint64_t b21 = ROL64(a08 ^ D3, 55);                                \
    uint64_t b06 = ROL64(a09 ^ D4, 20);                                \
    uint64_t b07 = ROL64(a10 ^ D0,  3);                                \
    uint64_t b17 = ROL64(a11 ^ D1, 10);                                \
    uint64_t b02 = ROL64(a12 ^ D2, 43);                                \
    uint64_t b12 = ROL64(a13 ^ D3, 25);                                \
    uint64_t b22 = ROL64(a14 ^ D4, 39);                                \
    uint64_t b23 = ROL64(a15 ^ D0, 41);                                \
    uint64_t b08 = ROL64(a16 ^ D1, 45);                                \
    uint64_t b18 = ROL64(a17 ^ D2, 15);                                \
    uint64_t b03 = ROL64(a18 ^ D3, 21);                                \
    uint64_t b13 = ROL64(a19 ^ D4,  8);                                \
    uint64_t b14 = ROL64(a20 ^ D0, 18);                                \
    uint64_t b24 = ROL64(a21 ^ D1,  2);                                \
    uint64_t b09 = ROL64(a22 ^ D2, 61);                                \
    uint64_t b19 = ROL64(a23 ^ D3, 56);                                \
    uint64_t b04 = ROL64(a24 ^ D4, 14);                                \
    a00 = b00 ^ ((~b01) & b02) ^ (RC);                                 \
    a01 = b01 ^ ((~b02) & b03);                                        \
    a02 = b02 ^ ((~b03) & b04);                                        \
    a03 = b03 ^ ((~b04) & b00);                                        \
    a04 = b04 ^ ((~b00) & b01);                                        \
    a05 = b05 ^ ((~b06) & b07);                                        \
    a06 = b06 ^ ((~b07) & b08);                                        \
    a07 = b07 ^ ((~b08) & b09);                                        \
    a08 = b08 ^ ((~b09) & b05);                                        \
    a09 = b09 ^ ((~b05) & b06);                                        \
    a10 = b10 ^ ((~b11) & b12);                                        \
    a11 = b11 ^ ((~b12) & b13);                                        \
    a12 = b12 ^ ((~b13) & b14);                                        \
    a13 = b13 ^ ((~b14) & b10);                                        \
    a14 = b14 ^ ((~b10) & b11);                                        \
    a15 = b15 ^ ((~b16) & b17);                                        \
    a16 = b16 ^ ((~b17) & b18);                                        \
    a17 = b17 ^ ((~b18) & b19);                                        \
    a18 = b18 ^ ((~b19) & b15);                                        \
    a19 = b19 ^ ((~b15) & b16);                                        \
    a20 = b20 ^ ((~b21) & b22);                                        \
    a21 = b21 ^ ((~b22) & b23);                                        \
    a22 = b22 ^ ((~b23) & b24);                                        \
    a23 = b23 ^ ((~b24) & b20);                                        \
    a24 = b24 ^ ((~b20) & b21);                                        \
} while (0)

__device__ __forceinline__ void hash_nonce_digest(const ParamsHost& p,
                                                  uint64_t nonce,
                                                  uint64_t digest_be[4]) {
    // Absorb a single 64-byte block: 32-byte challenge (lanes 0..3),
    // 32-byte big-endian nonce starting at lane 4. Only the low 64 bits of
    // the nonce are non-zero, occupying lane 7 after byte swap. Ethereum
    // Keccak padding writes 0x01 at the byte right after the message and
    // sets the top bit of the last rate byte (lane 16).
    uint64_t a00 = p.challenge[0];
    uint64_t a01 = p.challenge[1];
    uint64_t a02 = p.challenge[2];
    uint64_t a03 = p.challenge[3];
    uint64_t a04 = 0;
    uint64_t a05 = 0;
    uint64_t a06 = 0;
    uint64_t a07 = bswap64(nonce);
    uint64_t a08 = 0x0000000000000001ULL;
    uint64_t a09 = 0;
    uint64_t a10 = 0;
    uint64_t a11 = 0;
    uint64_t a12 = 0;
    uint64_t a13 = 0;
    uint64_t a14 = 0;
    uint64_t a15 = 0;
    uint64_t a16 = 0x8000000000000000ULL;
    uint64_t a17 = 0;
    uint64_t a18 = 0;
    uint64_t a19 = 0;
    uint64_t a20 = 0;
    uint64_t a21 = 0;
    uint64_t a22 = 0;
    uint64_t a23 = 0;
    uint64_t a24 = 0;

    KECCAK_ROUND(0x0000000000000001ULL);
    KECCAK_ROUND(0x0000000000008082ULL);
    KECCAK_ROUND(0x800000000000808aULL);
    KECCAK_ROUND(0x8000000080008000ULL);
    KECCAK_ROUND(0x000000000000808bULL);
    KECCAK_ROUND(0x0000000080000001ULL);
    KECCAK_ROUND(0x8000000080008081ULL);
    KECCAK_ROUND(0x8000000000008009ULL);
    KECCAK_ROUND(0x000000000000008aULL);
    KECCAK_ROUND(0x0000000000000088ULL);
    KECCAK_ROUND(0x0000000080008009ULL);
    KECCAK_ROUND(0x000000008000000aULL);
    KECCAK_ROUND(0x000000008000808bULL);
    KECCAK_ROUND(0x800000000000008bULL);
    KECCAK_ROUND(0x8000000000008089ULL);
    KECCAK_ROUND(0x8000000000008003ULL);
    KECCAK_ROUND(0x8000000000008002ULL);
    KECCAK_ROUND(0x8000000000000080ULL);
    KECCAK_ROUND(0x000000000000800aULL);
    KECCAK_ROUND(0x800000008000000aULL);
    KECCAK_ROUND(0x8000000080008081ULL);
    KECCAK_ROUND(0x8000000000008080ULL);
    KECCAK_ROUND(0x0000000080000001ULL);
    KECCAK_ROUND(0x8000000080008008ULL);

    digest_be[0] = bswap64(a00);
    digest_be[1] = bswap64(a01);
    digest_be[2] = bswap64(a02);
    digest_be[3] = bswap64(a03);
}

__device__ __forceinline__ bool lt256(const uint64_t d[4], const uint64_t t[4]) {
    #pragma unroll 4
    for (uint32_t i = 0; i < 4; i++) {
        if (d[i] < t[i]) return true;
        if (d[i] > t[i]) return false;
    }
    return false;
}

__global__ __launch_bounds__(256, 4)
void scan_hash_cuda(const ParamsHost* __restrict__ p,
                    ResultHost* __restrict__ r) {
    const uint64_t iters  = p->iters;
    const uint64_t total  = p->total;
    const uint64_t start  = p->start;
    const uint64_t stride = p->stride;
    const uint64_t t0     = p->target[0];
    const uint64_t t1     = p->target[1];
    const uint64_t t2     = p->target[2];
    const uint64_t t3     = p->target[3];

    uint64_t baseIndex = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * iters;

    #pragma unroll 1
    for (uint64_t i = 0; i < iters; i++) {
        uint64_t index = baseIndex + i;
        if (index >= total) return;

        uint64_t nonce = start + index * stride;
        uint64_t digest[4];
        hash_nonce_digest(*p, nonce, digest);

        // Inline lt256 against constants pulled into registers above so the
        // compiler doesn't reload them every iteration.
        bool below = false;
        if      (digest[0] < t0) below = true;
        else if (digest[0] > t0) below = false;
        else if (digest[1] < t1) below = true;
        else if (digest[1] > t1) below = false;
        else if (digest[2] < t2) below = true;
        else if (digest[2] > t2) below = false;
        else if (digest[3] < t3) below = true;
        else if (digest[3] > t3) below = false;

        if (below) {
            if (atomicCAS(&r->found, 0U, 1U) == 0U) {
                r->nonce = nonce;
                r->digest[0] = digest[0];
                r->digest[1] = digest[1];
                r->digest[2] = digest[2];
                r->digest[3] = digest[3];
            }
            return;
        }
    }
}

static void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

static uint8_t hex_byte(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
    throw std::runtime_error("invalid hex");
}

static std::vector<uint8_t> parse_hex(std::string s, size_t want) {
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    if (s.size() != want * 2) throw std::runtime_error("hex length mismatch");
    std::vector<uint8_t> out(want);
    for (size_t i = 0; i < want; i++) {
        out[i] = static_cast<uint8_t>((hex_byte(s[2 * i]) << 4) | hex_byte(s[2 * i + 1]));
    }
    return out;
}

static uint64_t load_le64(const uint8_t* p) {
    uint64_t x = 0;
    for (int i = 7; i >= 0; i--) x = (x << 8) | p[i];
    return x;
}

static uint64_t load_be64(const uint8_t* p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x = (x << 8) | p[i];
    return x;
}

static std::string hex64(uint64_t x) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(x));
    return std::string(buf);
}

static int cmp256_host(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 0; i < 4; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static bool lt256_host(const uint64_t d[4], const uint64_t t[4]) {
    return cmp256_host(d, t) < 0;
}

static uint64_t parse_u64(const std::string& s) {
    size_t idx = 0;
    int base = 10;
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) base = 16;
    uint64_t v = std::stoull(s, &idx, base);
    if (idx != s.size()) throw std::runtime_error("invalid integer: " + s);
    return v;
}

static std::string arg_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

int main(int argc, char** argv) {
    std::string challenge_hex;
    std::string target_hex;
    std::string share_target_hex;
    uint64_t start = 0;
    uint64_t total = 0;
    uint64_t batch = 1ULL << 24;
    uint64_t iters = 8;
    uint64_t groupSize = 256;
    uint64_t stride = 1;
    double seconds = 5.0;
    int numStreams = 2; // v2: double-stream pipelining
    int device = 0;

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
            else if (a == "--seconds") seconds = std::stod(arg_value(i, argc, argv));
            else if (a == "--streams") numStreams = static_cast<int>(parse_u64(arg_value(i, argc, argv)));
            else if (a == "--device") device = static_cast<int>(parse_u64(arg_value(i, argc, argv)));
            else if (a == "--kernel" || a == "--inflight") (void)arg_value(i, argc, argv); // accepted for wrapper compatibility.
            else if (a == "--help" || a == "-h") {
                std::cout << "usage: hash_gpu_cuda --challenge 0x<32b> --target 0x<32b> [--share-target 0x<32b>] [--seconds 5] [--start 0] [--total N] [--batch N] [--iters N] [--group N] [--streams 2] [--device 0]\n";
                return 0;
            } else {
                throw std::runtime_error("unknown arg: " + a);
            }
        }
        if (challenge_hex.empty() || target_hex.empty()) throw std::runtime_error("--challenge and --target are required");
        if (iters == 0 || batch == 0 || groupSize == 0 || stride == 0) {
            throw std::runtime_error("--iters, --batch, --group, and --stride must be positive");
        }
        if (numStreams < 1) numStreams = 1;
        if (numStreams > 4) numStreams = 4;

        auto challenge = parse_hex(challenge_hex, 32);
        auto target = parse_hex(target_hex, 32);
        std::vector<uint8_t> share_target;
        const bool has_share_target = !share_target_hex.empty();
        if (has_share_target) share_target = parse_hex(share_target_hex, 32);

        ParamsHost hostParams{};
        uint64_t networkTarget[4]{};
        uint64_t shareTarget[4]{};
        uint64_t effectiveTarget[4]{};
        for (int i = 0; i < 4; i++) {
            hostParams.challenge[i] = load_le64(challenge.data() + i * 8);
            networkTarget[i] = load_be64(target.data() + i * 8);
            if (has_share_target) shareTarget[i] = load_be64(share_target.data() + i * 8);
            effectiveTarget[i] = networkTarget[i];
        }
        if (has_share_target && cmp256_host(shareTarget, networkTarget) > 0) {
            for (int i = 0; i < 4; i++) effectiveTarget[i] = shareTarget[i];
        }
        for (int i = 0; i < 4; i++) {
            hostParams.target[i] = effectiveTarget[i];
        }
        hostParams.iters = iters;
        hostParams.stride = stride;

        int deviceCount = 0;
        cuda_check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
        if (device < 0 || device >= deviceCount) throw std::runtime_error("invalid CUDA device index");
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

        // Per-stream device memory + pinned host result buffers. Pinned host
        // memory lets cudaMemcpyAsync overlap with kernel execution.
        std::vector<cudaStream_t> streams(numStreams);
        std::vector<ParamsHost*>  dParams(numStreams, nullptr);
        std::vector<ResultHost*>  dResult(numStreams, nullptr);
        std::vector<ResultHost*>  hResult(numStreams, nullptr); // pinned
        std::vector<uint64_t>     batchStart(numStreams, 0);
        std::vector<uint64_t>     batchSize(numStreams, 0);
        std::vector<bool>         inFlight(numStreams, false);

        for (int s = 0; s < numStreams; s++) {
            cuda_check(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking), "cudaStreamCreate");
            cuda_check(cudaMalloc(&dParams[s], sizeof(ParamsHost)),  "cudaMalloc params");
            cuda_check(cudaMalloc(&dResult[s], sizeof(ResultHost)),  "cudaMalloc result");
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&hResult[s]), sizeof(ResultHost), cudaHostAllocDefault), "cudaHostAlloc result");
        }

        const auto t0 = std::chrono::steady_clock::now();
        uint64_t checked = 0;
        uint64_t submitted = 0;
        uint64_t cursor = start;
        bool found = false;
        ResultHost finalResult{};

        std::cout << "CUDA device: " << device << " " << prop.name << "\n";

        auto drain_stream = [&](int s) {
            if (!inFlight[s]) return;
            cuda_check(cudaStreamSynchronize(streams[s]), "cudaStreamSynchronize");
            inFlight[s] = false;
            ResultHost rh = *hResult[s];
            checked += batchSize[s];
            if (rh.found && !found) {
                finalResult = rh;
                found = true;
            }
        };

        bool stopSubmitting = false;
        int nextStream = 0;
        while (!stopSubmitting || std::any_of(inFlight.begin(), inFlight.end(), [](bool b){ return b; })) {
            // Check global stop conditions.
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            if (seconds > 0 && elapsed >= seconds) stopSubmitting = true;
            if (total > 0 && submitted >= total) stopSubmitting = true;
            if (found) stopSubmitting = true;

            int s = nextStream;
            if (!stopSubmitting) {
                // If the slot is busy, drain it before reusing.
                if (inFlight[s]) drain_stream(s);
                if (found) { stopSubmitting = true; continue; }

                uint64_t remaining = total > 0 ? (total > submitted ? total - submitted : 0) : batch;
                uint64_t thisBatch = std::min(batch, remaining);
                if (thisBatch == 0) { stopSubmitting = true; continue; }

                hostParams.start = cursor;
                hostParams.total = thisBatch;
                ResultHost zero{};
                *hResult[s] = zero;
                batchStart[s] = cursor;
                batchSize[s]  = thisBatch;

                cuda_check(cudaMemcpyAsync(dParams[s], &hostParams, sizeof(hostParams), cudaMemcpyHostToDevice, streams[s]), "cudaMemcpyAsync params");
                cuda_check(cudaMemcpyAsync(dResult[s], hResult[s],  sizeof(ResultHost), cudaMemcpyHostToDevice, streams[s]), "cudaMemcpyAsync zero");

                uint64_t threads = (thisBatch + iters - 1) / iters;
                uint64_t blocks64 = (threads + groupSize - 1) / groupSize;
                if (blocks64 > static_cast<uint64_t>(INT32_MAX)) throw std::runtime_error("too many CUDA blocks; lower --batch");

                scan_hash_cuda<<<static_cast<unsigned int>(blocks64),
                                 static_cast<unsigned int>(groupSize),
                                 0,
                                 streams[s]>>>(dParams[s], dResult[s]);
                cuda_check(cudaGetLastError(), "kernel launch");

                cuda_check(cudaMemcpyAsync(hResult[s], dResult[s], sizeof(ResultHost), cudaMemcpyDeviceToHost, streams[s]), "cudaMemcpyAsync back");
                inFlight[s] = true;

                if (thisBatch > UINT64_MAX / stride) throw std::runtime_error("stride overflow");
                cursor += thisBatch * stride;
                submitted += thisBatch;

                nextStream = (nextStream + 1) % numStreams;
            } else {
                // Drain any still-pending streams once submission stops.
                for (int sd = 0; sd < numStreams; sd++) drain_stream(sd);
                break;
            }
        }

        for (int s = 0; s < numStreams; s++) {
            cudaFree(dParams[s]);
            cudaFree(dResult[s]);
            cudaFreeHost(hResult[s]);
            cudaStreamDestroy(streams[s]);
        }

        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double rate = elapsed > 0 ? static_cast<double>(checked) / elapsed : 0.0;
        std::cout << "checked: " << checked << "\n";
        std::cout << "elapsed: " << elapsed << "s\n";
        std::cout << "rate: " << static_cast<uint64_t>(rate) << " H/s\n";

        if (found) {
            const bool network_hit = lt256_host(finalResult.digest, networkTarget);
            const bool share_hit = has_share_target && lt256_host(finalResult.digest, shareTarget);
            if (!network_hit && !share_hit) throw std::runtime_error("candidate did not satisfy target");
            if (network_hit) {
                std::cout << "FOUND\n";
            } else if (share_hit) {
                std::cout << "SHARE\n";
            }
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
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
