"""Disk-backed BF16 embedding gather and GPU-resident HYB1 matrix-vector products."""
import json
import struct

import cupy as cp
import numpy as np

from compress import HEADER, packed_size


class DiskEmbedding:
    """Read already-hashed row IDs from a local Hy4-flash safetensors shard."""

    def __init__(self, path, tensor):
        with open(path, "rb") as handle:
            size, = struct.unpack("<Q", handle.read(8))
            info = json.loads(handle.read(size))[tensor]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        if (info["dtype"] != "BF16" or len(shape) != 2 or shape[1] != 96
                or end - start != shape[0] * 192):
            raise ValueError("Expected a 96-column BF16 Engram table")
        self.rows = np.memmap(path, dtype="<u2", mode="r", shape=shape, offset=8 + size + start)

    def gather(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(self.rows)):
            raise ValueError("Engram row ID out of range")
        # Transfer raw BF16 words, promote only the selected rows on the GPU.
        words = cp.asarray(self.rows[ids])
        return (words.astype(cp.uint32) << 16).view(cp.float32)


# Each CUDA block computes one output dot product directly from MSB-first signs.
_dot = cp.RawKernel(r'''
extern "C" __global__ void binary_dot(const unsigned char *signs,
    const float *x, const float *scales, int width, float *out) {
    __shared__ float sums[128];
    int tid = threadIdx.x;
    unsigned long long start = (unsigned long long)blockIdx.x * width;
    float acc = 0;
    for (int col = tid; col < width; col += blockDim.x) {
        unsigned long long bit = start + col;
        float val = x[col] * scales[col];
        acc += ((signs[bit >> 3] >> (7 - (bit & 7))) & 1) ? val : -val;
    }
    sums[tid] = acc;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    if (!tid) out[blockIdx.x] = sums[0];
}
''', "binary_dot")


class PackedLinear:
    def __init__(self, data):
        magic, self.rows, self.cols, self.rank, per_row = HEADER.unpack_from(data)
        m, n, k = self.rows, self.cols, self.rank
        if (magic != b"HYB1" or per_row not in (0, 1) or not 0 < k <= min(m, n)
                or len(data) != packed_size(m, n, k, per_row)):
            raise ValueError("Invalid packed matrix")
        offset = HEADER.size

        def read(dtype, count):
            nonlocal offset
            array = np.frombuffer(data, dtype, count, offset)
            offset += array.nbytes
            return cp.asarray(array)

        self.left = read("u1", (m * k + 7) // 8)
        self.right = read("u1", (k * n + 7) // 8)
        self.row_scale = read("<f2", m) if per_row else None
        self.latent_scale = read("<f2", k)
        self.col_scale = read("<f2", n)

    def __call__(self, x):
        if x.shape != (self.cols,):
            raise ValueError("Expected one input vector of the matrix width")
        x = cp.ascontiguousarray(x, dtype=cp.float32)
        latent = cp.empty(self.rank, cp.float32)
        out = cp.empty(self.rows, cp.float32)
        _dot((self.rank,), (128,), (self.right, x, self.col_scale.astype(cp.float32),
                                     np.int32(self.cols), latent))
        _dot((self.rows,), (128,), (self.left, latent, self.latent_scale.astype(cp.float32),
                                     np.int32(self.rank), out))
        if self.row_scale is not None:
            out *= self.row_scale
        return out
