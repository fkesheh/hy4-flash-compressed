"""Codec for SVQ1: HYB1 sign factors with learned 4-bit scale codebooks."""
import argparse
from pathlib import Path

import numpy as np

import compress

CODEBOOK_SIZE = 16
MAGIC = b"SVQ1"


def _quantize_scales(values):
    values = np.asarray(values, dtype=np.float32)
    centroids = np.quantile(values, np.linspace(0.0, 1.0, CODEBOOK_SIZE)).astype(np.float32)
    for _ in range(100):
        labels = np.abs(values[:, None] - centroids[None, :]).argmin(axis=1)
        sums = np.bincount(labels, weights=values, minlength=CODEBOOK_SIZE)
        counts = np.bincount(labels, minlength=CODEBOOK_SIZE)
        updated = np.sort(np.where(counts, sums / np.maximum(counts, 1), centroids))
        if np.allclose(centroids, updated, rtol=0, atol=1e-9):
            centroids = updated
            break
        centroids = updated
    centroids = centroids.astype("<f2")
    labels = np.abs(values[:, None] - centroids.astype(np.float32)[None, :]).argmin(axis=1)
    return centroids, labels.astype(np.uint8)


def _pack_nibbles(labels):
    labels = np.asarray(labels, dtype=np.uint8)
    packed = np.zeros((len(labels) + 1) // 2, dtype=np.uint8)
    pairs = len(labels) // 2
    packed[:pairs] = (labels[:2 * pairs:2] << 4) | labels[1:2 * pairs:2]
    if len(labels) & 1:
        packed[-1] = labels[-1] << 4
    return packed.tobytes()


def _unpack_nibbles(data, count):
    packed = np.frombuffer(data, dtype=np.uint8)
    labels = np.empty(len(packed) * 2, dtype=np.uint8)
    labels[::2] = packed >> 4
    labels[1::2] = packed & 15
    return labels[:count]


def encode_hyb1(data):
    """Convert one per-row-scale HYB1 payload into a self-contained SVQ1 payload."""
    magic, rows, cols, rank, per_row = compress.HEADER.unpack_from(data)
    if magic != b"HYB1" or per_row != 1:
        raise ValueError("SVQ1 expects a per-row-scale HYB1 payload")
    offset = compress.HEADER.size
    left_bytes = (rows * rank + 7) // 8
    right_bytes = (rank * cols + 7) // 8
    signs = data[offset:offset + left_bytes + right_bytes]
    offset += left_bytes + right_bytes
    scales = np.frombuffer(data, dtype="<f2", count=rows + rank + cols,
                           offset=offset).astype(np.float32)
    parts = [compress.HEADER.pack(MAGIC, rows, cols, rank, per_row), signs]
    cursor = 0
    for count in (rows, rank, cols):
        centroids, labels = _quantize_scales(scales[cursor:cursor + count])
        parts.extend((centroids.tobytes(), _pack_nibbles(labels)))
        cursor += count
    return b"".join(parts)


def decode_factors(data):
    """Decode SVQ1 to float32 factors without reference weights."""
    magic, rows, cols, rank, per_row = compress.HEADER.unpack_from(data)
    if magic != MAGIC or per_row != 1:
        raise ValueError("Invalid SVQ1 payload")
    offset = compress.HEADER.size
    left_bytes = (rows * rank + 7) // 8
    right_bytes = (rank * cols + 7) // 8
    signs = data[offset:offset + left_bytes + right_bytes]
    if len(signs) != left_bytes + right_bytes:
        raise ValueError("Truncated SVQ1 signs")
    offset += left_bytes + right_bytes
    scales = []
    for count in (rows, rank, cols):
        end = offset + CODEBOOK_SIZE * 2
        centroids = np.frombuffer(data, dtype="<f2", count=CODEBOOK_SIZE,
                                  offset=offset).astype(np.float32)
        if end > len(data):
            raise ValueError("Truncated SVQ1 codebook")
        offset = end
        index_bytes = (count + 1) // 2
        labels = _unpack_nibbles(data[offset:offset + index_bytes], count)
        if len(labels) != count:
            raise ValueError("Truncated SVQ1 scale indices")
        scales.extend(centroids[labels])
        offset += index_bytes
    if offset != len(data):
        raise ValueError("Trailing bytes in SVQ1 payload")
    hyb = compress.HEADER.pack(b"HYB1", rows, cols, rank, per_row) + signs
    hyb += np.asarray(scales, dtype="<f2").tobytes()
    return compress.unpack(hyb)


def decode_matrix(data):
    left, right = decode_factors(data)
    return left @ right


def encode_file(source, destination):
    Path(destination).write_bytes(encode_hyb1(Path(source).read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="per-row-scale HYB1 input")
    parser.add_argument("output", type=Path, help="SVQ1 output")
    args = parser.parse_args()
    encode_file(args.source, args.output)


if __name__ == "__main__":
    main()
