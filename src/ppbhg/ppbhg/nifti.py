import gzip
import struct
from pathlib import Path

import numpy as np


_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def load_nifti(path):
    """Minimal NIfTI-1 reader for uncompressed or gzipped scalar volumes."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(348)
        if len(header) != 348:
            raise ValueError(f"{path} is too small to be a NIfTI file")

        endian = "<"
        sizeof_hdr = struct.unpack("<i", header[:4])[0]
        if sizeof_hdr != 348:
            sizeof_hdr = struct.unpack(">i", header[:4])[0]
            endian = ">"
        if sizeof_hdr != 348:
            raise ValueError(f"{path} does not look like a NIfTI-1 file")

        dims = struct.unpack(endian + "8h", header[40:56])
        ndim = dims[0]
        shape = tuple(int(v) for v in dims[1 : 1 + ndim])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        bitpix = struct.unpack(endian + "h", header[72:74])[0]
        vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
        dtype = _DTYPES.get(datatype)
        if dtype is None:
            raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")

        f.seek(vox_offset)
        count = int(np.prod(shape))
        data = np.frombuffer(f.read(count * (bitpix // 8)), dtype=np.dtype(dtype).newbyteorder(endian))
    return data.reshape(shape, order="F").astype(np.float32, copy=False)
