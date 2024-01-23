import struct
import numpy as np

orientations = {
    0: "Left to Right, Top to Bottom",
    1: "Right to Left, Top to Bottom",
    2: "Left to Right, Bottom to Top",
    3: "Right to Left, Bottom to Top",
    4: "Top to Bottom, Left to Right",
    5: "Top to Bottom, Right to Left",
    6: "Bottom to Top, Left to Right",
    7: "Bottom to Top, Right to Left"
}

descriptors = {
    1: "Red",
    2: "Green",
    3: "Blue",
    4: "Alpha",
    6: "Luma (Y)",
    7: "Color Difference",
    8: "Depth (Z)",
    9: "Composite Video",
    50: "RGB",
    51: "RGBA",
    52: "ABGR",
    100: "Cb, Y, Cr, Y (4:2:2)",
    102: "Cb, Y, Cr (4:4:4)",
    103: "Cb, Y, Cr, A (4:4:4:4)"
}

packings = {
    0: "Packed into 32-bit words",
    1: "Filled to 32-bit words, Padding First",
    2: "Filled to 32-bit words, Padding Last"
}

encodings = {
    0: "No encoding",
    1: "Run Length Encoding"
}

transfers = {
    1: "Printing Density",
    2: "Linear",
    3: "Logarithmic",
    4: "Unspecified Video",
    5: "SMPTE 274M",
    6: "ITU-R 709-4",
    7: "ITU-R 601-5 system B or G",
    8: "ITU-R 601-5 system M",
    9: "Composite Video (NTSC)",
    10: "Composite Video (PAL)",
    11: "Z (Linear Depth)",
    12: "Z (Homogenous Depth)"
}

colorimetries = {
    1: "Printing Density",
    4: "Unspecified Video",
    5: "SMPTE 274M",
    6: "ITU-R 709-4",
    7: "ITU-R 601-5 system B or G",
    8: "ITU-R 601-5 system M",
    9: "Composite Video (NTSC)",
    10: "Composite Video (PAL)"
}

propertymap = [
    # (field name, offset, length, type)

    ('magic', 0, 4, 'magic'),
    ('offset', 4, 4, 'I'),
    ('dpx_version', 8, 8, 'utf8'),
    ('file_size', 16, 4, 'I'),
    ('ditto', 20, 4, 'I'),
    ('filename', 36, 100, 'utf8'),
    ('timestamp', 136, 24, 'utf8'),
    ('creator', 160, 100, 'utf8'),
    ('project_name', 260, 200, 'utf8'),
    ('copyright', 460, 200, 'utf8'),
    ('encryption_key', 660, 4, 'I'),

    ('orientation', 768, 2, 'H'),
    ('image_element_count', 770, 2, 'H'),
    ('width', 772, 4, 'I'),
    ('height', 776, 4, 'I'),

    ('data_sign', 780, 4, 'I'),
    ('descriptor', 800, 1, 'B'),
    ('transfer_characteristic', 801, 1, 'B'),
    ('colorimetry', 802, 1, 'B'),
    ('depth', 803, 1, 'B'),
    ('packing', 804, 2, 'H'),
    ('encoding', 806, 2, 'H'),
    ('line_padding', 812, 4, 'I'),
    ('image_padding', 816, 4, 'I'),
    ('image_element_description', 820, 32, 'utf8'),

    ('input_device_name', 1556, 32, 'utf8'),
    ('input_device_sn', 1588, 32, 'utf8')
]


def read_dpx_meta_data(f):
    f.seek(0)
    b = f.read(4)
    magic = b.decode(encoding='unicode_escape')
    if magic != "SDPX" and magic != "XPDS":
        return None
    endianness = ">" if magic == "SDPX" else "<"

    meta = {}

    for p in propertymap:
        f.seek(p[1])
        b = f.read(p[2])
        if p[3] == 'magic':
            meta[p[0]] = b.decode(encoding='unicode_escape')
            meta['endianness'] = "be" if magic == "SDPX" else "le"
        elif p[3] == 'utf8':
            meta[p[0]] = b.decode(encoding='unicode_escape')
        elif p[3] == 'B':
            meta[p[0]] = struct.unpack(endianness + 'B', b)[0]
        elif p[3] == 'H':
            meta[p[0]] = struct.unpack(endianness + 'H', b)[0]
        elif p[3] == 'I':
            meta[p[0]] = struct.unpack(endianness + 'I', b)[0]

    return meta


def read_dpx_image_data(file, metadata):
    depth = metadata['depth']
    packing = metadata['packing']

    if depth != 10 and depth != 16:
        raise Exception(f"{depth} bits DPX are not currently supported.")
    if packing != 1 and packing != 0:
        raise Exception(f"DPX with packing {packings[packing] if packing in packings else 'unknown'} are not "
                        f"currently supported.")
    if (packing == 0 and depth == 10) or (packing == 1 and depth == 16):
        raise Exception(f"Combination of bit depth and packing type is not currently supported.")
    if metadata['encoding'] != 0:
        raise Exception(f"Unsupported encoding scheme: {encodings[metadata['encoding']] if metadata['encoding'] in encodings else 'unknown'}")
    if metadata['descriptor'] != 50:
        raise Exception(f"Only RGB DPX are supported, but detected descriptor is {descriptors[metadata['descriptor']] if metadata['descriptor'] in descriptors else 'unknown'}")

    width = metadata['width']
    height = metadata['height']
    img = np.empty((height, width, 3), dtype=int)

    file.seek(metadata['offset'])

    if depth == 10 and packing == 1:
        count = width * height
        raw_depth = 1
        raw_dtype = np.int32
    elif depth == 16 and packing == 0:
        raw_depth = 3
        count = raw_depth * width * height
        raw_dtype = np.int16
    else:
        count = width * height
        raw_depth = 1
        raw_dtype = np.int8

    raw = np.fromfile(file, dtype=np.dtype(raw_dtype), count=count, sep="")

    if metadata['endianness'] == 'be':
        raw = raw.byteswap()

    if depth == 10 and packing == 1:
        raw = raw.reshape((height, width))
        img[:, :, 0] = ((raw >> 22) & 0x000003FF)
        img[:, :, 1] = ((raw >> 12) & 0x000003FF)
        img[:, :, 2] = ((raw >> 2) & 0x000003FF)

    elif depth == 16 and packing == 0:
        raw = raw.reshape((height, width, raw_depth))
        img = raw

    return img


def write_dpx(f, image: np.ndarray, meta):
    depth = meta['depth']
    packing = meta['packing']

    endianness = ">" if meta['endianness'] == 'be' else "<"
    for p in propertymap:
        if p[0] in meta:
            f.seek(p[1])
            if p[3] == 'magic':
                bytes = ('SDPX' if meta['endianness'] == 'be' else 'XPDS').encode(encoding='UTF-8')
            elif p[3] == 'utf8':
                bytes = meta[p[0]].encode(encoding='UTF-8')
            else:
                bytes = struct.pack(endianness + p[3], meta[p[0]])
            f.write(bytes)

    if depth == 10 and packing == 1:
        raw = ((((image[:, :, 0]).astype(np.dtype(np.int32)) & 0x000003FF) << 22)
               | (((image[:, :, 1]).astype(np.dtype(np.int32)) & 0x000003FF) << 12)
               | (((image[:, :, 2]).astype(np.dtype(np.int32)) & 0x000003FF) << 2)
               )
    elif depth == 16 and packing == 0:
        raw = image.reshape(len(image), -1).flatten()

    if meta['endianness'] == 'be':
        raw = raw.byteswap()

    f.seek(meta['offset'])
    raw.tofile(f, sep="")

