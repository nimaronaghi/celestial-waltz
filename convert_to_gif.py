import struct
from recorder import SimulationRecorder


def lzw_encode(data, min_code_size=8):
    clear = 1 << min_code_size
    end = clear + 1
    code_size = min_code_size + 1
    codes = [clear] + list(data) + [end]

    out = []
    cur = 0
    bits = 0
    for code in codes:
        cur |= code << bits
        bits += code_size
        while bits >= 8:
            out.append(cur & 0xFF)
            cur >>= 8
            bits -= 8
    if bits:
        out.append(cur)

    blocks = []
    i = 0
    while i < len(out):
        chunk = bytes(out[i:i+255])
        blocks.append(bytes([len(chunk)]) + chunk)
        i += 255
    blocks.append(b"\x00")
    return bytes([min_code_size]) + b"".join(blocks)


def save_gif(frames, width, height, path, delay=10):
    header = b"GIF89a"
    packed = 0xF0  # global color table flag + 8-bit color + 2 colors
    lsd = struct.pack("<HHBBB", width, height, packed, 0, 0)
    gct = bytes([0, 0, 0, 255, 255, 255])

    gif = bytearray(header + lsd + gct)
    for frame in frames:
        # graphics control extension
        gif.extend(b"\x21\xF9\x04\x04" + struct.pack("<H", delay) + b"\x00\x00")
        # image descriptor
        gif.extend(struct.pack("<BHHHHB", 0x2C, 0, 0, width, height, 0))
        gif.extend(lzw_encode(frame))
    gif.append(0x3B)
    with open(path, "wb") as f:
        f.write(gif)


def main():
    rec = SimulationRecorder.load("simulation.bin")
    size = 200
    frames = []
    for pts in rec.frames:
        pixels = [0] * (size * size)
        for x, y, _ in pts:
            ix = int((x + 2) / 4 * (size - 1))
            iy = int((y + 2) / 4 * (size - 1))
            if 0 <= ix < size and 0 <= iy < size:
                pixels[iy * size + ix] = 1
        frames.append(pixels)
    save_gif(frames, size, size, "simulation.gif")


if __name__ == "__main__":
    main()
