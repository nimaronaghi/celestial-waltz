import os
import numpy as np

import struct
import os
from recorder import SimulationRecorder
from PIL import Image


def save_gif(frames, path, delay=10):
    images = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=delay,
        optimize=True,
    )


def main():
    if not os.path.exists("simulation.bin"):
        raise FileNotFoundError("simulation.bin not found")
    rec = SimulationRecorder.load("simulation.bin")
    size = 200
    frames = []
    for pts in rec.frames:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for x, y, z in pts:
            ix = int((x + 2) / 4 * (size - 1))
            iy = int((y + 2) / 4 * (size - 1))
            if 0 <= ix < size and 0 <= iy < size:
                c = int(max(0, min(255, 128 + 127 * z)))
                img[iy, ix] = (0, c, 255 - c)
        frames.append(img)
    save_gif(frames, "simulation.gif")
    Image.fromarray(frames[-1]).save("simulation.png")
                pixels[iy * size + ix] = 1
        frames.append(pixels)
    save_gif(frames, size, size, "simulation.gif")
    img = Image.open("simulation.gif")
    img.save("simulation.png")


if __name__ == "__main__":
    main()
