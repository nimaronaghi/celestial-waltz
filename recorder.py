"""Simple recording of simulation frames for playback."""

import struct
from typing import Iterable, List, Tuple


class SimulationRecorder:
    """Record particle positions for each simulation step."""

    def __init__(self, particle_count: int):
        self.particle_count = particle_count
        self.frames: List[List[Tuple[float, float, float]]] = []

    def add_frame(self, particles: Iterable[Tuple[float, float, float]]):
        self.frames.append([(p.x, p.y, p.z) for p in particles])

    def save(self, path: str):
        """Write recorded frames to a binary file."""
        with open(path, "wb") as f:
            header = struct.pack("III", len(self.frames), self.particle_count, 3)
            f.write(header)
            for frame in self.frames:
                for x, y, z in frame:
                    f.write(struct.pack("fff", x, y, z))

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            header = f.read(12)
            frames, count, comps = struct.unpack("III", header)
            recorder = cls(count)
            for _ in range(frames):
                data = []
                for _ in range(count):
                    xyz = struct.unpack("fff", f.read(12))
                    data.append(xyz)
                recorder.frames.append(data)
        return recorder
