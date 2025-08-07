import math
import numpy as np
from PIL import Image
from nbody import BarnesHutSimulation

# parameters
particles_per_galaxy = 200  # 200 per galaxy -> 400 total
steps = 300                 # total simulation steps
record_every = 1            # record every step
size = 400                  # image size (pixels)
rotation_speed = 0.4        # rotation speed around z axis

def normalize(value, min_v, max_v):
    return (value - min_v) / (max_v - min_v + 1e-8)

# initialize simulation
sim = BarnesHutSimulation(num_particles=particles_per_galaxy, dt=0.02, mode="bh", integrator="leapfrog", initial="two_galaxies")

frames = []

for step in range(steps):
    sim.step()
    # record
    if step % record_every != 0:
        continue
    # rotate positions for perspective
    angle = rotation_speed * 2 * math.pi * (step / steps)
    ca = math.cos(angle)
    sa = math.sin(angle)
    # compute z-range for colour mapping
    zs = [p.z for p in sim.particles]
    z_min = min(zs)
    z_max = max(zs)
    # prepare image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for p in sim.particles:
        # rotate in XY plane
        x_rot = ca * p.x - sa * p.y
        y_rot = sa * p.x + ca * p.y
        # define bounds for mapping (choose range [-5,5])
        x_norm = (x_rot + 5) / 10.0
        y_norm = (y_rot + 5) / 10.0
        ix = int(x_norm * (size - 1))
        iy = int(y_norm * (size - 1))
        if 0 <= ix < size and 0 <= iy < size:
            depth = normalize(p.z, z_min, z_max)
            # assign color (blue to yellow gradient)
            r = int(255 * depth)
            g = int(255 * (1 - depth))
            b = 255
            # draw pixel (accumulate brightness)
            current = img[iy, ix]
            img[iy, ix] = np.maximum(current, np.array([r,g,b], dtype=np.uint8))
    frames.append(Image.fromarray(img))
    print(f"Recorded frame {len(frames)}/{steps}", end='\r')

# save GIF
output_path = "galaxy_collision_big.gif"
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
print(f"\nGIF saved to {output_path}, frames: {len(frames)}")
