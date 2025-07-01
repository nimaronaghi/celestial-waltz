import tkinter as tk
from tkinter import ttk
import time
import math

from nbody import BarnesHutSimulation


class GalaxyApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("N-Body Galaxy Simulation")
        self.canvas_size = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        controls = tk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X)

        self.n_var = tk.IntVar(value=200)
        self.dt_var = tk.DoubleVar(value=0.01)
        self.iter_var = tk.IntVar(value=200)
        self.eps_var = tk.DoubleVar(value=0.05)

        tk.Label(controls, text="Particles").pack(side=tk.LEFT)
        tk.Scale(controls, from_=50, to=500, orient=tk.HORIZONTAL, variable=self.n_var).pack(side=tk.LEFT)
        tk.Label(controls, text="Time step").pack(side=tk.LEFT)
        tk.Scale(controls, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.dt_var).pack(side=tk.LEFT)
        tk.Label(controls, text="Iterations").pack(side=tk.LEFT)
        tk.Scale(controls, from_=50, to=1000, orient=tk.HORIZONTAL, variable=self.iter_var).pack(side=tk.LEFT)
        tk.Label(controls, text="Softening").pack(side=tk.LEFT)
        tk.Scale(controls, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.eps_var).pack(side=tk.LEFT)

        # Enable velocity-based coloring by default for better visual feedback
        self.color_var = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Color by velocity", variable=self.color_var).pack(side=tk.LEFT)

        ttk.Button(controls, text="Start", command=self.start).pack(side=tk.LEFT)

        self.sim = None
        self.current_iter = 0

    def start(self):
        n = self.n_var.get()
        dt = self.dt_var.get() / 100.0
        iterations = self.iter_var.get()
        eps = self.eps_var.get() / 100.0
        self.sim = BarnesHutSimulation(num_particles=n, dt=dt, eps=eps)
        self.current_iter = 0
        self.total_iter = iterations
        self.update_simulation()

    def project(self, x, y, z):
        distance = 3.0
        scale = self.canvas_size / 4
        factor = scale / (z + distance)
        cx = self.canvas_size / 2
        cy = self.canvas_size / 2
        px = cx + x * factor
        py = cy - y * factor
        # Clamp coordinates so that the 2x2 particle remains fully visible
        # within the canvas boundaries
        px = max(2, min(self.canvas_size - 2, px))
        py = max(2, min(self.canvas_size - 2, py))
        return px, py

    def draw(self):
        self.canvas.delete("all")
        if not self.sim:
            return
        colorize = self.color_var.get()
        max_speed = 0.0
        if colorize:
            for p in self.sim.particles:
                speed = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz)
                if speed > max_speed:
                    max_speed = speed

        for p in self.sim.particles:
            px, py = self.project(p.x, p.y, p.z)
            color = "white"
            if colorize and max_speed > 0:
                speed = math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz)
                t = min(1.0, speed / max_speed)
                r = int(t * 255)
                b = int((1.0 - t) * 255)
                color = f"#{r:02x}00{b:02x}"
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill=color, outline="")

    def update_simulation(self):
        if self.sim and self.current_iter < self.total_iter:
            self.sim.step()
            self.current_iter += 1
            self.draw()
            self.root.after(10, self.update_simulation)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = GalaxyApp()
    app.run()
