import tkinter as tk
from tkinter import ttk
import math

from nbody import BarnesHutSimulation


class GalaxyApp:
    CAMERA_DISTANCE = 3.0
    NEAR_PLANE = 0.01

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
        # new options for integrator, initial conditions and solver mode
        self.integrator_var = tk.StringVar(value="leapfrog")
        self.initial_var = tk.StringVar(value="spiral")
        self.mode_var = tk.StringVar(value="bh")

        tk.Label(controls, text="Particles / galaxy").pack(side=tk.LEFT)
        tk.Scale(controls, from_=50, to=500, orient=tk.HORIZONTAL, variable=self.n_var).pack(side=tk.LEFT)
        tk.Label(controls, text="Time step (code units)").pack(side=tk.LEFT)
        tk.Scale(controls, from_=0.001, to=0.1, resolution=0.001, orient=tk.HORIZONTAL, variable=self.dt_var).pack(side=tk.LEFT)
        tk.Label(controls, text="Iterations").pack(side=tk.LEFT)
        tk.Scale(controls, from_=50, to=1000, orient=tk.HORIZONTAL, variable=self.iter_var).pack(side=tk.LEFT)
        tk.Label(controls, text="Softening (code units)").pack(side=tk.LEFT)
        tk.Scale(controls, from_=0, to=0.5, resolution=0.005, orient=tk.HORIZONTAL, variable=self.eps_var).pack(side=tk.LEFT)

        # dropdown for integrator selection
        tk.Label(controls, text="Integrator").pack(side=tk.LEFT)
        integrator_menu = ttk.OptionMenu(controls, self.integrator_var, self.integrator_var.get(), "euler", "leapfrog", "rk4")
        integrator_menu.pack(side=tk.LEFT)
        # dropdown for initial condition selection
        tk.Label(controls, text="Initial").pack(side=tk.LEFT)
        initial_menu = ttk.OptionMenu(controls, self.initial_var, self.initial_var.get(), "spiral", "plummer", "kuzmin", "two_galaxies")
        initial_menu.pack(side=tk.LEFT)
        # dropdown for solver mode
        tk.Label(controls, text="Solver").pack(side=tk.LEFT)
        mode_menu = ttk.OptionMenu(controls, self.mode_var, self.mode_var.get(), "bh", "direct")
        mode_menu.pack(side=tk.LEFT)

        # Enable velocity-based coloring by default for better visual feedback
        self.color_var = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Color by velocity", variable=self.color_var).pack(side=tk.LEFT)

        ttk.Button(controls, text="Start", command=self.start).pack(side=tk.LEFT)

        self.sim = None
        self.current_iter = 0

    def start(self):
        n = self.n_var.get()
        dt = self.dt_var.get()
        iterations = self.iter_var.get()
        eps = self.eps_var.get()
        # create simulation with selected options
        self.sim = BarnesHutSimulation(
            num_particles=n,
            dt=dt,
            eps=eps,
            integrator=self.integrator_var.get(),
            initial=self.initial_var.get(),
            mode=self.mode_var.get(),
        )
        self.current_iter = 0
        self.total_iter = iterations
        self.update_simulation()

    def project(self, x, y, z):
        """Project into the viewport, or return None for an invisible point.

        The camera is at z=-CAMERA_DISTANCE and looks along positive z.
        NEAR_PLANE is a rendering distance in code units, not a force cutoff.
        Offscreen particles are culled rather than moved onto the image border.
        """
        if not all(math.isfinite(value) for value in (x, y, z)):
            return None
        depth = z + self.CAMERA_DISTANCE
        if depth <= self.NEAR_PLANE:
            return None
        scale = self.canvas_size / 4
        factor = scale / depth
        cx = self.canvas_size / 2
        cy = self.canvas_size / 2
        px = cx + x * factor
        py = cy - y * factor
        if not (0 <= px <= self.canvas_size and 0 <= py <= self.canvas_size):
            return None
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
            projected = self.project(p.x, p.y, p.z)
            if projected is None:
                continue
            px, py = projected
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
