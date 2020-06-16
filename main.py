import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
from collections import defaultdict

class Cell:
    """A class representing a single cell."""
    
    def __init__(self, x, y, vx, vy, cell_id, cell_type, 
                 sticks_to=np.array(()), radius=0.01, styles=None):
        """Initialize the cell's id, group, type, position, velocity, and 
        radius. cell_id and cell_type are non-negative integers. 
        sticks_to is a NumPy array of cell types that stick to the cell. 
        Any key-value pairs passed in the styles dictionary will be passed as 
        arguments to Matplotlib's Circle patch constructor.
        """
        
        self.cell_id = cell_id
        self.group_id = cell_id
        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.cell_type = cell_type
        self.sticks_to = sticks_to
        self.radius = radius
        self.mass = self.radius**2
        
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # max cell type count
        self.color = self.colors[self.cell_type]
        
        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {'edgecolor': self.color, 'fill': False}
    
    # For convenience, map the components of the cell's position and
    # velocity vector onto the attributes x, y, vx and vy.
    
    @property
    def x(self):
        return self.r[0]
    @x.setter
    def x(self, value):
        self.r[0] = value
    @property
    def y(self):
        return self.r[1]
    @y.setter
    def y(self, value):
        self.r[1] = value
    @property
    def vx(self):
        return self.v[0]
    @vx.setter
    def vx(self, value):
        self.v[0] = value
    @property
    def vy(self):
        return self.v[1]
    @vy.setter
    def vy(self, value):
        self.v[1] = value
    
    def overlaps(self, other):
        """Overlapping with another cell?"""
        
        return np.linalg.norm(self.r - other.r) < self.radius + other.radius
    
    def sticks(self, other):
        """Sticking to another cell?"""
        
        return other.cell_type in self.sticks_to
    
    def set_group_id(self, other):
        self.group_id = other.group_id

    ##################################################################
    ### GET BACK TO draw FUNCTION LATER TO SEE HOW add_patch WORKS ###
    ##################################################################
    
    def draw(self, ax):
        """Add this cell's Circle patch to the Matplotlib Axes ax."""
        
        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle
    
    def advance(self, dt):
        """Advance the cell's position forward in time by dt."""
        self.r += self.v * dt

class Simulation:
    """A class for a simple molecular dynamics simulation. The simulation is 
    carried out on a square domain: 0 <= x < 1, 0 <= y < 1.
    """
    
    CellClass = Cell
    
    def __init__(self, n, cell_ids, ntypes, radius, sticks_to, styles=None):
        """Initialize the simulation with n cells with radii radius.
        radius can be a single value or a sequence with n values. ntypes is
        the number of types. sticks_to is a dictionary of lists. Any key-value 
        pairs passed in the styles dictionary will be passed as arguments to 
        Matplotlib's Circle patch constructor when drawing the cells.
        """
        
        self.init_cells(n, cell_ids, ntypes, radius, sticks_to, styles)
        self.dt = 0.01
        
        # groups = defaultdict(list)
        # self.groups = groups
        
    def place_cell(self, idy, typ, sti, rad, styles):
        # Choose x, y so that the cell is entirely inside the
        # domain of the simulation.
        x, y = rad + (1 - 2*rad) * np.random.random(2)
        vr = 0.05
        vphi = 2*np.pi * np.random.random()
        vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
        cell = self.CellClass(x, y, vx, vy, idy, typ, sti, rad, styles)
        for c2 in self.cells:
            if c2.overlaps(cell):
                break
        else:
            self.cells.append(cell)
            return True
        return False

    def init_cells(self, n, cell_ids, ntypes, radius, sticks_to, styles=None):
        """Initialize the n cells of the simulation.
        Positions and velocities are chosen randomly; radius can be a single
        value or a sequence with n values.
        """

        try:
            # iterator = iter(radius)
            assert n == len(radius) # Assert other stuff too
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)

        self.n = n
        self.cell_ids = cell_ids
        self.ntypes = ntypes
        all_types = [np.random.choice(np.arange(ntypes)) for i in range(n)]
        new_sticks_to = defaultdict(list)
        self.cells = []
        for i, rad in enumerate(radius):
            # Try to find a random initial position for this particle.
            new_sticks_to.update({i: sticks_to[all_types[i]]})
            while not self.place_cell(cell_ids[i],
                                      all_types[i],
                                      new_sticks_to[i], radius[i], styles):
                pass
        self.sticks_to = new_sticks_to
        self.all_types = all_types
    
    def adjust_velocities(self, c1, c2):
        """Change the cell's velocity."""
        
        c1_group = c1.group_id
        c2_group = c2.group_id
        if c1.cell_type in c2.sticks_to: # Should be SYMMETRIC!!!!!!!
            for c in self.cells:
                if c.group_id == c2_group:
                    c.group_id = c1_group

            for c in self.cells:
                if c.group_id == c1_group:
                    c.v = (c1.v + c2.v) / 2 # Don't know why went this way.

        else:
            for c in self.cells:
                if c.group_id in (c1_group, c2_group):
                    c.v = -c.v

    def handle_collisions(self):
        """Detect and handle any collisions between the cells.
        When two non-adhering cells collide, they do so elastically: their 
        velocities change such that both energy and momentum are conserved.
        When two adhering cells collide, they stick.
        """ 
    
        pairs = combinations(range(self.n), 2)
        for i, j in pairs:
            if self.cells[i].overlaps(self.cells[j]):
                self.adjust_velocities(self.cells[i], self.cells[j])

    def handle_boundary_collisions(self, c):
        """Bounce the cells off the walls elastically."""
        
        if c.x - c.radius < 0:
            c.x = c.radius
            c.vx = -c.vx
        if c.x + c.radius > 1:
            c.x = 1-c.radius
            c.vx = -c.vx
        if c.y - c.radius < 0:
            c.y = c.radius
            c.vy = -c.vy
        if c.y + c.radius > 1:
            c.y = 1-c.radius
            c.vy = -c.vy
            
    def advance_animation(self):
        """Advance the animation by dt, returning the updated Circles list."""

        for i, c in enumerate(self.cells):
            c.advance(self.dt)
            self.handle_boundary_collisions(c)
            self.circles[i].center = c.r
        self.handle_collisions()
        return self.circles
    
    def advance(self):
        """Advance the animation by dt."""
        
        for i, c in enumerate(self.particles):
            c.advance(self.dt)
            self.handle_boundary_collisions(c)
        self.handle_collisions()
        
    def init(self):
        """Initialize the Matplotlib animation."""

        self.circles = []
        for cell in self.cells:
            self.circles.append(cell.draw(self.ax))
        return self.circles
    
    def animate(self, i):
        """The function passed to Matplotlib's FuncAnimation routine."""

        self.advance_animation()
        return self.circles
            
    def setup_animation(self):
        self.fig, self.ax = plt.subplots()
        for s in ['top','bottom','left','right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

    def save_or_show_animation(self, anim, save, filename='cells.mp4'):
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, bitrate=1800)
            anim.save(filename, writer=writer)
        else:
            plt.show()

    def do_animation(self, save=False, interval=1, filename='cells.mp4'):
        """Set up and carry out the animation of the molecular dynamics.
        To save the animation as a MP4 movie, set save=True.
        """

        self.setup_animation()
        anim = animation.FuncAnimation(self.fig, self.animate,
                init_func=self.init, frames=800, interval=interval, blit=True)
        self.save_or_show_animation(anim, save, filename)

if __name__ == '__main__':
    Ncells = 300
    IDs = np.arange(Ncells)
    Ntypes = 3
    Radii = np.ones(Ncells)*0.01

    Sticks_to = defaultdict(list)
    Sticks_to[0] = [1, 2]
    Sticks_to[1] = [0, 2]
    Sticks_to[2] = [0, 1]
    
    styles = {'edgecolor': 'C0', 'linewidth': 2, 'fill': False}
    sim = Simulation(Ncells, IDs, Ntypes, Radii, Sticks_to) # DON'T PASS IN styles.
    sim.do_animation(save=True)