from utils.typing_ import ListOrSlice, Tuple, List

from .grid import Grid
from .backend import backend as bd


class LineDetector:
    def __init__(self, name=None):
        self.grid = None
        self.E = []
        self.H = []
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        self.grid = grid
        self.grid.detectors.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ) -> Tuple[List, List, List]:
        if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
            if len(x) != len(y) or len(y) != len(z) or len(z) != len(x):
                raise IndexError(
                    "sources require grid to be indexed with slices or equal length list-indices"
                )
            return x, y, z


        if isinstance(x, list):
            x = slice(x[0], x[-1], None)
        if isinstance(y, list):
            y = slice(y[0], y[-1], None)
        if isinstance(z, list):
            z = slice(z[0], z[-1], None)

        x0 = x.start if x.start is not None else 0
        y0 = y.start if y.start is not None else 0
        z0 = z.start if z.start is not None else 0
        x1 = x.stop if x.stop is not None else self.grid.Nx
        y1 = y.stop if y.stop is not None else self.grid.Ny
        z1 = z.stop if z.stop is not None else self.grid.Nz

        m = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        x = [v.item() for v in bd.array(bd.linspace(x0, x1, m, endpoint=False), bd.int)]
        y = [v.item() for v in bd.array(bd.linspace(y0, y1, m, endpoint=False), bd.int)]
        z = [v.item() for v in bd.array(bd.linspace(z0, z1, m, endpoint=False), bd.int)]
        return x, y, z

    def detect_E(self):
        E = self.grid.E[self.x, self.y, self.z]
        self.E.append(E)

    def detect_H(self):
        H = self.grid.H[self.x, self.y, self.z]
        self.H.append(H)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        z = f"[{self.z[0]}, ... , {self.z[-1]}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s

    def detector_values(self):
        return {"E": self.E, "H": self.H}