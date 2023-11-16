from utils.typing_ import ListOrSlice, IntOrSlice

from .grid import Grid
from .backend import backend as bd


# Base class
class Boundary:
    def __init__(self, name: str = None):
        self.grid = None
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        self.grid = grid
        self.grid.boundaries.append(self)
        self.x = self._handle_slice(x)
        self.y = self._handle_slice(y)
        self.z = self._handle_slice(z)

        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

    def _handle_slice(self, s: ListOrSlice) -> IntOrSlice:
        if isinstance(s, list):
            if len(s) > 1:
                raise ValueError(
                    "Use slices or single numbers to index the grid for a boundary"
                )
            return s[0]
        if isinstance(s, slice):
            if (
                s.start is not None
                and s.stop is not None
                and (s.start == s.stop or abs(s.start - s.stop) == 1)
            ):
                return s.start
            return s
        raise ValueError("Invalid grid indexing used for boundary")

    def update_phi_E(self):
        pass

    def update_phi_H(self):
        pass

    def update_E(self):
        pass

    def update_H(self):
        pass
    
    def promote_dtypes_to_complex(self):
        self.phi_E = bd.complex(self.phi_E)
        self.phi_H = bd.complex(self.phi_H)

        self.psi_Ex = bd.complex(self.psi_Ex)
        self.psi_Ey = bd.complex(self.psi_Ey)
        self.psi_Ez = bd.complex(self.psi_Ez)
        
        self.psi_Hx = bd.complex(self.psi_Hx)
        self.psi_Hy = bd.complex(self.psi_Hy)
        self.psi_Hz = bd.complex(self.psi_Hz)

    def __repr__(self):
        return f"PML(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"

        def _handle_slice(s):
            return (
                str(s)
                .replace("slice(", "")
                .replace(")", "")
                .replace(", ", ":")
                .replace("None", "")
            )

        x = _handle_slice(self.x)
        y = _handle_slice(self.y)
        z = _handle_slice(self.z)
        s += f"        @ x={x}, y={y}, z={z}".replace(":,", ",")
        if s[-1] == ":":
            s = s[:-1]
        return s + "\n"

class PeriodicBoundary(Boundary):
    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        super()._register_grid(grid=grid, x=x, y=y, z=z)

        if self.x == 0 or self.x == -1:
            self.__class__ = _PeriodicBoundaryX
            if hasattr(grid, "_xlow_boundary") or hasattr(grid, "_xhigh_boundary"):
                raise AttributeError("grid already has an xlow/xhigh boundary!")
            setattr(grid, "_xlow_boundary", self)
            setattr(grid, "_xhigh_boundary", self)
        elif self.y == 0 or self.y == -1:
            self.__class__ = _PeriodicBoundaryY
            if hasattr(grid, "_ylow_boundary") or hasattr(grid, "_yhigh_boundary"):
                raise AttributeError("grid already has an ylow/yhigh boundary!")
            setattr(grid, "_ylow_boundary", self)
            setattr(grid, "_yhigh_boundary", self)
        elif self.z == 0 or self.z == -1:
            self.__class__ = _PeriodicBoundaryZ
            if hasattr(grid, "_zlow_boundary") or hasattr(grid, "_zhigh_boundary"):
                raise AttributeError("grid already has an zlow/zhigh boundary!")
            setattr(grid, "_zlow_boundary", self)
            setattr(grid, "_zhigh_boundary", self)
        else:
            raise IndexError(
                "A periodic boundary should be placed at the boundary of the "
                "grid using a single index (either 0 or -1)"
            )

class _PeriodicBoundaryX(PeriodicBoundary):
    def update_E(self):
        self.grid.E[0, :, :, :] = self.grid.E[-1, :, :, :]

    def update_H(self):
        self.grid.H[-1, :, :, :] = self.grid.H[0, :, :, :]

class _PeriodicBoundaryY(PeriodicBoundary):
    def update_E(self):
        self.grid.E[:, 0, :, :] = self.grid.E[:, -1, :, :]

    def update_H(self):
        self.grid.H[:, -1, :, :] = self.grid.H[:, 0, :, :]


class _PeriodicBoundaryZ(PeriodicBoundary):
    def update_E(self):
        self.grid.E[:, :, 0, :] = self.grid.E[:, :, -1, :]

    def update_H(self):
        self.grid.H[:, :, -1, :] = self.grid.H[:, :, 0, :]
