from .utils.typing_ import Tensorlike, ListOrSlice

from .backend import backend as bd
from .grid import Grid


class Object:
    def __init__(self, permittivity: Tensorlike, name: str = None):
        self.grid = None
        self.name = name
        self.permittivity = bd.array(permittivity)

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        self.grid = grid
        self.grid.objects.append(self)

        if self.permittivity.dtype is bd.complex().dtype:
            self.grid.promote_dtypes_to_complex()

        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )
        self.x = self._handle_slice(x, max_index=self.grid.Nx)
        self.y = self._handle_slice(y, max_index=self.grid.Ny)
        self.z = self._handle_slice(z, max_index=self.grid.Nz)

        self.Nx = abs(self.x.stop - self.x.start)
        self.Ny = abs(self.y.stop - self.y.start)
        self.Nz = abs(self.z.stop - self.z.start)

        if bd.is_array(self.permittivity) and len(self.permittivity.shape) == 3:
            self.permittivity = self.permittivity[:, :, :, None]
        self.inverse_permittivity = (
            bd.ones((self.Nx, self.Ny, self.Nz, 3),dtype=self.permittivity.dtype) / self.permittivity
        )

        if self.Nx > 1:
            self.inverse_permittivity[-1, :, :, 0] = self.grid.inverse_permittivity[
                -1, self.y, self.z, 0
            ]
        if self.Ny > 1:
            self.inverse_permittivity[:, -1, :, 1] = self.grid.inverse_permittivity[
                self.x, -1, self.z, 1
            ]
        if self.Nz > 1:
            self.inverse_permittivity[:, :, -1, 2] = self.grid.inverse_permittivity[
                self.x, self.y, -1, 2
            ]

        self.grid.inverse_permittivity[self.x, self.y, self.z] = 0

    def _handle_slice(self, s: ListOrSlice, max_index: int = None) -> slice:
        if isinstance(s, list):
            if len(s) == 1:
                return slice(s[0], s[0] + 1, None)
            raise IndexError(
                "One can only use slices or single indices to index the grid for an Object"
            )
        if isinstance(s, slice):
            start, stop, step = s.start, s.stop, s.step
            if step is not None and step != 1:
                raise IndexError(
                    "Can only use slices with unit step to index the grid for an Object"
                )
            if start is None:
                start = 0
            if start < 0:
                start = max_index + start
            if stop is None:
                stop = max_index
            if stop < 0:
                stop = max_index + stop
            return slice(start, stop, None)
        raise ValueError("Invalid grid indexing used for object")

    def update_E(self, curl_H):
        loc = (self.x, self.y, self.z)

        self.grid.E[loc] = self.grid.E[loc] +(
            self.grid.courant_number * self.inverse_permittivity * curl_H[loc]
            )

    def update_H(self, curl_E):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

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