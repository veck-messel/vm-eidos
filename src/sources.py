from math import pi, sin

from .utils.typing_ import Number, ListOrSlice, Tuple, List
from .utils.waveforms import hanning

from .grid import Grid
from .backend import backend as bd

class PointSource:
    def __init__(
        self,
        period: Number = 15,
        amplitude: float = 1.0,
        phase_shift: float = 0.0,
        name: str = None,
        pulse: bool = False,
        cycle: int = 5,
        hanning_dt: float = 10.0,
    ):
        self.grid = None
        self.period = period
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.name = name
        self.pulse = pulse
        self.cycle = cycle
        self.frequency = 1.0 / period
        self.hanning_dt = hanning_dt if hanning_dt is not None else 0.5 / self.frequency

    def _register_grid(self, grid: Grid, x: Number, y: Number, z: Number):
        self.grid = grid
        self.grid.sources.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        try:
            (x,), (y,), (z,) = x, y, z
        except (TypeError, ValueError):
            raise ValueError("a point source should be placed on a single grid cell.")
        self.x, self.y, self.z = grid._handle_tupple((x, y, z))
        self.period = grid._handle_time(self.period)

    def update_E(self):
        q = self.grid.time_steps_passed
        if self.pulse:
            t1 = int(2 * pi / (self.frequency * self.hanning_dt / self.cycle))
            if q < t1:
                src = self.amplitude * hanning(
                    self.frequency, q * self.hanning_dt, self.cycle
                )
            else:
                src = 0
        else:
            src = self.amplitude * sin(2 * pi * q / self.period + self.phase_shift)
        self.grid.E[self.x, self.y, self.z, 2] += src

    def update_H(self):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(period={self.period}, "
            f"amplitude={self.amplitude}, phase_shift={self.phase_shift}, "
            f"name={repr(self.name)})"
        )

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"{self.x}"
        y = f"{self.y}"
        z = f"{self.z}"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s

class LineSource:
    def __init__(
        self,
        period: Number = 15,
        amplitude: float = 1.0,
        phase_shift: float = 0.0,
        name: str = None,
        pulse: bool = False,
        cycle: int = 5,
        hanning_dt: float = 10.0,
    ):
        
        self.grid = None
        self.period = period
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.name = name
        self.pulse = pulse
        self.cycle = cycle
        self.frequency = 1.0 / period
        self.hanning_dt = hanning_dt if hanning_dt is not None else 0.5 / self.frequency

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        self.grid = grid
        self.grid.sources.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

        self.period = grid._handle_time(self.period)

        L = len(self.x)
        vect = bd.array(
            (bd.array(self.x) - self.x[L // 2]) ** 2
            + (bd.array(self.y) - self.y[L // 2]) ** 2
            + (bd.array(self.z) - self.z[L // 2]) ** 2,
            bd.float,
        )

        self.profile = bd.exp(-(vect ** 2) / (2 * (0.5 * vect.max()) ** 2))
        self.profile /= self.profile.sum()
        self.profile *= self.amplitude

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ) -> Tuple[List, List, List]:
        
        if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
            if len(x) != len(y) or len(y) != len(z) or len(z) != len(x):
                raise IndexError(
                    "sources require grid to be indexed with slices or equal length list-indices"
                )
            x = [self.grid._handle_distance(_x) for _x in x]
            y = [self.grid._handle_distance(_y) for _y in y]
            z = [self.grid._handle_distance(_z) for _z in z]
            return x, y, z

        if isinstance(x, list):
            x = slice(
                self.grid._handle_distance(x[0]),
                self.grid._handle_distance(x[-1]),
                None,
            )
        if isinstance(y, list):
            y = slice(
                self.grid._handle_distance(y[0]),
                self.grid._handle_distance(y[-1]),
                None,
            )
        if isinstance(z, list):
            z = slice(
                self.grid._handle_distance(z[0]),
                self.grid._handle_distance(z[-1]),
                None,
            )

        x0 = self.grid._handle_distance(x.start if x.start is not None else 0)
        y0 = self.grid._handle_distance(y.start if y.start is not None else 0)
        z0 = self.grid._handle_distance(z.start if z.start is not None else 0)
        x1 = self.grid._handle_distance(x.stop if x.stop is not None else self.grid.Nx)
        y1 = self.grid._handle_distance(y.stop if y.stop is not None else self.grid.Ny)
        z1 = self.grid._handle_distance(z.stop if z.stop is not None else self.grid.Nz)

        m = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        if m < 2:
            raise ValueError("a LineSource should consist of at least two gridpoints")
        x = [v.item() for v in bd.array(bd.linspace(x0, x1, m, endpoint=False), bd.int)]
        y = [v.item() for v in bd.array(bd.linspace(y0, y1, m, endpoint=False), bd.int)]
        z = [v.item() for v in bd.array(bd.linspace(z0, z1, m, endpoint=False), bd.int)]

        return x, y, z

    def update_E(self):
        q = self.grid.time_steps_passed
        if self.pulse:
            t1 = int(2 * pi / (self.frequency * self.hanning_dt / self.cycle))
            if q < t1:
                vect = self.profile * hanning(
                    self.frequency, q * self.hanning_dt, self.cycle
                )
            else:
                vect = self.profile * 0
        else:
            vect = self.profile * sin(2 * pi * q / self.period + self.phase_shift)
        for x, y, z, value in zip(self.x, self.y, self.z, vect):
            self.grid.E[x, y, z, 2] += value

    def update_H(self):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(period={self.period}, "
            f"amplitude={self.amplitude}, phase_shift={self.phase_shift}, "
            f"name={repr(self.name)})"
        )

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        z = f"[{self.z[0]}, ... , {self.z[-1]}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s
