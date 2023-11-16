from math import pi, sin

from .utils.typing_ import Number
from .utils.waveforms import *

from .grid import Grid


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
