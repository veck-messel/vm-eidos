from tqdm import tqdm

from .utils import constants as const
from .utils.typing_ import Tuple, Number, Tensorlike

from .backend import backend as bd


def curl_E(E: Tensorlike) -> Tensorlike:
    curl = bd.zeros(E.shape, dtype=E.dtype)

    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl

def curl_H(H: Tensorlike) -> Tensorlike:
    curl = bd.zeros(H.shape, dtype=H.dtype)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl

class Grid:
    from .visualization import visualize

    def __init__(
            self,
            shape: Tuple[Number, Number, Number],
            grid_spacing: float = 155e-9,
            permittivity: float = 1.0,
            permeability: float = 1.0,
            courant_number: float = None):
        
        self.grid_spacing = float(grid_spacing)

        self.Nx, self.Ny, self.Nz = self._handle_tupple(shape)

        self.D = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)

        max_courant_number = float(self.D) ** (-0.5)
        if courant_number is None:
            self.courant_number = 0.99 * max_courant_number
        elif courant_number > max_courant_number:
            raise ValueError(
                f"courant_number {courant_number} too high for "
                f"a {self.D}D simulation"
            )
        else:
            self.courant_number = float(courant_number)

        self.time_step = self.courant_number * self.grid_spacing / const.c

        self.E = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.H = bd.zeros((self.Nx, self.Ny, self.Nz, 3))

        if bd.is_array(permittivity) and len(permittivity.shape) == 3:
            permittivity = permittivity[:, :, :, None]
        self.inverse_permittivity = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / bd.array(
            permittivity, dtype=bd.float
        )

        if bd.is_array(permeability) and len(permeability.shape) == 3:
            permeability = permeability[:, :, :, None]
        self.inverse_permeability = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / bd.array(
            permeability, dtype=bd.float
        )

        self.time_steps_passed = 0

        self.sources = []

        self.boundaries = []

        self.detectors = []

        self.objects = []

        self.folder = None
    
    def _handle_distance(self, distance: Number) -> int:
        if not isinstance(distance, int):
            return int(float(distance) / self.grid_spacing + 0.5)
        return distance

    def _handle_time(self, time: Number) -> int:
        if not isinstance(time, int):
            return int(float(time) / self.time_step + 0.5)
        return time

    def _handle_tupple(
            self,
            shape: Tuple[Number, Number, Number]) -> Tuple[int, int, int]:
        if len(shape) != 3:
            raise ValueError(
                f"invalid grid shape {shape}\n"
                f"grid shape should be a 3D tuple containing floats or ints"
        )
        x, y, z = shape
        x = self._handle_distance(x)
        y = self._handle_distance(y)
        z = self._handle_distance(z)
        return x, y, z

    def _handle_slice(self, s: slice) -> slice:
        start = (s.start
                 if not isinstance(s.start, float)
                 else self._handle_distance(s.start))
        stop = (
            s.stop if not isinstance(s.stop, float) else self._handle_distance(s.stop)
        )
        step = (
            s.step if not isinstance(s.step, float) else self._handle_distance(s.step)
        )
        return slice(start, stop, step)
    
    def _handle_single_key(self, key):
        try:
            len(key)
            return [self._handle_distance(k) for k in key]
        except TypeError:
            if isinstance(key, slice):
                return self._handle_slice(key)
            else:
                return [self._handle_distance(key)]

    @property
    def x(self) -> int:
        return self.Nx * self.grid_spacing

    @property
    def y(self) -> int:
        return self.Ny * self.grid_spacing

    @property
    def z(self) -> int:
        return self.Nz * self.grid_spacing

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)

    @property
    def time_passed(self) -> float:
        return self.time_steps_passed * self.time_step

    def run(self, total_time: Number, progress_bar: bool = True):
        if isinstance(total_time, float):
            total_time /= self.time_step
        time = range(0, int(total_time), 1)
        if progress_bar:
            time = tqdm(time)
        for _ in time:
            self.step()

    def step(self):
        self.update_E()
        self.update_H()
        self.time_steps_passed += 1

    def update_E(self,):
        for boundary in self.boundaries:
            boundary.update_phi_E()

        curl = curl_H(self.H)
        self.E += self.courant_number * self.inverse_permittivity * curl

        for obj in self.objects:
            obj.update_E(curl)

        for boundary in self.boundaries:
            boundary.update_E()

        for src in self.sources:
            src.update_E()

        for det in self.detectors:
            det.detect_E()

    def update_H(self, ):
        for boundary in self.boundaries:
            boundary.update_phi_H()

        curl = curl_E(self.E)
        self.H -= self.courant_number * self.inverse_permeability * curl

        for obj in self.objects:
            obj.update_H(curl)

        for boundary in self.boundaries:
            boundary.update_H()

        for src in self.sources:
            src.update_H()

        for det in self.detectors:
            det.detect_H()

    def reset(self):
        self.H *= 0.0
        self.E *= 0.0
        self.time_steps_passed *= 0

    def add_source(self, name, source):
        source._register_grid(self)
        self.sources[name] = source

    def add_boundary(self, name, boundary):
        boundary._register_grid(self)
        self.boundaries[name] = boundary

    def add_object(self, name, obj):
        obj._register_grid(self)
        self.objects[name] = obj

    def add_detector(self, name, detector):
        detector._register_grid(self)
        self.detectors[name] = detector

    def promote_dtypes_to_complex(self):
        self.E = self.E.astype(bd.complex)
        self.H = self.H.astype(bd.complex)
        [boundary.promote_dtypes_to_complex() for boundary in self.boundaries]

    def __setitem__(self, key, attr):
        if not isinstance(key, tuple):
            x, y, z = key, slice(None), slice(None)
        elif len(key) == 1:
            x, y, z = key[0], slice(None), slice(None)
        elif len(key) == 2:
            x, y, z = key[0], key[1], slice(None)
        elif len(key) == 3:
            x, y, z = key
        else:
            raise KeyError("maximum number of indices for the grid is 3")

        attr._register_grid(
            grid=self,
            x=self._handle_single_key(x),
            y=self._handle_single_key(y),
            z=self._handle_single_key(z),
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape=({self.Nx},{self.Ny},{self.Nz}), "
            f"grid_spacing={self.grid_spacing:.2e}, courant_number={self.courant_number:.2f})"
        )

    def __str__(self):
        s = repr(self) + "\n"
        if self.sources:
            s = s + "\nsources:\n"
            for src in self.sources:
                s += str(src)
        if self.detectors:
            s = s + "\ndetectors:\n"
            for det in self.detectors:
                s += str(det)
        if self.boundaries:
            s = s + "\nboundaries:\n"
            for bnd in self.boundaries:
                s += str(bnd)
        if self.objects:
            s = s + "\nobjects:\n"
            for obj in self.objects:
                s += str(obj)
        return s