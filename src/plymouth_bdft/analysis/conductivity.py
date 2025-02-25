# Warnning: None of the following has been tested,
# Its likey it doesnt even run properly
# How to exactly intergrate it with the above is also not fully worked out yet.

import numpy as np
from plymouth_bdft.immutiblenamespace import imns
# from plotting_utils import plt, plot_points

infinity = object()
im = 0+1j

rho_f = lambda beta, mu: lambda E:  (
        E < mu
        if beta is infinity else (
            1/(1 + np.exp(-(E - mu) * beta))
        )
)


@imns
class KSpace:
    spacial_resolution: float = None

    kx: np.ndarray = None
    ky: np.ndarray = None

    wx: np.ndarray = None
    wy: np.ndarray = None

    DK: float = None
    Nx: int = None
    Ny: int = None

    mask: np.ndarray = None
    padding: (int, int) = (0, 0)

    def partial_kx(kspace, F):
        out = np.zeros_like(F)
        out[1:, :]   = (F[1:, :] - F[:-1, :])/kspace.spacial_resolution[0]
        out[:-1, :] += out[1:, :]
        out[1:-1, :] /= 2
        return out

    def partial_ky(kspace, F):
        out = np.zeros_like(F)
        out[:,  1:]  = (F[:, 1:] - F[:, :-1])/kspace.spacial_resolution[1]
        out[:, :-1] += out[:, 1:]
        out[:, 1:-1] /= 2
        return out

    def calculate_dule_space(self):
        return self(
                wx = 2*np.pi*np.fft.fftfreq(self.kx.shape[0], 1/self.spacial_resolution[0]),
                wy = 2*np.pi*np.fft.fftfreq(self.ky.shape[0], 1/self.spacial_resolution[1]),
        )

    def calculate_LD(self):
        Lx = self.kx[-1] - self.kx[0]
        Ly = self.ky[-1] - self.ky[0]
        dx = self.kx[1] - self.kx[0]
        dy = self.ky[1] - self.ky[0]
        Nx = self.kx.shape[0]
        Ny = self.ky.shape[0]
        return self(  # Lx = Lx, Ly = Ly,
                    Nx = Nx, Ny = Ny,
                    DK = Lx*Ly / (Nx*Ny)**.5,
                    spacial_resolution=[dx, dy])

    def init_with_square_tile(self, delta: float = .5, tileing_number: int = 2,):
        ky = np.arange(0, tileing_number * np.pi*4/3**.5, delta)
        kx = np.arange(0, tileing_number * np.pi*4/3*3,   delta)
        return self(kx = kx, ky = ky).calculate_LD().calculate_dule_space()

    def init_with_hex(self, dule_vectors: np.ndarray, N: int = 100, padding: (int, int) = (0, 0)):
        '''Given the primitive lattice vectors `dule_vectors` that define a triangular lattice,
        who's Wigner-Sitz cell is there for a hexagon, a space k-space will be initialised, along with a mask
        that selects points inside that Wigner-Sites hexagon.'''

        # width of the non-padded region is 2*B,
        # density is thus 2B/N
        # thus for M exstra points, with the same density, a padding width of 2B/N * M is requied
        B = 2/3*np.max(dule_vectors)
        Mx, My = padding
        margin_x = 2*B*Mx/N
        margin_y = 2*B*My/N
        xlinspace = np.linspace(-(B+margin_x), B, (N+Mx))
        ylinspace = np.linspace(-(B+margin_y), B, (N+My))

        test_points = np.array(np.meshgrid(xlinspace, ylinspace)).transpose()

        d1, d2 = dule_vectors
        control_points = [d1, d1 + d2, d2]
        mask = np.ones(test_points.shape[:-1], dtype=bool)

        for cp in control_points:
            mask *= np.abs(np.einsum('abi,i->ab', test_points, cp)) < np.dot(cp, cp)/2
            # | test_point dot control_point | < 1/2
            # control points are the nerist naobur cites of the triangular lattice
            # d1

        return self(
                kx = xlinspace,
                ky = ylinspace,
                mask = mask,
                padding = padding,
        ).calculate_LD().calculate_dule_space()

    def integrate(self, integrand: np.ndarray) -> float:
        if self.mask is not None:
            integrand = integrand[self.mask]
        return self.DK * np.sum(integrand)

    def convolution_laplace(self, F: np.ndarray, l: float, direction=0, tol=None):
        tol = tol or 1e-9
        assert direction == 0,  'not implemented y kdirecion'
        assert F.shape == (self.kx.shape[0], self.ky.shape[0]), (F.shape, (self.kx.shape[0], self.ky.shape[0]))
        N = self.padding[direction]
        M = self.kx.shape[direction] - N
        assert M > 0
        deltak = self.spacial_resolution[direction]
        assert (_tol := np.exp(-l*deltak*N)) < tol, _tol

        # F3[a,b,c] = F[a-c,b]
        # F3 = np.lib.stride_tricks.as_strided(F, shape=(*F.shape, N), strides=(*F.strides, -F.strides[0]), writeable=False)
        a = np.arange(M+N)
        b = np.arange(M)
        c = np.arange(N)
        result = 2*np.pi*l*np.sum(
                (
                    np.exp(-l * c[None, None, :] * deltak)
                    * F[
                        a[:, None, None] - c[None, None, :],
                        b[None, :, None]
                    ]
                ),
                axis=-1)*deltak
        result[:N, :] = 0  # setting the padding to zero
        assert result.shape == (M+N, M), result.shape
        return result

    def k_mesh(self) -> np.ndarray:
        kx, ky = self.kx, self.ky
        return np.array(np.meshgrid(kx, ky)).transpose([2, 1, 0])

    def padding_mask(self) -> np.ndarray:
        Nx = self.kx.shape[0]
        Ny = self.ky.shape[0]
        Px, Py = self.padding
        padding_mask = np.ones((Nx, Ny), bool)
        padding_mask[np.arange(Nx) < Px] = False
        padding_mask[:, np.arange(Ny) < Py] = False
        assert np.sum(padding_mask) == (Nx-Px)*(Ny-Py)

        def apply(mesh: np.ndarray):
            if len(mesh.shape) == 2:
                return mesh[padding_mask].reshape(Nx-Px, Ny-Py)
            if len(mesh.shape) == 3:
                return mesh[padding_mask].reshape(Nx-Px, Ny-Py, mesh.shape[2])
            if len(mesh.shape) == 1:
                assert mesh.shape[0] == Nx * Ny, mesh.shape[0]
                return self.reshape_points_to_mesh(mesh)[padding_mask]
            raise Exception('Not Implemented', mesh.shape)

        return apply

    def reshape_mesh_to_points(self, mesh: np.ndarray) -> np.ndarray:
        return mesh.reshape(self.Nx*self.Ny, -1)

    def reshape_points_to_mesh(self, points: np.ndarray) -> np.ndarray:
        return points.reshape(self.Nx, self.Ny, -1)

    def _check_reshapes(self):
        mesh = self.k_mesh()
        assert np.all(mesh == self.reshape_points_to_mesh(self.reshape_mesh_to_points(mesh)))
        return self

    def _check_dimentions_of_kmesh(self):
        assert self.k_mesh().shape == (self.Nx, self.Ny, 2)
        return self

    def plot_k_points(self, ignore_padding=True, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()

        mesh = self.k_mesh()
        all_points = mesh
        bz1 = mesh[self.mask]
        if ignore_padding:
            apply_padding_mask = self.padding_mask()
            all_points = apply_padding_mask(all_points)
            bz1 = all_points[apply_padding_mask(self.mask)]

        ax.scatter(*all_points.transpose(), s=.2, label='points outside mask')
        ax.scatter(*bz1.transpose(), s=.3, label='points inside mask')

        ax.set_aspect(1)
        ax.set_title('K-points')
        ax.legend()

        return ax

    def plot_over_k(self, F, ax=None, ignore_padding=True, **plot_args):
        plot_args['cmap'] = plot_args.get('cmap', 'inferno')
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()

        mesh = self.k_mesh()
        if ignore_padding:
            apply_mask = self.padding_mask()
            mesh = apply_mask(mesh)
            F = apply_mask(F)

        # ax.scatter(*mesh.transpose([2, 0, 1]), c=F, **plot_args)
        pcm = ax.pcolormesh(*mesh.transpose([2, 0, 1]), F, **plot_args)
        ax.figure.colorbar(pcm, ax=ax)
        return ax


@imns
class Units:
    tau: float
    hbar: float
    e: float
    me: float
    four_pi_epsilon: float
    bolzman: float

    seconds: float
    meters: float
    jules: float
    kg: float
    volts: float
    amps: float

    au_time: float
    au_current: float
    hatrees: float
    bhor: float

    def beta(self, T):
        return infinity if T == 0 else 1/(self.bolzman * T)


SI = Units(
    tau = 17e-12,  # https://arxiv.org/pdf/1712.08965
    hbar = 6.626e-34 / (2*np.pi),
    e = 1.602e-19,
    me = 9.109e-31,
    four_pi_epsilon = 1.113e-10,
    bolzman = 1.3e-23,

    seconds = 1,
    meters = 1,
    jules = 1,
    kg = 1,
    volts = 1,
    amps = 1,


    au_time = None,
    au_current = None,
    hatrees = None,
    bhor = None,
)

SI = SI(bhor = SI.four_pi_epsilon * SI.hbar**2 / (SI.me * SI.e**2))
SI = SI(hatrees = SI.hbar**2 / (SI.me * SI.bhor**2))
SI = SI(au_time = SI.hbar / SI.hatrees)
SI = SI(au_current = SI.e * SI.hatrees / SI.hbar)


AU = Units(
    tau = SI.tau / SI.au_time,
    hbar = 1,
    e = 1,
    me = 1,
    four_pi_epsilon = 1,
    bolzman = SI.bolzman / SI.hatrees,

    seconds = 1/SI.au_time,
    meters = 1/SI.bhor,
    jules = 1/SI.hatrees,
    kg = 1/SI.me,
    volts = SI.hatrees / SI.e,
    amps = 1/SI.au_current,

    au_time = 1,
    hatrees = 1,
    bhor = 1,
    au_current = 1,
)


@imns
class Classical_Conductivity:
    Energy: np.ndarray = None
    units: Units = AU

    kspace: KSpace = None
    EField: np.ndarray = lambda: np.array([1, 0])
    kernel_factor: np.ndarray = None
    sigma: np.ndarray = None

    Temp: float = 293  # 19.8 C in kelvin
    mu: float = 0  # fermi-level

    def init_kspace(self, **kwargs):
        return self(
                kspace = (KSpace(**kwargs)
                          .calculate_LD()
                          .calculate_dule_space()
                          )
        )

    def plot_fermi_distribution(self, **plot_args):
        kT = self.Temp*self.units.bolzman
        return self.kspace.plot_over_k(rho_f(1/kT, self.mu)(self.Energy), **plot_args)

    def calculate_kernel_factor(self):
        ''' Z(wx, wy) = 1/(1+1im*(τ*e/hbar)*(wx*Efield_x + wy*Efield_y))
            kernel_factor(wx, wy) = 1im*(e^2 * τ)/(hbar^2) .* (Z.(wx, wy)).^2 '''

        Zinv = 1 + im * (self.units.tau * self.units.e / self.units.hbar) * (
                (self.kspace.wx * self.EField[0])[:, np.newaxis] +
                (self.kspace.wy * self.EField[1])[np.newaxis, :]
        )  # indicies read [x][y]

        kf = im * self.units.e**2 * self.units.tau * self.units.hbar**-2 * Zinv**-2

        return self(kernel_factor = kf)

    def calculate_rho(self, E=1, tol=None):
        l = self.units.hbar / self.units.e / self.units.tau / E
        return self.kspace.convolution_laplace(
                self.kspace.reshape_points_to_mesh(rho_f(self.units.beta(self.Temp), self.mu)(self.Energy))[:, :, 0],
                l=l,
                tol=tol)

    def calculate_current(self, E=1, tol=None):
        return - self.units.e * self.kspace.partial_kx(self.Energy) * self.calculate_rho(E=E, tol=tol)

    def calculate_conductivity(self, l=1):
        return self(
                sigma = 0
        )
