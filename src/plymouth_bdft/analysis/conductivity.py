import numpy as np
from numpy import linalg, ndarray, array, arange
from plymouth_bdft.immutiblenamespace import imns
from dataclasses import dataclass

def rombus_area(b0,b1):
    # area = |b0||b1|sqrt(1 - ((b0 dot b1)/(|b0||b1|)) ^2 )
    b01norm = linalg.norm(b0) * linalg.norm(b1)
    b01dot = b0.dot(b1)
    return b01norm * (1-(b01dot/b01norm)**2)**.5


@imns
class KSpace:

    # k-space has dimentions 1/L
    b1: ndarray = None
    b2: ndarray = None

    # aria covered (dimensions 1/L^2)
    A: float = None

    # Exstent in the x and y
    N1: int = None
    N2: int = None

    _sample_points_cash: [bool, array] = lambda: [False, None]

    def Primitive_Grid(b1, b2, N):
        return KSpace(b1=b1, b2=b2,
                      N1=N, N2=N,
                      A =  rombus_area(b1,b2))

    def partial_1(kspace, F):
        out = np.zeros_like(F)
        out[1:, :]   = (F[1:, :] - F[:-1, :]) * kspace.N1 / linalg.norm(kspace.b1)
        out[:-1, :] += out[1:, :] # central difference
        out[1:-1, :] /= 2
        return out

    def partial_2(kspace, F):
        out = np.zeros_like(F)
        out[:, 1:]   = (F[:, 1:] - F[:, :-1]) * kspace.N2 / linalg.norm(kspace.b2)
        out[:, :-1] += out[:, 1:] # central difference
        out[:, 1:-1] /= 2
        return out

    def sample_points(kspace):
        cash = kspace._sample_points_cash
        if cash[0] is False:
            n1 = arange(kspace.N1)
            n2 = arange(kspace.N2)
            cash[1] = kspace.b1[None,None,:] * n1[:,None,None]/kspace.N1 +  kspace.b2[None,None,:] * n2[None,:,None]/kspace.N2
            cash[0] = True
        return cash[1]
                     

    def integrate(self, integrand: ndarray) -> float:
        return self.A * np.sum(integrand) / (self.N1 * self.N2 )

    def grid_to_array(kspace, grid):
        array = grid.reshape(kspace.N1*kspace.N2, -1)
        if array.shape[-1] == 1:
            return array[:,0]
        return array

    def array_to_grid(kspace,array):
        grid = array.reshape(kspace.N1, kspace.N2, -1)
        if grid.shape[-1] == 1:
            return grid[:,:,0]
        return grid
    

    def plot_sample_points(kspace, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()

        sample_points = kspace.sample_points()
        ax.scatter(*sample_points.transpose([2,0,1]), s=.2, label='points')

        ax.set_aspect(1)
        ax.set_title('K-points')
        ax.legend()

        return ax

    def plot_over_sample_grid(kspace, F, ax=None, **plot_args):
        plot_args['cmap'] = plot_args.get('cmap', 'inferno')
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()

        sample_points = kspace.sample_points()

        # ax.scatter(*mesh.transpose([2, 0, 1]), c=F, **plot_args)
        pcm = ax.pcolormesh(*sample_points.transpose([2, 0, 1]), F, **plot_args)
        ax.figure.colorbar(pcm, ax=ax)
        ax.set_aspect(1)
        return ax


def assert_norm(vec):
    assert (vec @ vec - 1)**2 < 1e-20, ('exspected normalised vector', vec @ vec)

def wrap_inplace(n:np.ndarray, M:int):
    n[n>=0] = n[n>=0] % M
    n[n<0] = (M - np.abs(n[n<0]) % M) % M



def eq_1_8_9_varable_angle(nu, E, Ep, bs, M=None, l=1, S_g=8):
    'tested notebooks/scripts/independent_integral_test.py with pijul hash 4DV3ENCQH3ZV5WMK2KUV47LW7ZHUQJFGLEF5P4FLVFLVYLPYTQ4QC'
    N = nu.shape[0]
    M = M or 2*N
    assert nu.shape == (N,N), nu.shape
    m = np.arange(M)
    _1 = (slice(M),None, None)
    _2 = (None, slice(M),None)
    _3 = (None, None, slice(M))

    def wrap(n, NF):
        n[n>=0] = n[n>=0] % NF
        n[n<0] = (NF - np.abs(n[n<0]) % NF ) % NF
        return n

    a1,a2 = np.linalg.inv(bs).transpose() # a1 dot b1 = 1, b1 dot b2 = 0
    aE = [a1@E, a2@E]
    aEp = [a1@Ep, a2@Ep] 

    b1,b2 = bs
    bE = [b1@E, b2@E]
    bEp = [b1@Ep, b2@Ep]


    lower_so = abs(min(*bEp))
    upper_so = abs(max(*bEp))

    L = np.linalg.norm(b1)
    Lp = lower_so + upper_so
    shift = lower_so/Lp

    zero_segment = np.zeros((M,M))
    for s in range(S_g):
        print(f'{s}/{S_g}', end='\r')
        n1, n2 = (N/M * aE[i] * L * ((m[_1]-m[_3]) - s * M) +  N/M * aEp[i] * Lp * (m[_2] - shift * M) for i in (0,1))
        n1 = wrap(np.round(n1).astype('int'), N)
        n2 = wrap(np.round(n2).astype('int'), N)

        zero_segment += np.sum(nu[n1,n2] * np.exp(-l*L*(m[_3]/M + s)), axis=-1)

    zero_segment *= 2 * np.pi * (-np.expm1(-l*L/M))


    _1 = (slice(M), None)
    _2 = (None, slice(M))

    n1, n2 = (N/M * aE[i] * L * m[_1] +  N/M * aEp[i] * Lp * (m[_2] - shift * M) for i in (0,1))
    n1 = wrap(np.round(n1).astype('int'), N)
    n2 = wrap(np.round(n2).astype('int'), N)
    out = np.zeros((N,N))
    out[n1,n2] = zero_segment[m[_1],m[_2]]

    # return Eq_Output(
    #     Eq = "1.8.9 variable angle",
    #     perams=dict(M=M, N=N, l=l, L=L, Lp=Lp, shift=shift, S_g=S_g),
    #     out = out,
    #     )
     
    return out


@dataclass
class Classical:
    M: int
    kspace: KSpace
    energy: np.ndarray
    rho: np.ndarray = None
    eoverhbar: float = 1.0
    beta: float = 1.0
    mu: float = 1.0
   
    def calculate_rho(self, l, M=None, S_g = 10, theta = None):
        b1 = self.kspace.b1
        b2 = self.kspace.b2

        if theta is None:
            theta = np.log(np.complex128(*b1)).imag

        i,j = np.eye(2)
        E  =  np.cos(theta)*i + np.sin(theta)*j
        Ep = -np.sin(theta)*i + np.cos(theta)*j
        rho = eq_1_8_9_varable_angle(
            nu = 1/(1+np.exp(- self.beta * (self.energy - self.mu))),
            E = E, 
            Ep = Ep,
            bs = np.array([b1,b2]),
            M = M,
            l = l,
            S_g = S_g,
        )
        return rho

    def calculate_j(self, l, M = None, S_g = 10, theta = None):
        b1 = self.kspace.b1
        b2 = self.kspace.b2
        BZ_area =  np.sqrt((np.linalg.norm(b1)*np.linalg.norm(b2))**2 - (b1 @ b2)**2)

        ep_p1 = self.kspace.partial_1(self.energy)
        ep_p2 = self.kspace.partial_2(self.energy)

        rho = self.calculate_rho(l=l, M=M, S_g=S_g, theta=theta)

        j1 = - self.eoverhbar * BZ_area * np.average(ep_p1 * rho)
        j2 = - self.eoverhbar * BZ_area * np.average(ep_p2 * rho)

        jx, jy = b1 * j1 + b2 * j2
        return jx, jy

    def calculate_js(self, Etaus: np.ndarray, M = None, S_g = 10, theta=None):
        #Etaus = np.concat([np.linspace(0, (5e-2)-(1e-5))[1:], np.linspace(5e-2, .5)])
        j1s = []
        j2s = []
        for i,Etau in enumerate(Etaus):
            print(f'{i}/{len(Etaus)}'.rjust(10), end='\r')
            l = 1 / Etau / self.eoverhbar
            j1, j2 = self.calculate_j(l = l, M=M, S_g=S_g, theta=theta)
            j1s.append(j1)
            j2s.append(j2)

        j1s = np.array(j1s)
        j2s = np.array(j2s)

        b1 = self.kspace.b1
        b2 = self.kspace.b2
        jxs, jys = b1[:,None] * j1s[None,:] + b2[:,None] * j2s[None,:]

        return jxs, jys 

    def calculate_sigma(self, Etaus: np.ndarray, js=None, M = None, S_g = 10, theta=None):
        if js is None:
            jxs, jys = self.calculate_js(M=M, S_g = S_g, theta=theta)
        else:
            jxs, jys = js

        b1 = self.kspace.b1
        if theta is None:
            theta = np.log(np.complex128(*b1)).imag


        # should conpensate for the angle at which E is at
        
        sigma_x = (jxs[1:] -jxs[:-1])/(Etaus[1:] - Etaus[:-1])
        sigma_y = (jys[1:] -jys[:-1])/(Etaus[1:] - Etaus[:-1])
        return sigma_x, sigma_y 
        
    def __repr_(self):
        calculated_rho = 'calculated rho' if self.rho is not None else 'not calculated rho'
        return f'Classical_Conductivity({calculated_rho}{hash(self)})'

    def plot_energy(self, ax=None):
        if ax is None:
            import matplotlib as plt
            _, ax = plt.subplots()
        self.kspace.plot_over_sample_grid(self.energy)
