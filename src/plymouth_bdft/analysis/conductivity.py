# Warnning: None of the following has been tested,
# Its likey it doesnt even run properly
# How to exactly intergrate it with the above is also not fully worked out yet.

import numpy as np
from numpy import linalg, ndarray, array, arange
from plymouth_bdft.immutiblenamespace import imns

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


@imns
class llc_config:
    l: float = None # charicteristic length 
    Ehat: ndarray = None # float[2], direction of applied electric field 
    Ehat_perp: ndarray = None # float[2] perpendicular direction
    L: float = None # segmentation length
    N: int = None # primitive grid number
    M: int = None # segment grid number
    Sg: int = None # segments number
    nu: ndarray = None # input density
    input_kspace: KSpace = None #  input mesh, bi
    output: ndarray = None # computed output llc

    def pullback(self, signal, domain, target, points,  shift=array([0,0])):
        NF = signal.shape[0]
        J = linalg.inv(domain.transpose()) @ target.transpose()
        mapped = np.einsum('ij,abcj->iabc', J, points) - shift[:,None,None,None]
    
        upper = np.ceil(mapped).astype('int')
        upper[upper >= NF] = upper[upper >= NF] % NF
        upper[upper < 0] = (NF - np.abs(upper[upper < 0]) % NF) % NF
    
        lower = np.floor(mapped).astype('int')
        lower[lower >= NF] = lower[lower >= NF] % NF
        lower[lower < 0] = (NF - np.abs(lower[lower < 0]) % NF) % NF

        alpha, beta = mapped % 1
    
        return ( alpha     * beta     * signal[upper[0], upper[1]]
               + (1-alpha) * beta     * signal[lower[0], upper[1]]
               + alpha     * (1-beta) * signal[upper[0], lower[1]]
               + (1-alpha) * (1-beta) * signal[lower[0], lower[1]] )

    def compute(self):
        M = self.M
        L = self.L

        segment_offsets = [
                self.Ehat_perp @ self.input_kspace.b1,
                self.Ehat_perp @ self.input_kspace.b2
        ]

        lower_so = abs(min(segment_offsets))
        upper_so = abs(max(segment_offsets))
        sum_so = lower_so + upper_so
        segment_domain = array([L*self.Ehat, sum_so*self.Ehat_perp])
        primitive_domain =  array([self.input_kspace.b1,self.input_kspace.b2])

        m = arange(M)
        m1 = m[:,None,None]
        m2 = m[None,:,None]
        m3 = m[None,None,:]

        zero_segment = np.zeros((M,M))
        x,y = np.eye(2)

        for segment_number in range(self.Sg):
            expt = np.exp(L*(m1/M -m3/M - segment_number))
            zero_segment[:,:] += np.sum(self.pullback(
                signal = self.nu,
                domain = primitive_domain,
                target = segment_domain,
                points = x*(m1-m3-segment_number*M)[:,:,:,None] + y*(m2 - lower_so/sum_so*M)[:,:,:,None],
            )*expt, axis=2)


        output = np.zeros((self.N, self.N))
        n = arange(self.N)
        n1 = n[:,None,None,None]
        n2 = n[None,:,None,None]
        output = self.pullback(
            signal = zero_segment,
            domain = segment_domain,
            target = primitive_domain,
            points = x*n1 + y*n2,
        )[:,:,0]
        return self(
            output = output * 2*np.pi*(-np.expm1(-self.l*L/M))
        )

    def plot(self,ax=None):
        return self.input_kspace.plot_over_sample_grid(self.output)

               
def llc(kspace, signal, l, N, area, axis):
    'calclates f(t) = l*L_s^l {f(t-s)}'
    'I need to change this function to be a propper sum (i waisted my time on it so far and just lost another week)'
    a = b = c = arange(N)
    if axis == 0:
        X = a[:,None,None] - c[None,None,:]
        Y = b[None,:,None]
    else:
        X = a[:,None,None]
        Y = b[None,:,None] - c[None,None,:]

    #kprime = n[:,None]/N * axis[None,:]
    #kernel = np.exp(-l*kprime)
    kernel = np.exp(-l*np.einsum('i,pi->p',axis,kspace))
    # kprime -> A,B index offsets
    # then sample X = a - A, Y = b - B with appropreate index offsets
    out = np.zeros_like(signal)

#    return - area * * np.expm1(-lbnorm/N) * np.sum(kernel[None,None,:] * signal[X,Y], axis=-1)
    return out


class Classical:
    _id = [0]
   
    def __init__(self, kspace:KSpace, energy:ndarray, M, eoverhbar = 1, beta = 1, mu = 1 ):
        self._id[0] += 1
        self.id = self._id[0]

        self.kspace = kspace
        self.energy = energy
        self.eoverhbar = 1
        self.beta = 1
        self.mu = 1

        self.velocity = [lambda: self.kspace.partial_1(self.energy), lambda: self.kspace.partial_2(self.energy)]

        self.charge_density = [(lambda Ehat:
                        lambda l: 2*np.pi * llc(
                            signal = 1/(1+np.exp(-self.beta*(self.energy - self.mu))),
                            lbnorm = l*[linalg.norm(self.kspace.b1), linalg.norm(self.kspace.b2)][Ehat],
                            area = self.kspace.A,
                            N = [self.kspace.N1, self.kspace.N2][Ehat],
                            axis=Ehat))(Ehat) for Ehat in [0,1]]

        self.current_densities = [(lambda Jhat: [(lambda Ehat:
                        lambda l: - self.eoverhbar * self.charge_density[Ehat](l) * self.velocity[Jhat]()
                            )(Ehat) for Ehat in [0,1]]
                            )(Jhat) for Jhat in [0,1]]

        self.currents = [(lambda Jhat: [(lambda Ehat:
                        lambda l: self.kspace.integrate(self.current_densities[Jhat][Ehat](l))
                            )(Ehat) for Ehat in [0,1]]
                            )(Jhat) for Jhat in [0,1]]

        def __repr_(self):
            return f'Classical_Conductivity(id=${self.id})'

    def plot_energy(self, ax=None):
        if ax is None:
            import matplotlib as plt
            _, ax = plt.subplots()
        self.kspace.plot_over_sample_grid(self.energy)
