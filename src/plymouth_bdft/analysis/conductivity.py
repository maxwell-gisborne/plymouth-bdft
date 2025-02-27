# Warnning: None of the following has been tested,
# Its likey it doesnt even run properly
# How to exactly intergrate it with the above is also not fully worked out yet.

import numpy as np
from plymouth_bdft.immutiblenamespacu import imns


def llc(signal, lbnorm , N, area, axis):
    'calclates f(t) = l*L_s^l {f(t-s)}'
    assert axis in {0,1}, axis
    a = b = c = np.arange(N)
    if axis == 0:
        X = a[:,None,None] - c[None,None,:]
        Y = b[None,:,None]
    else:
        X = a[:,None,None]
        Y = b[None,:,None] - c[None,None,:]

    kernel = - np.exp(-lbnorm * c/N) 
    return - area *1/np.expm1(-lbnorm) * np.expm1(-lbnorm/N) * np.sum(kernel[None,None,:] * signal[X,Y], axis=-1)


def rombus_area(b0,b1):
    # area = |b0||b1|sqrt(1 - ((b0 dot b1)/(|b0||b1|)) ^2 )
    b01norm = np.linalg.norm(b0) * np.linalg.norm(b1)
    b01dot = b0.dot(b1)
    return b01norm * (1-(b01dot/b01norm)**2)**.5

@imns
class Conductivity:
    bi: np.ndarray
    area: float
    N: int
    n: np.ndarray
    sample_points: np.ndarray
    energy: np.ndarray
    eoverhbar=1
    T=1
    mu=1

    def initialise_samplegrid(self, N, bi):
        n = np.arange(N)
        b1,b2 = bi
        return self(
            bi = bi,
            N = N,
            area = rombus_area(b1,b2),
            sample_points = n[:,None, None]/N * b1[None, None,:] + n[None,:,None]/N * b2[None,None,:],             
        )

    def get_sample_points(self):
        return self.sample_points.reshape(-2,2)

    def set_energies(self, band: np.ndarray):
        return self(energy = band.reshape(self.N,self.N))
        

    def get_fermi_density(self):
        return 1/(1 - np.exp(self.beta*(self.energy - self.mu)))

    def get_derivitive(self, axis):
        assert axis in {0,1}, axis
        if axis == 0:
            return (self.energy[1:,:] - self.energy[:-1,:]) / np.linalg.norm((self.sample_points[1:,:] - self.sample_points[:-1,:]), axis=-1)
        else:
            return (self.energy[:,1:] - self.energy[:,:-1]) / np.linalg.norm((self.sample_points[:,1:] - self.sample_points[:,:-1]), axis=-1)

    def get_density(self, l, axis, beta=1, mu=0):
        assert axis in {0,1}, axis
        
        return 2*np.pi * llc(
            signal = self.get_fermi_density(beta, mu),
            lbnorm = l*[ np.linalg.norm(self.bi[0]), np.linalg.norm(self.bi[1]) ][axis],
            area = self.area,
            axis=axis)

    def get_current_densitie(self, l, Jhat, Ehat):
        velocity = self.get_derivitive(Jhat)
        density = self.get_density(l, Ehat)
        return self.eoverhbar * density * velocity

    def get_currents(self, l, Jhat, Ehat):
        return np.sum(self.get_current_densitie(self,l, Jhat, Ehat)) * self.area/self.N
