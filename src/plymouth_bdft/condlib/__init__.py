from plymouth_bdft.analysis.conductivity import KSpace, Classical
from plymouth_bdft.analysis import BigDFT_Data, AU
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from numpy.ctypeslib import ndpointer
import ctypes

module_file = Path(__file__).parent.resolve()


class Complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]
    def __complex__(self):
        return complex(self.real, self.imag)

plt.style.use(module_file.joinpath('../normal.mplstyle').resolve())


#    ,===================================================.
#   /              ,---------------------.                \
#  /              (   Loading my C code   )                \
# |                `---------------------’                  |
  
condlib = ctypes.cdll.LoadLibrary(module_file.joinpath('lib.so'))

condlib.version.restype = ctypes.c_char_p
def version():
    return condlib.version().decode('utf-8')



# |          ,-----------------------------------.          |
# |        /      main calculation function      \          |

condlib.calculate_sigma.restype = None
condlib.calculate_sigma.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # sigma_tau
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # epsilon
    ctypes.c_double, # E
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # ai
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # Ehat
    ctypes.c_size_t, # N
    ctypes.c_double, # e tau / hbar
]

def sigma(epsilon, E, Ehat, ai, etau_hbar:float = 1.0):
    assert len(epsilon.shape) == 2, epsilon.shape
    assert epsilon.shape[0] == epsilon.shape[1], epsilon.shape
    assert ai.shape == (2,2), ai.shape
    N = epsilon.shape[0]
    output = np.zeros((2,2,N,N))
    condlib.calculate_sigma(
                          output, epsilon,
                          E,
                          ai, Ehat,
                          N, etau_hbar)
    return output 

# |         \                                    /          |
# |          `----------------------------------‘           |
# |                                                         |
# |          ,-----------------------------------.          |
# |         /      nabla calculation function     \         |

condlib.calculate_nabla.restype = None
condlib.calculate_nabla.argtypes = [
    ndpointer(np.complex128, flags="C_CONTIGUOUS"), # nabla out 
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # image in
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # ai
    ctypes.c_size_t, # N
]

def nabla(image:np.ndarray, basis:np.ndarray):
    '''calculates the gradient of the image.
    The image is sampled on points interpolated from the vectors of ai.
    '''
    print('executinge nabla calc')
    assert len(image.shape) == 2, image.shape
    assert image.shape[0] == image.shape[1], image.shape
    N = image.shape[0]
    assert basis.shape == (2,2), basis.shape
    output = np.zeros((2,N,N), dtype=np.complex128)
    condlib.calculate_nabla(
        output, image,
        np.linalg.inv(basis), N
    )
    return output
   


# |         \                                    /          |
# |          `----------------------------------‘           |
# |                                                         |
# |          ,-----------------------------------.          |
# |         /      nu_k reconstruction            \         |

condlib.nuk_reconstruction.restype = None
condlib.nuk_reconstruction.argtypes = [
    ndpointer(np.complex128, flags="C_CONTIGUOUS"), # nu_k[i,j] (output)
    ndpointer(np.complex128, flags="C_CONTIGUOUS"), # nu_ab
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # ai
    ctypes.c_size_t, # N
]

def nuk_reconstruct(nu_ab:np.ndarray, ai:np.ndarray):
    print('reconstructing nuk from nu_ab')
    assert len(nu_ab.shape) == 2, nu_ab.shape
    assert nu_ab.shape[1] == nu_ab.shape[0], nu_ab.shape
    N = nu_ab.shape[0]
    assert ai.shape == (2,2), ai.shape
    nuk = np.zeros((N,N), np.complex128)
    condlib.nuk_reconstruction(
        nuk, nu_ab,
        ai, N
    )
    return np.real(nuk)
   

# |         \                                    /         |
# |          `----------------------------------‘          |
# |                                                        |
# |          ,-----------------------------------.         |
# |          /      calculation of DFT of nu      \

condlib.calculate_nu.restype = None
condlib.calculate_nu.argtypes = [
    ndpointer(np.complex128, flags="C_CONTIGUOUS"), # nu_ab
    ctypes.c_double, # beta
    ctypes.c_double, # mu
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # epsilon
    ctypes.c_size_t, # N
]

def nu_ab(epsilon, ai, beta=1, mu=0):
    print('executing nu_ab calc')
    assert len(epsilon.shape) == 2, epsilon.shape
    assert epsilon.shape[0] == epsilon.shape[1], epsilon.shape
    N = epsilon.shape[0]

    #            ,-> sign of b 
    #  output  = 2 × N × N
    #                 \   `-> b
    #                  `-> a   

    output = np.zeros((N,N), dtype=np.complex128)
    condlib.calculate_nu(
        output, beta,
        mu, epsilon, N
    )
    return output

condlib.calculate_nu_slow.restype = None
condlib.calculate_nu_slow.argtypes = [
    ndpointer(np.complex128, flags="C_CONTIGUOUS"), # nu_ab
    ctypes.c_double, # beta
    ctypes.c_double, # mu
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # epsilon
    ctypes.c_size_t, # N
]

def nu_ab_slow(epsilon, ai, beta=1, mu=0):
    print('executing nu_ab calc slow')
    assert len(epsilon.shape) == 2, epsilon.shape
    assert epsilon.shape[0] == epsilon.shape[1], epsilon.shape
    N = epsilon.shape[0]

    #            ,-> sign of b 
    #  output  = 2 × N × N
    #                 \   `-> b
    #                  `-> a   

    output = np.zeros((N,N), dtype=np.complex128)
    condlib.calculate_nu_slow(
        output, beta,
        mu, epsilon, N
    )
    return output

# | 
# |         \                                    /          |
# |          `----------------------------------‘           |
# |                                                         |
# |          ,-----------------------------------.          |
# |         /      calculation of Gamma           \         |

condlib.calculate_gamma.restype = None
condlib.calculate_gamma.argtypes = [
    ndpointer(np.complex128, flags="C_CONTIGUOUS"), # gamma out
    ndpointer(ctypes.c_double,flags="C_CONTIGUOUS"), # epsilon
    ndpointer(ctypes.c_double,flags="C_CONTIGUOUS"), # Ehat_dot_a
    ctypes.c_size_t, ctypes.c_double, # N, A
]

def calculate_gamma(epsilon, Ehat, ai):
    print('calculating gamma')
    assert ai.shape == (2,2)
    N = epsilon.shape[0]
    assert epsilon.shape == (N,N)
    gamma = np.zeros((2, N, N), dtype=np.complex128)
    assert Ehat.shape == (2,)
    assert Ehat @ Ehat == 1
    Ehat_p = np.array([-Ehat[1], Ehat[0]])
    Ehat = np.array([Ehat, Ehat_p]).T @ np.linalg.inv(ai)

    Ehat_dot_a = np.array([ [Ehat[0] @ ai[0], Ehat[0] @ ai[1]],
                            [Ehat[1] @ ai[0], Ehat[1] @ ai[1]] ])
    A = (2*np.pi/3)**2 * 2 * 3**.5
    condlib.calculate_gamma(
        gamma, epsilon, Ehat_dot_a,
        N, A
    )
    return gamma

# | 
# |         \                                    /          |
# |          `----------------------------------‘           |
# |                                                         |
# |          ,-----------------------------------.          |
# |         /      sigma_bar used for testing     \         |
# | 

condlib.F.restype = Complex
condlib.F.argtypes = [
    ctypes.c_int, # a
    ctypes.c_int, # b
    ctypes.c_int, # beta
    ctypes.c_double, # E
    ndpointer(ctypes.c_double,flags="C_CONTIGUOUS", shape=(2,2)), # Ehat
    ndpointer(ctypes.c_double,flags="C_CONTIGUOUS", shape=(2,2)), # ai
    ctypes.c_double, # etau_hbar
]

def F(a,b, beta, E, Ehat, ai, etau_hbar):
    Ehat_beta = np.array([Ehat, [Ehat[1], -Ehat[0]]],dtype=np.float64)
    return complex(condlib.F(a, b, beta, E, Ehat_beta, ai, etau_hbar))


# |         \                                    /          |
# |          `----------------------------------‘           |
# |                                                         |
#  \                                                       /
#   `=====================================================‘
   
