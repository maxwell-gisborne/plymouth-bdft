import pytest

from pathlib import Path

from plymouth_bdft.analysis.conductivity import KSpace, Classical
from plymouth_bdft.analysis import BigDFT_Data, AU
import plymouth_bdft.condlib as condlib
import numpy as np
from matplotlib import pyplot as plt

def savefig(fig, name: str):
    figure_path = Path('figures')
    if not figure_path.exists():
        figure_path.mkdir()
    fig.savefig(figure_path.joinpath(name).with_suffix('.svg'))


plt.style.use('normal.mplstyle')


#   ,=====================================================.
#  /                  initialize bdft data                 \
# |            ---------------------------                  |

N = 32

@pytest.fixture
def data():
    data:BigDFT_Data = (BigDFT_Data(data_dir = Path('test_run/run_dir/data'))
        (units = AU)
        .resolve_paths()
        .load_meta()
        .load_matrixes()
        .project_to_surface()
        .calculate_geometry()
        .calculate_sublattice()
        .calculate_compound_channels()
        .print('into kspace')
        .calculate_Reduced_Hamiltonian()
        .into_kspace(N)
        .display()
    )

    k_points = data.kspace.sample_points()

    global ai
    global bi
    global norm_b
    global epsilon

    ai = data.geometry.ai
    bi = data.geometry.bi
    norm_b = np.linalg.norm(bi[0])
    epsilon = data.index_by_cc.Bands[0].reshape(N,N)

    return data

# |                                                         |
#  \                                                       /
#   `=====================================================‘



#   ,=====================================================.
#  /               calling into ny C wrappers              \
# |               -----------------------------             |
# |         ,-----------------------------------.           |
# |        /       exstracting constants         \          |



Ehat = np.array([1.0,0.0])
etau_hbar = 1

# |         \                                    /         |
# |          `----------------------------------‘          |
# |                                                        |
# |          ,-----------------------------------.         |
# |         /           testing nabla             \        |
# |         |           seems to work             |        |

@pytest.fixture
def nabla_epsilon():
    return condlib.nabla(epsilon,ai,norm_b)
    

def test_epsilon(data, nabla_epsilon):
    nabla_epsilon_old = data.kspace.partial_1(epsilon)[None,:,:] * data.geometry.ai[0][:,None,None] + data.kspace.partial_2(epsilon)[None,:,:] * data.geometry.ai[1][:,None,None]
    assert np.all( np.abs(nabla_epsilon_old - nabla_epsilon) < 1e-9 )



# |         \                                    /         |
# |          `----------------------------------‘          |
# |                                                        |
# |          ,-----------------------------------.         |
# |         /           testing nu                \        |
# |         |                                     |        |
# |         |   nu_ab is the constants of the     |        |
# |         |   harmonic series of nu             |        |
# |         |                                     |        |


@pytest.fixture
def nuab():
    return condlib.nu_ab(epsilon,ai)

def test_reconstructed_nuab(nuab, data):
    nu_k_re = np.real(condlib.nuk_reconstruct(nuab, ai))

    nu_k = 1/(1+np.exp(-epsilon))
    diff = nu_k_re / nu_k
    assert np.all(np.abs(1-diff) < 1e-14), np.average(np.abs(diff))

    if False:
        fig, axs = plt.subplots(2,2)

        plt.colorbar(axs[1][0].imshow(np.abs(nuab)))
        axs[1][0].set_title('nu_ab')

        plt.colorbar(axs[0][0].imshow(nu_k_re))
        axs[0][0].set_title('reconstruction')

        plt.colorbar(axs[0][1].imshow(nu_k))
        axs[0][1].set_title('direct')

        plt.colorbar(axs[1][1].imshow(1-diff))
        # axs[1][1].set_title('reconstructed/direct')

        fig.tight_layout()
        fig.show()




# |         |                                    |         |
# |         \                                    /         |
# |          `----------------------------------‘          |
# |                                                        |
# |          ,-----------------------------------.         |
# |         /          testing Gamma              \        |
# |         |                                     |        |
# |         |       Not sure how                  |        |
# |         | check the inverse transform         |        |
def test_gamma():
    assert False, 'idk how to do this yet'
    gamma_10 = condlib.calculate_gamma(epsilon, ai, 1.0)
# |         |                                     |        |
# |         \                                     /        |
# |          `-----------------------------------‘         |
# |                                                        |
# |          ,-----------------------------------.         |
# |         /   testing full sigma calculation    \        |
# |         |                                     |        |
# |         |   I needa way to validate this      |        |
# |         | To do this I can evaluate at a      |        |
# |         | single point which I can do manualy |        |
# |         |                                     |        |
# |         |   Can check the expansion           |        |



def test_Esweep():
    E = 1
    global out
    out = condlib.sigma(epsilon, E, Ehat, ai, etau_hbar)

    sigmas = []
    Es = np.linspace(0.01,1000)
    for E in Es:
        sigmas.append(np.sum(condlib.sigma(epsilon, E, Ehat, ai, etau_hbar)[0,0]))
    
    sigmas = np.array(sigmas)
    [print(E,', ', sigma,) for E, sigma in zip(Es, sigmas)]
    print()
    fig, ax = plt.subplots()
    ax.plot(Es, sigmas)
    savefig(fig, 'E_Sweep')



# |         \                                    /          |
# |          `----------------------------------‘           |
# |                                                         |
#  \                                                       /
#   `=====================================================‘


