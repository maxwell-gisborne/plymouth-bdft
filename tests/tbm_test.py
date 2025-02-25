import pytest

from pathlib import Path

fast_mode = False


def savefig(fig, name: str):
    figure_path = Path('figures')
    if not figure_path.exists():
        figure_path.mkdir()
    fig.savefig(figure_path.joinpath(name).with_suffix('.svg'))


@pytest.fixture
def run_path():
    return Path('/home/maxwell/Projects/visualising/safe/tests/test_run/run_dir')


def test_imports(run_path):
    from plymouth_bdft.analysis import BigDFT_Data
    BigDFT_Data(data_dir = run_path.joinpath('data'))


@pytest.fixture
def new_data(run_path):
    from plymouth_bdft.analysis import BigDFT_Data
    return BigDFT_Data(data_dir = run_path.joinpath('data'))


def test_loading(new_data):
    data = (new_data
            .resolve_paths()
            .load_meta()
            .load_matrixes())
    _ = data  # I should probably some how check the data.


@pytest.fixture
def loaded_data(new_data):
    return (new_data
            .resolve_paths()
            .load_meta()
            .load_matrixes())


def test_geometry_1(loaded_data):
    if fast_mode: return 'Disabled'
    (loaded_data.project_to_surface()
     .calculate_geometry()  # defaults the center point to average atomic possition
     .calculate_sublattice()
     )  # I also want to plot the geometric fit and other peramiters
    # should also assert that the a0 is correct


@pytest.fixture
def pre_cc_data(loaded_data):
    return (loaded_data.project_to_surface()
            .calculate_geometry()  # defaults the center point to average atomic possition
            .calculate_sublattice()
            )


def test_cc_and_reduced_ham(pre_cc_data):
    if fast_mode: return 'Disabled'
    data = (pre_cc_data
            .calculate_compound_channels()
            .calculate_Reduced_Hamiltonian())
    _ = data  # I want to plot the reduced hamiltonian


@pytest.fixture
def pre_FT_data(pre_cc_data):
    return (pre_cc_data
            .calculate_compound_channels()
            .calculate_Reduced_Hamiltonian())


def test_geometry_2(pre_FT_data):
    if fast_mode: return 'Disabled'
    calc = pre_FT_data.verify_geometry()
    a0  = calc.geometry.lattice_spaceing()
    savefig(pre_FT_data.plot_atoms(add_geometry=True).figure,
            'Geometry Plot')
    assert a0 - 1.42 < 1e-9  # angstromes


def test_band_calculation(pre_FT_data):
    if fast_mode: return 'Disabled'
    import numpy as np
    hsp = {'G1': [0., 0., 0.],
           'M': [0.5, 0., 0.5],
           'K': [0.666666, 0., 0.333333],
           'G2': [0., 0., 0.]}

    high_symitry_points = np.array(list(hsp.values()))[:, [0, 2]]

    k_points, _ = pre_FT_data.geometry.dule_space_interpolate(high_symitry_points, delta=.01)
    data = (pre_FT_data
            .calculate_FT(k_points)
            .calculate_bands()
            )
    _ = data  # I should probably plot the band data.
    # also having a test were I compare the plot to sams data on the same run would be good.


@pytest.fixture
def hsp_data(pre_FT_data):
    import numpy as np
    hsp = {'G1': [0., 0., 0.],
           'M': [0.5, 0., 0.5],
           'K': [0.666666, 0., 0.333333],
           'G2': [0., 0., 0.]}

    high_symitry_points = np.array(list(hsp.values()))[:, [0, 2]]

    k_points, _ = pre_FT_data.geometry.dule_space_interpolate(high_symitry_points, delta=.01)
    data = (pre_FT_data
            .calculate_FT(k_points)
            .calculate_bands()
            )

    return data


def test_sam(run_path, hsp_data):
    if fast_mode: return 'Disabled'
    from BigDFT import Systems, Fragments, Logfiles, TB
    from BigDFT.PostProcessing import BigDFTool
    from BigDFT.Spillage import MatrixMetadata
    import numpy as np

    import matplotlib.pyplot as plt

    sq3 = np.sqrt(3)
    a0 = 2.5  # lattice parameter

    coord = {'prim': np.array([[0, 0, 0],
                               [a0/sq3, 0, 0]]),
             'conv': np.array([[0, 0, 0],
                               [a0/sq3, 0, 0],
                               [a0*sq3/2, 0, a0/2],
                               [a0*5/sq3/2, 0, a0/2]])}

    cell = {'prim': np.array([[a0*sq3/2, 0, a0/2],
                              [0, float("inf"), 0],
                              [a0*sq3/2, 0, -a0/2]]),
            'conv': np.array([[a0*sq3, 0, 0],
                              [0, float("inf"), 0],
                              [0, 0, a0]])}

    positions = [{'C': list(j)} for j in coord['prim']]
    posinp = {'positions': positions, 'units': 'angstroem'}

    frag = Fragments.Fragment(posinp=posinp)
    sys_cs = Systems.System()
    sys_cs["FRA:1"] = frag

    sys_cs.cell.cell = cell['prim']

    d = 5  # 1nn or 3nn are advised for graphene
    tb = TB.TightBinding(sys_cs, d=d)

    log = Logfiles.Logfile(run_path.joinpath('log.yaml'))
    tool = BigDFTool()
    # no unit conversion, so i belive the energy untis are in hartrees (AU)
    h = tool.get_matrix_h(log)
    s = tool.get_matrix_s(log)

    # metadatafile = f'data-{name}/sparsematrix_metadata.dat'
    metadatafile = run_path.joinpath('data/sparsematrix_metadata.dat')
    metadata = MatrixMetadata(metadatafile)

    posinp = log.astruct  # ['posinp']
    sys_ls = Systems.system_from_dict_positions(posinp['positions'])

    hsp = {'G1': [0., 0., 0.],
           'M': [0.5, 0., 0.5],
           'K': [0.666666, 0., 0.333333],
           'G2': [0., 0., 0.]}

    k = tb.k_path(hsp)

    m_sh = tb.shell_matrix(sys_ls, [h, s], metadata)
    Hk, Sk, Ek = tb.k_matrix(k, m_sh)
    k = tb.k_path(hsp)

    # Ef = log.fermi_level

    def std_dementions(ax):
        ax.set_ylabel('Energy (eV)', size=10)
        ax.set_ylim([-14, 14])
        ax.set_xlim([0, 4])
        return ax

    AU_eV = 27.21138386
    #  AU_to_A = 0.52917721092

    dk = np.linalg.norm(k[1:]-k[:-1], axis=1)
    kpath = np.concatenate(([0], np.cumsum(dk)))

    

    fig, axs = plt.subplots(2)
    std_dementions(axs[0]).plot(kpath, Ek*AU_eV, ms=.7)
    std_dementions(hsp_data.plot_bands(ax = axs[1]))
    axs[0].set_title('Sam\'s Bands')
    axs[1].set_title('Maxwell\'s Bands')
    fig.tight_layout()
    savefig(fig, 'band_structure_comparison_sam')


def test_dos(pre_FT_data):
    if fast_mode: return 'Disabled'
    'This is just to test that the dos works, that I can calculate it'
    savefig(pre_FT_data.init_kspace(40).calculate_dos(alpha=.4).dos.plot().figure, '40 atoms DOS')
    assert True, 'manually 2024-10-23'


def test_conductivity(pre_FT_data):
    if fast_mode: return 'Disabled'
    'test if the conductivity is being calculated'
    pre_FT_data.init_kspace(40).calculate_conductivity()


def test_conductivity_regularisation(pre_FT_data):
    '''As the regularisaton peramiter delta goes to zero,
    the conductivity should approch a non-zero value for the diagonal elements'''

    if fast_mode: return 'Disabled'
    import numpy as np
    sigmas = []
    Ns = [10, 30, 50, 70, 90]
    for N in Ns:
        sigmas.append(np.array(pre_FT_data.init_kspace(N).calculate_conductivity().conds))

    import matplotlib.pyplot as plt

    sigmas = np.array(sigmas)
    for band in range(8):
        fig, axs = plt.subplots(2, 2)
        axs[0][0].scatter(Ns, np.array(sigmas[:, band, 0, 0]))
        axs[0][1].scatter(Ns, np.array(sigmas[:, band, 0, 1]))
        axs[1][0].scatter(Ns, np.array(sigmas[:, band, 1, 0]))
        axs[1][1].scatter(Ns, np.array(sigmas[:, band, 1, 1]))

        fig.suptitle(f'Band {band}')
        savefig(fig, name=f'regularisation conductivity band-{band}')

    assert False, 'Manualy Reviewd on 2024-10-23'


def test_Kspace(pre_FT_data):
    if fast_mode: return 'Disabled'
    'test if the Kspace is bing clculated'
    savefig(pre_FT_data
            .init_kspace(50)
            .verify_geometry()
            .kspace
            ._check_dimentions_of_kmesh()
            ._check_reshapes()
            .plot_k_points()
            .figure,
            name='k-space sampleing')
    assert True, 'manualy checked on 2024-10-23'


def test_convergence():
    assert False

def test_check_reduced_hamiltonian_vanishes(pre_FT_data):
    if fast_mode: return 'Disabled'
    savefig(pre_FT_data.plot_Hr().figure, name='Hr')
    savefig(pre_FT_data.plot_Sr().figure, name='Sr')
    assert True, 'requiers manual checking'
    assert False, 'Needs to find the actual locality somehow'


def test_translational_independence_for_conductivity():
    assert False


def test_compare_tbm_dos_with_exstended_dos(pre_FT_data):
    if fast_mode: return 'Disabled'
    calc = pre_FT_data.init_kspace(50).calculate_dos(alpha=.4)
    savefig(calc.comparitive_dos_plot().figure, name='comparitive dos full vs tbm')
    assert True, 'Manualy checked on 2024-10-23'


def test_check_Z_is_constant():
    assert False


def test_fts():
    'Check all the FTs are unitary'
    assert False


def test_validate_kspace_derivitives():
    assert False


def test_eigen_solver_quality():
    assert False


def test_Graphene_DOS_compare_to_analytic():
    assert False


def test_sigma_tau_ratio_verify():
    assert False


def test_SinvH_after_read(pre_cc_data):
    if fast_mode: return 'Disabled'
    import numpy as np
    print('Testing in the support function indexing')
    SinvH = np.linalg.inv(pre_cc_data.index_by_sf.S) @ pre_cc_data.index_by_sf.H
    diff = (pre_cc_data.index_by_sf.SinvH - SinvH)
    # this is a non-sensical way of measureing the descrepency
    assert np.max(diff)/np.max(SinvH) < 1e-6, np.max(SinvH)/np.average(SinvH) < 1e-6


def test_SinvH_after_fft(pre_FT_data):
    if fast_mode: return 'Disabled'
    'Testing in the compund channel indexing in reciprical space'
    import numpy as np
    data = pre_FT_data.init_kspace(40)

    Sk = data.index_by_cc.S_K
    Hk = data.index_by_cc.H_K
    Sk_inv = np.linalg.inv(Sk.transpose([2, 0, 1])).transpose([1, 2, 0])
    SinvH_k = np.einsum('ijk,jlk-> ilk', Sk_inv, Hk)
    A, B = data.index_by_cc.SinvH_K, SinvH_k

    diff = np.abs(A - B)
    assert np.abs(np.max(diff)/np.max(A)), np.abs(np.average(diff)/np.average(A))


def test_SinvH_after_fft_disk(pre_FT_data):
    if fast_mode: return 'Disabled'
    SinvH_disk = pre_FT_data.index_by_sf.SinvH
    Sk = pre_FT_data.index_by_cc.S_K
    Hk = pre_FT_data.index_by_cc.H_K

    assert False, 'need to calculate SinvH_disk_k somehow'


def test_SinvH_after_diagonalisation():
    assert False
