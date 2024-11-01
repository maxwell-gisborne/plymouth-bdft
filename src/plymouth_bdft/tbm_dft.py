from typing import List
import numpy as np
from math import log
from pathlib import Path

from plymouth_bdft.immutiblenamespace import imns

from scipy.io import mmread  # reads Matrix Market in the batrix
from scipy.sparse import csc_matrix  # class for sparse matrix

from plymouth_bdft.plotting_utils import plt, plot_points

import plymouth_bdft.conductivity as conductivity

im = 0+1j


GRAPHENE_HSP = {
        'G1': [0., 0.],
        'M':  [0.5, 0.5],
        'K':  [0.666666, 0.333333],
        'G2': [0., 0.]}


def load_csc_matrix(path):
    with open(path, 'r') as file:
        return np.array(csc_matrix(mmread(file)).todense())


@imns
class DOS:
    alpha_multiple: float = 25520
    samples: np.ndarray = None  # Energy
    gamma: float = 1e2   # Full Width Half Max of smoothing gaussian  [L]
    alpha: float = None  # gamma = alpha / N   [per atom]
    number: int = 300
    range: np.ndarray = None
    density: np.ndarray = None

    def calculate_range(self):
        energy_samples = self.samples
        mine = np.min(energy_samples)
        maxe = np.max(energy_samples)
        e_range = np.linspace(mine, maxe, self.number)

        gamma = self.gamma
        if self.alpha is not None:
            N = energy_samples.shape[0]
            gamma = self.alpha_multiple * self.alpha / N
            # was a nice number so that alpha = 1 gives reasonable results for 100 x 100 k-points

        return self(range=e_range, gamma=gamma)

    def calculate_dos(self):
        # damping = 1/(2 sigma^2)
        # sigma = standad deviation
        # sigma = gamma / (2 sqrt(2 ln(2)))
        # gamma = full width half max

        damping = 4*log(2)*self.gamma**-2
        return self(
                density=np.sum(
                    np.exp(-(damping*(self.range[:, np.newaxis] - self.samples[np.newaxis]))**2),
                    axis=1)
                    )

    def normalise(self):
        s = np.sum(self.density)
        l = self.range[-1] - self.range[0]
        d = self.range[1]  - self.range[0]
        area = s*l/d
        return self(density=self.density / area)

    def plot(self, ax=None, **rest):
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.range, self.density, **rest)
        ax.set_yticks([])
        ax.set_ylabel('Density')
        ax.set_xlabel('Energy')
        ax.set_title('Dos')
        return ax

    def plot_multiple_smoothings(self, alphas: [float], ax=None, **rest):
        for alpha in alphas:
            ax = self(alpha=alpha).calculate_range().calculate_dos().normalise().plot(ax=ax, **({'label': f'$\\alpha = {alpha}$'} | rest))
        ax.set_title(f'TBM DoS while changing smoothing $\\gamma =  {self.alpha_multiple} \\times \\alpha / N$; $N = {self.samples.shape[0]}$ samples')
        return ax


@imns
class Geometry:
    Ovec: np.ndarray
    deltai: np.ndarray
    ai: np.ndarray
    bi: np.ndarray = None

    def calculate_dule_vectors(self):
        bi = []
        for i, ai in enumerate(self.ai):
            v0 = ai
            for j, aj in enumerate(self.ai):
                if i == j: continue
                v0 = v0 - (aj * v0.dot(aj))/(aj.dot(aj))
            bi.append(2*np.pi * v0 / np.dot(v0, ai))
        return self(bi = np.array(bi))._verify_dule_vectors()

    def _verify_dule_vectors(self):
        assert self.bi is not None

        for i, ai in enumerate(self.ai):
            for j, bj in enumerate(self.bi):
                if i == j:
                    assert np.dot(ai, bj) - 2*np.pi < 1e-15
                else:
                    assert np.dot(ai, bj) < 1e-15

        return self

    def lattice_spaceing(self):
        mags = [np.sqrt(np.dot(delta, delta)) for delta in self.deltai]
        assert mags[0] == mags[1] and mags[1] == mags[2], mags
        return float(mags[0])

    def dule_space_interpolate(self, controll_points: List[np.ndarray], delta=.01) -> np.ndarray:
        'Interpolate between a list of points writen as a fraction of the dule space points.'
        points = np.array([[0, 0]])
        offsets = [0]
        for p1, p2 in zip(controll_points[:-1], controll_points[1:]):
            d = p2 - p1
            L = np.linalg.norm(d)
            d = d/L  # d is normalised
            t = np.arange(0, L, delta)  # [0, L)
            coords = np.einsum('t,i->ti', t, d) + p1[None, :]
            stretch = np.einsum('bi,tb->ti', self.bi, coords)
            points = np.concat((points, stretch))
            real_L = np.linalg.norm(stretch[-1, :] - stretch[0, :])  # + np.linalg.norm(np.einsum('bi,b->i', self.bi, d))/len(coords)
            offsets.append(real_L+offsets[-1])
            # TODO: There is some off by one bug in the offsets I think but cba to figure it out, if you make delta smallenough itbecomes irrelevant
            # TODO: Come back to this because its driveing me crazy

        return points[1:, :], offsets  # drop the first point that is only there to make the shapes compatible for the first np.concat call


@imns
class Index_by_Atom:
    positions: np.ndarray = None,
    range: np.ndarray = None

    distence_from_center: np.ndarray = None,
    sort_from_center: np.ndarray = None,
    Symbols: np.ndarray = None

    sublattice: np.ndarray = None
    cell: np.ndarray = None


@imns
class Index_by_Support_Function:
    atom: np.ndarray = None
    channel: np.ndarray = None
    range: np.ndarray = None
    sublattice: np.ndarray = None
    H: np.ndarray = None
    S: np.ndarray = None
    SinvH: np.ndarray = None


@imns
class Index_by_Sublattice:
    range: np.ndarray
    Rmod: np.ndarray


@imns
class Index_by_Compound_Channel:
    range: np.ndarray = None
    sublattice: np.ndarray = None
    channel: np.ndarray = None
    support_function: np.ndarray = None

    H_R: dict = None
    S_R: dict = None
    SinvH_R: dict = None
    lattice_vectors: list = None

    H_K: np.ndarray = None
    S_K: np.ndarray = None
    SinvH_K: np.ndarray = None
    k_points: np.ndarray = None

    Bands: np.ndarray = None
    States: np.ndarray = None


Optional = lambda: None


@imns
class Units:
    E: float
    L: float
    name: str
    comment: str = ""

    def __reper__(self):
        return f'Units({self.name})'


AU = Units(E = 1,
           L = 1,
           name = 'bhor',
           comment = "this is the unit on disk")

eVag = Units(
        E = 27.21138386,  # AU to eV
        L = 0.52917721092,  # Au to Ang
        name = 'angstroem',
        )


@imns
class BigDFT_Data:
    data_dir: Path
    meta_path: Path = 'sparsematrix_metadata.dat'
    H_path: Path = 'hamiltonian_sparse.mtx'
    S_path: Path = 'overlap_sparse.mtx'
    SinvH_path: Path = 'sinvh_sparse.mtx'
    geometry: Geometry = Optional
    index_by_atoms: Index_by_Atom             = Optional
    index_by_sf:    Index_by_Support_Function = None
    index_by_sl:    Index_by_Sublattice       = Optional
    index_by_cc:    Index_by_Compound_Channel = Optional
    dos: DOS = Optional
    conds: [conductivity.Classical_Conductivity] = None
    _comments: list = lambda: list()
    Rmod: np.ndarray = None
    units: Units = eVag
    hsp_offsets: None = None
    hsp_labels: None = None
    kspace: conductivity.KSpace = None
    tags: dict = lambda: {
        'resolved_paths': False,
        'loaded_meta': False,
        'matrixes_loaded': False,
        'projected_2d': False,
        'calculated_center': False,
        'calculated_sublattices': False,
        'calculated_compound_channels': False,
        'calculated_Reduced_Hamiltonian': False,
        'calculated_FT': False,
        'calculated_dTBH': False,
        'initialized_kspace': False,
        'calculated_DOS': False,
        'calculated_conductivity': False,
    }

    def comment(self, *comments):
        return self(_comments = self._comments + list(comments))

    def resolve_paths(self):
        if self.tags['resolved_paths']:
            return self

        def exists(path):
            assert path.exists(), path
            return path
        data_dir = exists(self.data_dir).absolute()

        return self(
            data_dir = data_dir,
            meta_path = exists(data_dir.joinpath(self.meta_path)),
            H_path = exists(data_dir.joinpath(self.H_path)),
            S_path = exists(data_dir.joinpath(self.S_path)),
            SinvH_path = exists(data_dir.joinpath(self.SinvH_path)),
            tags = self.tags | {'resolved_paths': True}
        )

    def load_matrixes(self):
        assert self.tags['resolved_paths']
        if self.tags['matrixes_loaded']:
            return self

        return self(
            index_by_sf = self.index_by_sf(
                H = load_csc_matrix(self.H_path) * self.units.E,
                S = load_csc_matrix(self.S_path),
                SinvH = load_csc_matrix(self.SinvH_path) * self.units.E,
            ),
            tags = self.tags | {'matrixes_loaded': True}
        )

    def load_meta(self):
        filename = self.meta_path
        assert self.meta_path is not None, 'reslove_paths first'
        assert filename.exists()

        positions = []           # atom-index => ionic cartisian position
        symbols = []             # atom-index => symbol-index
        symbols_dictionary = []  # symbol-index => String of atom type
        atom = []                # alpha => atom-index
        chanl = []               # alpha => channel-index

        with open(filename, "r") as ifile:
            # Read the first line
            matinfo = next(ifile).split()
            matdim, natoms, ntypes = [int(x) for x in matinfo[:3]]

            symbol_index = range(ntypes)
            atom_index = range(natoms)
            alpha = range(matdim)

            next(ifile)          # Units
            line = next(ifile)   # skip geocode
            line = next(ifile)   # skip shift

            # generate symbols_dictionary
            for i, line in zip(symbol_index, ifile):
                nz, nelpsp, name = line.split()[:3]
                symbols_dictionary.append(name)

            # generate positions and symbols
            for i, line in zip(atom_index, ifile):
                sym_indx, x, y, z, *_ = line.split()
                symbols.append(symbols_dictionary[int(sym_indx) - 1])
                positions.append([float(x), float(y), float(z)])

            # generate SFs
            channel_count = {}
            for i, line in zip(alpha, ifile):
                atom_index = int(line.split()[0]) - 1
                atom.append(atom_index)
                cc = channel_count[atom_index] = (channel_count.get(atom_index) or 0) + 1
                chanl.append(cc-1)

            return self(
                index_by_atoms = Index_by_Atom(
                    positions = np.array(positions) * self.units.L,
                    Symbols = np.array(symbols),
                    range = np.arange(len(symbols)),
                ),
                index_by_sf = self.index_by_sf(
                    atom = np.array(atom),
                    channel = np.array(chanl),
                    range = np.arange(len(atom)),
                ),
                tags = self.tags | {'loaded_meta': True}
            )

    def project_to_surface(self):
        if self.tags['projected_2d']:
            return self

        return self(
            index_by_atoms = self.index_by_atoms(
                positions = self.index_by_atoms.positions[:, (2, 0)]  # (X,Y,Z) -> (X',Y') = (Z,X)
            ),
            tags = self.tags | {'projected_2d': True},
        )

    def calculate_geometry(self, center_point: np.ndarray = None):
        if center_point is None:
            # default set the center point to be the point the average positions.
            center_point = np.sum(self.index_by_atoms.positions, axis=0) / len(self.index_by_atoms.positions)

        distance_from_center = np.sqrt(np.sum((self.index_by_atoms.positions - center_point)**2, axis=1))

        index_by_distence = np.argsort(distance_from_center)
        center_index, nerist_nabour_index = index_by_distence[:2]
        center = self.index_by_atoms.positions[center_index]

        nerist_nabour = self.index_by_atoms.positions[nerist_nabour_index]

        delta0 = nerist_nabour - center
        delta1 = .5 * np.array([[-1, np.sqrt(3)], [-np.sqrt(3), -1]]) @ delta0
        delta2 = .5 * np.array([[-1, -np.sqrt(3)], [np.sqrt(3), -1]]) @ delta0
        a0 = delta0 - delta2
        a1 = delta0 - delta1
        ai = np.array([a0, a1])
        deltai = np.array([delta0, delta1, delta2])

        return self(
            geometry = Geometry(
                deltai = deltai,
                ai = ai,
                Ovec = center,
            ).calculate_dule_vectors(),
            index_by_atoms = self.index_by_atoms(
                distence_from_center = distance_from_center,
                sort_from_center = index_by_distence,
            ),
            tags = self.tags | {'calculated_center': True}
        )

    def calculate_sublattice(self,
                             nudge = 1e-6 * np.array([1, 1]),
                             tol = 9):
        ''' This function finds the assignes to each support function, which lattice, and sublattice it belongs too.
            This information is stored using the index_by system.
            If, for example, you want to find the sublattices corresponding to the each atom, you can find it in `self.index_by_atoms.sublattice`
            This is an array, whos length is the number of atoms in your system, and whos value is the corresponding sublattices.
            Since the Hamiltonian is daved on disk as a sparse array whos indexes correspond to support functions,
            the full Hamiltonian from the disk is stored in the within `self.index_by_sf`.
            '''

        assert self.tags['calculated_center']
        if self.tags['calculated_sublattices']:
            return self

        ai = self.geometry.ai
        Ovec = self.geometry.Ovec
        Rtilde = self.index_by_atoms.positions

        ai_inv = np.linalg.inv(ai)
        Rtilde = Rtilde - Ovec
        Rtilde_cs = np.einsum('ij,ni->nj', ai_inv, Rtilde)

        div, mod = np.divmod(Rtilde_cs + nudge, 1)
        mod = mod - nudge
        Rmod = np.unique(np.round(mod, tol), axis=0)
        Rmod_index = np.array([np.sum((p - mod)**2, axis=1) < 10**-tol
                               for p in Rmod])

        assert np.all(np.logical_xor(Rmod_index[0], Rmod_index[1]))

        sl_index = (
            np.arange(Rmod_index.shape[0])[np.newaxis] * np.ones(Rmod_index.shape[1])[:, np.newaxis]
        )[Rmod_index.transpose()].astype(int)

        return self(
            index_by_atoms = self.index_by_atoms(
                sublattice = sl_index,
                cell = div,
            ),
            index_by_sl = Index_by_Sublattice(
                range = np.arange(Rmod.shape[0]),
                Rmod = Rmod,
            ),
            index_by_sf = self.index_by_sf(
                sublattice = sl_index[self.index_by_sf.atom]),
            tags = self.tags | {'calculated_sublattices': True},
        )

    def reconstructed_positions(self):
        sublattices = self.index_by_atoms.sublattice
        cells = self.index_by_atoms.cell
        return np.einsum('ai,ja->ji', self.geometry.ai, self.index_by_sl.Rmod[sublattices] + cells) + self.geometry.Ovec

    def calculate_compound_channels(self):
        '''The compound channels are the residual indexes of the reduced hamiltonian.
        Calculating the reducded hamiltonion involves summation over the lattice index,
        Thus the compound channels are the sublattice index (sl) along with
        the degrees of freedome that are not related to the position of the corresponding orbital (ch).
        This function creates the `index_by_cc` namespace which is where the reduced hamiltonian will be stored.'''

        assert self.tags['calculated_sublattices']
        if self.tags['calculated_compound_channels']:
            return self

        indexes = []
        channels = np.unique(self.index_by_sf.channel)
        for sl in self.index_by_sl.range:
            for ch in channels:
                indexes.append((sl, ch))
        number_of_compund_channels = len(indexes)

        support_functions = number_of_compund_channels*[0]
        lattice_vectors  = number_of_compund_channels*[0]

        for B in range(len(indexes)):
            sl, ch = indexes[B]
            sfs = support_functions[B] = self.index_by_sf.range[
                (self.index_by_sf.channel == ch) & (self.index_by_sf.sublattice == sl)
            ]
            cells = self.index_by_atoms.cell[self.index_by_sf.atom[sfs]]
            lattice_vectors[B] = np.einsum('ai,na->ni', self.geometry.ai, cells)
        return self(
            index_by_cc = Index_by_Compound_Channel(
                range = np.arange(len(indexes)),
                channel = np.array([ch for ls, ch in indexes]),
                sublattice = np.array([sl for sl, ch in indexes]),
                support_function = np.array(support_functions),
                lattice_vectors = lattice_vectors,
            ),
            tags = self.tags | {'calculated_compound_channels': True}
        )

    def calculate_Reduced_Hamiltonian(self):
        ''' This function reindexes the elements from the full Hamiltonian, and S-matrix,
        into a reduced form such that  $H'[B1,B2][i] = H_{ (B1,0), (B2, i)}$ where B1,B2 are the compound channel indexes,
        i is the cell index, and 0 is central cell.'''

        assert self.tags['calculated_compound_channels']
        if self.tags['calculated_Reduced_Hamiltonian']:
            return self

        H = {}
        S = {}
        SinvH = {}

        # self.index_by_atoms.range[np.all(self.index_by_atoms.cell == [0, 0], axis=1)]

        for B1 in self.index_by_cc.range:
            B1_compatible_alphas = self.index_by_cc.support_function[B1]

            # This line finds the support function indexes which correspond to the compound channel
            # in which the atomic position is at the origin.
            alpha1 = B1_compatible_alphas[
                np.all(self.index_by_atoms.cell[self.index_by_sf.atom[B1_compatible_alphas]] == [0, 0], axis=1)
            ]

            assert alpha1.shape[0] == 1, (alpha1.shape, B1, self.index_by_cc.lattice_vectors[B1], self.index_by_cc.channel[B1])

            for B2 in self.index_by_cc.range:
                alpha2 = self.index_by_cc.support_function[B2]
                H[B1, B2] = self.index_by_sf.H[alpha1[0], alpha2]
                S[B1, B2] = self.index_by_sf.S[alpha1[0], alpha2]
                SinvH[B1, B2] = self.index_by_sf.SinvH[alpha1[0], alpha2]

        return self(
            index_by_cc = self.index_by_cc(
                H_R = H,
                S_R = S,
                SinvH_R = SinvH,
            ),
            tags = self.tags | {'calculated_Reduced_Hamiltonian': True}
        )

    def calculate_FT(self, k_points: np.ndarray, append=False):
        '''This function finds the furier transform of the reduced hamiltonion and S-matrix onto the k-points provided.'''
        assert self.tags['calculated_Reduced_Hamiltonian']

        N_KPs = k_points.shape[0]

        cc_number = self.index_by_cc.range.shape[0]
        H = np.zeros((cc_number, cc_number, N_KPs))*im  # 8 by 8 by number-of-kpoints
        S = np.zeros_like(H)*im
        SinvH = np.zeros_like(H)*im

        # TODO: replace the following with an fft library routine.
        # this requies k_points to be provided as linspace config
        # as opposed to a set of points, this may complicate DOS calculations.

        for B1 in self.index_by_cc.range:
            for B2 in self.index_by_cc.range:
                Rvecs = self.index_by_cc.lattice_vectors[B2]
                phase = np.exp(-im*np.einsum('ki,ri->kr', k_points, Rvecs))
                # Note that, when it comes to the calculation of band energies, the normalisation of the following ft does not matter,
                # since band energies come from solveing the equation ES|state> = H|state>
                H[B1, B2] = np.einsum('kr,r', phase, self.index_by_cc.H_R[B1, B2])
                S[B1, B2] = np.einsum('kr,r', phase, self.index_by_cc.S_R[B1, B2])
                SinvH[B1, B2] = np.einsum('kr,r', phase, self.index_by_cc.SinvH_R[B1, B2])

        if self.tags['calculated_FT'] and append:

            return self(
                index_by_cc = self.index_by_cc(
                    H_K = np.concatenate((self.index_by_cc.H_K, H), axis=2),  # axis-2 = k-points
                    S_K = np.concatenate((self.index_by_cc.S_K, S), axis=2),
                    SinvH_K = np.concatenate((self.index_by_cc.SinvH_K, SinvH), axis=2),
                    k_points = np.concatenate((self.index_by_cc.k_points, k_points), axis=1),
                ),
                tags = self.tags | {'calculated_dTBH': False, 'calculated_DOS': False}
            )

        return self(
                index_by_cc = self.index_by_cc(
                        H_K = H,
                        S_K = S,
                        SinvH_K = SinvH,
                        k_points = k_points,
                     ),
                tags = self.tags | {'calculated_FT': True, 'calculated_dTBH': False, 'calculated_DOS': False}
                 )

    def verify_SinvH(self):
        # TODO: Why arnt these two things the same?
        # Which one to use?
        # NOTE: Sam uses S, not SinvH

        assert self.tags['calculated_FT']

        print('Testing in the support function indexing')
        SinvH = np.linalg.inv(self.index_by_sf.S) @ self.index_by_sf.H
        diff = (self.index_by_sf.SinvH - SinvH)
        # this is a non-sensical way of measureing the descrepency
        assert np.max(diff)/np.max(SinvH) < 1e-6, np.max(SinvH)/np.average(SinvH) < 1e-6

        # print('testing in the compund channel indexing in direct space')
        print('I dont know how to  test SinvH in the compund channel indexing in direct space')
        # SR = self.index_by_cc.S_R
        # HR = self.index_by_cc.H_R
        # SR_inv = np.linalg.inv(SR.transpose([2, 0, 1])).transpose([1, 2, 0])
        # SinvH_R = np.einsum('ijk,jlk-> ilk', SR_inv, HR)  # TODO: compare this with SinvH from disk
        # A, B = self.index_by_cc.SinvH_R, SinvH_R
        # diff = np.abs(1 - A/B)
        # return np.max(diff), np.average(diff)

        print('Testing in the compund channel indexing in reciprical space')
        Sk = self.index_by_cc.S_K
        Hk = self.index_by_cc.H_K
        Sk_inv = np.linalg.inv(Sk.transpose([2, 0, 1])).transpose([1, 2, 0])
        SinvH_k = np.einsum('ijk,jlk-> ilk', Sk_inv, Hk)  # TODO: compare this with SinvH from disk
        A, B = self.index_by_cc.SinvH_K, SinvH_k

        diff = np.abs(A - B)
        return np.abs(np.max(diff)/np.max(A)), np.abs(np.average(diff)/np.average(A))

    def calculate_bands(self, controll_points = GRAPHENE_HSP, delta=.01):
        ''' This function calculates the band structure of the hailtonian by 1)
            interpelating in the dule space between the `controll_points` with a step side of delta,
            2) calculating the tight binding hamiltonian along that path, and 3) diagonalising that TBH.
            returning the result'''

        print(controll_points)
        if issubclass(dict, type(controll_points)):
            print('its a dict')
            labels = list(controll_points)
            controll_points = list(controll_points.values())
            print(labels)
        else:
            print('its not a dict')
            labels = None

        if not issubclass(np.ndarray, type(controll_points)):
            print('lets turn it into an arry')
            controll_points = np.array(controll_points)

        kpoints, cp_offsets = self.geometry.dule_space_interpolate(controll_points, delta=delta)

        return self.calculate_FT(kpoints).diagonalise_TBH()(
                    hsp_labels = labels,
                    hsp_offsets = cp_offsets
                )

    def diagonalise_TBH(self, states=None):
        ''' This function calculates eigen valies and vectors of the Furia transformed reduced hamiltonain.
            this corresponds to the band structure.
            However the bands are sorted by magnatude of eigen value, as opposed to the band quantum number.'''

        assert self.tags['calculated_FT']
        if self.tags['calculated_dTBH']:
            return self

        # #Note On Transposeing
        # I'm using the numpy inverse, and eigen solver,
        # which is good in the fact that they can operate on arrays of matricies
        # but are bad in the fact that they seem to produce inacurate results.
        # They work on the last two indexes of the array, and they need them to be square arrays
        # I'm like the first two indexes to be the two B indexes (or E and B indexes once diagonalised),
        # and the third nidex to be the k-point index, so in order to use these rooteines
        # I need to transpose the data. `transpose([2,0,1])` puts the k-point index first,
        # and moves the two B indexes to the end, allowing the `.inv` and `.eing` rooteins to be used.
        # The `transpose([1,2,0])` undose this returning the k-point index to the last place.

        Sk = self.index_by_cc.S_K
        Hk = self.index_by_cc.H_K
        Sk_inv = np.linalg.inv(Sk.transpose([2, 0, 1])).transpose([1, 2, 0])
        SinvH_k = np.einsum('ijk,jlk-> ilk', Sk_inv, Hk)  # TODO: This this differes from SinvH as found on disk, and im not sure why. `verify_SinvH()`
        bands, states = np.linalg.eig(SinvH_k.transpose([2, 0, 1]))  # index of states = (k-points, B-index, band)
        assert np.all(np.abs(bands.real) / np.abs(bands) - 1  < 1e-20), np.max(np.abs(bands.real) / np.abs(bands) - 1)
        assert len(bands.shape) == 2, bands.shape
        assert len(states.shape) == 3, states.shape
        bands = bands.real
        bands = bands.transpose([1, 0])       # indexes = (n_E, k-points)
        states = states.transpose([2, 1, 0])  # indexes = (n_E, B-index, k-points) were n_E is the eigen value
        assert bands.shape[0] == states.shape[0], (bands.shape, states.shape)

        # # Explination of sorting:
        # argsort returns an array of the same shape as its input,
        # its contents are the index, along the given axis, of the the sorted element
        # so that for a (10x2) array, argsort(axis=0) will be a (10x2) array,
        # the elements will run from 0->9, and each of the columns (axis-0) will have exactly one of these intigers
        # we can then perform the sort by using np.take_along_axis. It takes an input array, an index array, and an axis array.
        # It returns an array of the same shape of the index array. it interprets the value of the index array,
        # as an index along the given axis of the input array. The rest of the indexes are taken to be the same
        # as the index for the index array. So for an index of shape (10x5x3), with axis=1,
        # the output array will be of shape (10x5x3), and the value for the output at [8,3,1]
        # will be the input at [8,i,1] were i is the value of the index at [8,3,1].
        # To sort the eigen values is thus eaisy, simply argsort with axis = band-index = 0
        # and then take_along_axis with the same axis=band-index=0
        # However, to sort the eigen vectors by the eigen values, one needs to raise the sort to the same shape as the eigen vector array (`states`).
        # We need the elements of the states to be un-permuted, as they correspond to compound channel indexes, and have definite meaning (B=0 <=> sl=0, ch=0)
        # To do this, I duplicate the index that sorts by eigen value, by the number of Bs, and insert the copies in the B-index (axis=1)
        # That is waht is whats happening to calculate `sort_index_states` from `sort_index` and the number of B-indexs `states.shape[1]`

        # V = np.sum(np.sum(np.einsum('nak,nbk->nabk', states,states), axis=1), axis=1)
        sort_index = np.argsort(states[:, 0, :], axis=0)  # sort by B0 component
        sort_index = np.argsort(bands, axis=0)  # bands by energy
        sort_index_states = np.intc(np.einsum('ab,i->aib', sort_index, np.ones(states.shape[1])))

        bands = np.take_along_axis(bands, sort_index, axis=0)
        states = np.take_along_axis(states, sort_index_states, axis=0)
        assert len(bands.shape) == 2, bands.shape

        return self(
            index_by_cc = self.index_by_cc(
                Bands = bands,
                States = states,
            ),
            tags = self.tags | {'calculated_dTBH': True},
        )

    def init_kspace(self, N: int, padding=None):
        if padding is None:
            padding = (0, 0)

        assert self.tags['calculated_center']
        kspace = conductivity.KSpace().init_with_hex(self.geometry.bi, N, padding)
        return self(
                kspace = kspace,
                tags = self.tags | {'initialized_kspace': True},
        ).calculate_FT(kspace.reshape_mesh_to_points(kspace.k_mesh())).diagonalise_TBH()

    def calculate_dos(self, number=300, alpha = None, gamma=.5):
        assert self.tags['calculated_dTBH']
        assert self.kspace is not None

        energy_samples = self.index_by_cc.Bands

        assert energy_samples.shape[0] == 8, energy_samples.shape
        mask = self.kspace.mask.reshape(*energy_samples.shape[1:])

        energy_samples = energy_samples[:, mask]
        assert np.prod(energy_samples.shape) < np.prod(self.index_by_cc.Bands.shape)

        energy_samples = energy_samples.reshape(-1)

        dos = (DOS(
            samples = energy_samples,
            alpha = alpha,
            gamma = gamma,
            number = number)
               .calculate_range()
               .calculate_dos()
               .normalise()
              )

        return self(
                dos = dos,
                tags = self.tags | {'calculated_DOS': True},
        )

    def calculate_conductivity(self):
        assert self.kspace is not None
        assert self.tags['calculated_dTBH'], 'try initialising kspace with the `.init_kspace(N)` method'

        kspace = self.kspace

        conds = []
        for band in self.index_by_cc.Bands:
            energies = kspace.reshape_points_to_mesh(band)[:, :, 0]
            conds.append(conductivity.Classical_Conductivity(
                kspace=kspace,
                Energy=energies)
                         .calculate_kernel_factor()
                         .calculate_conductivity()
                     )

        return self(
                conds = conds,
                dos = self.dos,
                tags = self.tags | {'calculated_conductivity': True}
        )

    def get_sigmas(self):
        assert self.tags['calculated_conductivity'], 'please canculate the conductivities first'
        return np.array([cond.sigma for cond in self.conds])

    def calculate_rhos(self, l=1):
        assert self.tags['calculated_conductivity'], 'please canculate the conductivities first'

        return np.array([cond.calculate_rho(l=l) for cond in self.conds])

    def calculate_dos_full(self):
        Es = np.linalg.eigvals(self.index_by_sf.SinvH)
        return DOS(samples=Es.real)

    def eigen_error(self):
        vals = self.index_by_cc.Bands
        vecs = self.index_by_cc.States
        Hk = self.index_by_cc.H_K
        Sk = self.index_by_cc.S_K
        return np.abs(
                np.einsum('ndk,nbk->nk',
                          vecs.conj(),
                          np.einsum('bdk,ndk->nbk',
                                    Hk, vecs
                                    ) - np.einsum('nk,bdk,ndk->nbk',
                                                  vals, Sk, vecs
                                                  )
                          ) / vals
        )

    def verify_geometry(self):
        given_positions = self.index_by_atoms.positions
        recon_positions = self.reconstructed_positions()
        assert np.all(given_positions - recon_positions < 1e-9)

        cells = self.index_by_atoms.cell
        lat_vectors = np.einsum('ai,na->ni', self.geometry.ai, cells)

        recon_positions2 = lat_vectors + self.geometry.Ovec + np.einsum('ai,ja->ji', self.geometry.ai, self.index_by_sl.Rmod[self.index_by_atoms.sublattice])
        assert np.all(given_positions - recon_positions2 < 1e-9)

        for B in self.index_by_cc.range:
            atoms = self.index_by_sf.atom[self.index_by_cc.support_function[B]]
            given_positions = self.index_by_atoms.positions[atoms]

            sublattice = self.index_by_sl.Rmod[self.index_by_cc.sublattice[B]]
            lattice_vectors = self.index_by_cc.lattice_vectors[B]
            recon_positions3 = self.geometry.Ovec + np.einsum('ai,a->i', self.geometry.ai, sublattice) + lattice_vectors
            assert np.all(given_positions - recon_positions3 < 1e-9)

        return self

    def plot_eigen_errors(self, ax=None, averaged=True):
        if ax is None:
            _, ax = plt.subplots()

        ks = self.index_by_cc.k_points[:, 0]
        er = self.eigen_error()

        if averaged:
            ax.plot(ks, np.average(er, axis=0))
        else:
            for n, e in enumerate(er):
                ax.plot(ks, e, label = f'n = {n}')
        ax.set_xlabel('$k_x$')
        ax.set_title(r'Error In Solutions = $ \left\langle\phi_n\middle| H - E_n S \middle|\phi_n\right\rangle / E_n$ '
                     + ('(averaged over $n$)' if averaged else ''))
        ax.set_ylabel(f'min,av,max = ${np.min(er):.1E}$, ${np.average(er):.1E}$, ${np.max(er):.1E}$')
        return ax

    def plot_distance_from_center(self, ax=None):
        if ax is None:

            _, ax = plt.subplots()
        plot_points(ax, self.index_by_atoms.positions,
                    s = .1+100/self.index_by_atoms.distence_from_center,
                    c = np.log(self.index_by_atoms.distence_from_center))
        ax.set_title('Ionic Positions, points closer to the center are larger')
        return ax

    def plot_Hr(self, ax = None):
        assert self.tags['calculated_sublattices']
        if ax is None:
            _, ax = plt.subplots()
        for b1 in self.index_by_cc.range:
            lv = self.index_by_cc.lattice_vectors[b1]
            Hr = np.zeros(lv.shape[0])
            for b2 in self.index_by_cc.range:
                Hr += np.abs(self.index_by_cc.H_R[(b1, b2)])
            plot_points(ax, lv,
                        s = 100*np.log10(1+np.abs(Hr)))

        ax.set_title(r"Lattice Points weighted by $\sum_{b'} \left|H_{bb'}(\vec R)\right|$")
        ax.set_xlabel(r"scail $\propto log_{10} \left(1 +  \sum_{b'} \left| H_{bb'}(\vec R)\right|\right)$")

        return ax

    def plot_Sr(self, ax = None):
        assert self.tags['calculated_sublattices']
        if ax is None:
            _, ax = plt.subplots()
        for b1 in self.index_by_cc.range:
            lv = self.index_by_cc.lattice_vectors[b1]
            Sr = np.zeros(lv.shape[0])
            for b2 in self.index_by_cc.range:
                Sr += np.abs(self.index_by_cc.S_R[(b1, b2)])
            plot_points(ax, lv,
                        s = 100*np.log10(1+np.abs(Sr)))

        ax.set_title(r"Lattice Points weighted by $\sum_{b'} \left|S_{bb'}(\vec R)\right|$")
        ax.set_xlabel(r"scail $\propto log_{10} \left(1 +  \sum_{b'} \left| S_{bb'}(\vec R)\right|\right)$")

        return ax

    def plot_atoms(self, number=100, add_geometry=False, ax=None):
        assert self.tags['calculated_center']

        if ax is None:
            _, ax = plt.subplots()
        atoms = self.index_by_atoms.positions
        by_distance = self.index_by_atoms.sort_from_center

        if self.tags['calculated_sublattices']:
            sl = self.index_by_atoms.sublattice[by_distance]
            plot_points(ax,
                        atoms[by_distance][:number] - self.geometry.Ovec,
                        c=sl[:number][:100])

        else:
            plot_points(ax, atoms[by_distance][:number])

        if add_geometry:
            a0, a1 = self.geometry.ai
            delta0, delta1, delta2 = self.geometry.deltai

            ax.plot([0, a0[0]], [0, a0[1]], color='black')
            ax.plot([0, a1[0]], [0, a1[1]], color='black')

            ax.plot([0, delta0[0]], [0, delta0[1]], color='red',   label=r'$\vec\delta_0$')
            ax.plot([0, delta1[0]], [0, delta1[1]], color='green', label=r'$\vec\delta_1$')
            ax.plot([0, delta2[0]], [0, delta2[1]], color='blue', label=r'$\vec\delta_2$')
            ax.legend()

        return ax

    def plot_bands(self, ax=None, **kwds):

        assert self.tags['calculated_dTBH']
        if ax is None:
            _, ax = plt.subplots()

        k_points = self.index_by_cc.k_points

        dk = np.linalg.norm(k_points[1:]-k_points[:-1], axis=1)
        k_path = np.concatenate(([0], np.cumsum(dk)))

        # l = np.array([-r[1], r[0]])
        # on_line = np.abs(np.einsum('ki,i', k_points - c, l)) < 1e-3
        # k_points = k_points[on_line]

        for n, b in enumerate(self.index_by_cc.Bands):
            # ax.plot(np.einsum('i,ki', r, k_points), b[on_line], label=f'$n_E={n}$')
            ax.plot(k_path, b, **{'label': f'$n_E={n}$', **kwds})

        if self.hsp_labels is not None:
            ax.set_xticks(self.hsp_offsets, self.hsp_labels)

        ax.set_xlabel('k-points')
        ax.set_ylabel('Energy')
        ax.set_title('Band Energy')
        ax.set_xlim([0, k_path[-1]])

        return ax

    def plot_block_components(self):
        fig, axs = plt.subplots(4, 2)

        for i in range(4):
            for j in range(2):
                for comp in np.abs(self.index_by_cc.States)[i+4*j]:
                    axs[i, j].plot(self.index_by_cc.k_points[:, 0], comp, linewidth=.75)
                    axs[i, j].set_ylabel(f'n={i+4*j}')
                    axs[i, j].set_yticks([])
                    if i != 3:
                        axs[i, j].set_xticks([])
                    else:
                        axs[i, j].set_xlabel('K-points')

        fig.figure.suptitle('Components of Block States')
        fig.figure.tight_layout()
        return fig

    def plot_k_scailing(self, ax1 = None, ax2 = None):

        if ax2 is None:
            _, ax2 = plt.subplots()

        points = [(10, 1.9),
                  (20, 1.15),
                  (30, 1),
                  (40, .85),
                  (50, .83),
                  (60, .8),
                  (100, .6),
                  ]

        ax2.scatter(*list(zip(*[(gamma, N**2) for gamma, N in points])), c='b')
        ax2.set_xlabel('k-points sampled')
        ax2.set_ylabel('smoothing peramiter $\\gamma$')
        ax2.set_title('$(N, \\gamma)$ points chosen')

        for N, gamma in points:
            calc_N  = (self
                       .init_kspace(N)
                       .verify_geometry()
                       .calculate_dos(gamma=gamma)
                       )
            ax1 = calc_N.dos.plot(ax1, label=f'N={N}')

        ax1.figure.legend()
        return ax1, ax2

    def comparitive_dos_plot(self, ax=None):
        fd_gamma = .17
        full_dos = self.calculate_dos_full()(alpha=fd_gamma).calculate_range().calculate_dos().normalise()
        ax = self.dos.plot(ax=ax, label='TBM DoS $\\gamma = {self.dos.gamma}$')
        ax = full_dos.plot(ax = ax, label='Full DFT DoS $\\gamma = {fd_gamma}$')
        ax.figure.legend()
        return ax

    def __repr__(self):
        tags = self.tags
        width = max([len(tag) for tag in tags]) + 5
        title_line = f'''{self.__class__.__qualname__}'''
        title_line += '\n' + '-' * len(title_line) + '\n'
        return (title_line + f'''data_dir = "{self.data_dir}",\n\t'''
                + '\n\t'.join([str(tag).rjust(width) + f':\t[{status}]' for tag, status in self.tags.items()])
                + ('\n COMMENTS:\n\t' + '\n\t'.join([str(c) for c in self._comments])
                   if len(self._comments) != 0 else '')
                )

    def report_meta(self):
        string = []

        def title(name):
            l = len(name)
            a = l // 2
            b = (l // 2) + (l % 2)
            return '\n' + ' '.join([(30-a)*'=', name, (30 - b)*'=']) + '\n'

        shape = lambda array: f'Shape = {array.shape}'.rjust(50)

        string.append(title('Positions'))
        string.append(shape(self.index_by_atoms.positions))
        string.append(self.index_by_atoms.positions)

        string.append(title('Support Functions'))
        string.append(shape(self.index_by_sf.atom))
        string.append(
            np.array([self.index_by_sf.atom, self.index_by_sf.channel]).transpose()
        )

        string.append(title('Symbols'))
        string.append(shape(self.index_by_atoms.Symbols))
        string.append(self.index_by_atoms.Symbols)

        return '\n'.join(map(str, string))

