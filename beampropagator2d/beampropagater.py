# -------------------------------------------

# Created by:               jasper
# as part of the project:   Bachelorarbeit
# Date:                     11/7/19

# --------------------------------------------
from scipy.linalg import solve_banded as s_t

from .beam import *
from .observer import *
from .waveguides import *
from .helper_classes import  ProgressBar




class BeamPropagator2D:
    """ Collect a Beam, WaveguideBase and ComputationalGrid instance for BPM

    Base-Class that bundles all objects necessary for beam propagation
    (i.e. the comp. grid, waveguide and initial beam) and provides methods
    shared by all propagation schemes.

    Attributes
    ----------
    computational_grid : ComputationalGrid
        A ComputationalGrid instance that stores all the necessary information
        about the spatial dimensions of the system.
    waveguide : WaveguideBase
        A WaveguideBase instance describing the refractive index structure in the
        computational region.
    beam : Beam
        A Beam instance describing the properties of the initial field used for
        the BPM.
    """

    def __init__(self, computational_grid: ComputationalGrid = None,
                 waveguide: Union[WaveguideBase, CombinedWaveguide] = None,
                 beam: Union[Beam, SuperPosition] = None, E0=None, n_eff=None):
        """Create a BeamPropagator2D instance, bundling the structures
        "ComputationalGrid", "Waveguide" and "Beam" in preparation for the
        application of a propagation method. Supports the Methods:

        Parameters
        ----------
        computational_grid : ComputationalGrid
            A ComputationalGrid instance that stores all the necessary information
            about the spatial dimensions of the system.
        waveguide : WaveguideBase
            A WaveguideBase instance describing the refractive index structure in the
            computational region.
        beam : Beam
            A Beam instance describing the properties of the initial field used for
            the BPM.

        """
        print("{} {} {}".format(5*"#", self.__class__.__name__, (95 - len(self.__class__.__name__)) * "#"))
        print("# Intializing beam propagator instance")

        #  computational domain
        self.computational_grid = ComputationalGrid() if computational_grid is None else computational_grid

        # refractive index structure
        self.waveguide = WaveguideBase() if waveguide is None else waveguide
        # initial beam
        self.beam = GaussianBeam() if beam is None else beam

        # prepare a observer instance for data storage
        self.observer = Observer(self.computational_grid)
        # write waveguide structure into the computational grid
        self.waveguide.write_waveguide(self.computational_grid)
        # get the inital field in transversal direction for the grid at hand
        self.E0 = self.beam.calc_initial_field(self.computational_grid) if E0 is None else E0
        self.computational_grid.n_eff = self.computational_grid.n_eff if E0 is None else n_eff


    def run_simulation(self):
        """Excecute the propagation scheme implemented in the class

        Uses the propagate-method provide by the Beampropagator2D subclass used.
        """
        print("# Starting propagation simulation using {} propagtion routine".format(
                self.__class__.__name__))
        self.propagate()
        print("# Finished propagation simulation")

    def propagate(self):
        raise NotImplementedError

    def dump_data(self, filepath):
        log.info("Dumping BeamPropagator Data")
        self.computational_grid.dump_data(filepath + "_grid")
        self.beam.dump_data(filepath + "_beam")
        self.observer.dump_data(filepath + "_obs")



class Chung1990Solver(BeamPropagator2D):
    """Provides a finite difference beam propagation method following the
    crank-nichelson scheme proposed in [1].
    Calculating each step in the direction of propagation requires solving a
    tridiagonal system, which can be done in O(n) using the Thomas Algorithm.
    Here a function for solving banded systems provided by the
    scipy python package is used for efficient handling of large arrays.
    Absorbing boundary conditions are used to limit reflections at the
    limits of the computational domain.

    Notes
    -----
        While the scheme was proposed in [1], this implementation is taken from [2]
        The method employs the paraxial wave approximation in 2 dimensions.

    Refs
    ----
        [1] An Assessment of Finite Difference Beam Propagation Method
            Chung, Y. and Dagli, N.
            IEEE J. Quant. Elect., 26 (1990) 1335

        [2] Beam propagation method for design of optical waveguide devices
            Lifante, Ginés
            John Wiley & Sons, Ltd (2016)
    """

    def propagate(self):
        #
        alpha = 0.5
        # get propagation grid and properties of the light beam for cleaner code
        x, dx = self.computational_grid.x, self.computational_grid.dx
        z, dz = self.computational_grid.z, self.computational_grid.dz
        nxz, n0, k0 = self.computational_grid.n_xz, self.computational_grid.n_eff, self.beam.wavenumber
        # inital beam
        E0 = self.E0
        # write boundary layer: ------------------------------------------------
        k = self.computational_grid.k
        # arrays for holding information needed for the next propagation step temporarily
        E1 = np.array(E0, copy=True, dtype=np.clongdouble)
        E = np.zeros(E0.size, dtype=np.clongdouble)
        # initalize the tridiagonal matrix describing the problem at hand ------
        tri_mat = np.zeros((3, x.size), dtype=np.clongdouble)
        # the upper und lower diagonal is constant
        tri_mat[0, 1:] = tri_mat[2, :-1] = - alpha / dx ** 2
        self.observer.measure(E0)
        # solve each step in z direction ---------------------------------------
        for n in ProgressBar(range(z.size - 1), "BPM"):
            r = np.copy(E1)
            # get refractive index slices for current step ---------------------
            n_1 = nxz[:, n + 1]
            n_0 = nxz[:, n]

            # determine the vector containing the diagonal elements ------------
            tri_mat[1, :] = 2 * alpha / dx ** 2 - alpha * (
                    n_1 ** 2 - k ** 2 - 2 * 1j * n_1 * k -
                    n0 ** 2) * k0 ** 2 + 2 * 1j * k0 * n0 / dz
            # determine the vector containing the right-hand vector ------------
            r[1:-1] = (1 - alpha) / dx ** 2 * (E1[:-2] + E1[2:]) \
                      + ((1 - alpha) * (n_0[1:-1] ** 2 - k[1:-1] ** 2
                                        - 2 * 1j * n_0[1:-1] * k[1:-1] - n0 ** 2)
                         * k0 ** 2
                         - 2 * (1 - alpha) / dx ** 2
                         + 2 * 1j * k0 * n0 / dz) * E1[1:-1]

            # UPDATE FIELD VALUES USING DEDICATED TRIDIAGONAL SOLVER -----------
            # most computational effort.
            E = s_t((1, 1), tri_mat, r, overwrite_ab=False,
                    overwrite_b=False)

            # CALLBACKFUNC OBSERVABLES -----------------------------------------
            self.observer.measure(E)
            # ADVANCE Z-STEP
            E1[:] = E[:]
        # ----------------------------------------------------------------------
