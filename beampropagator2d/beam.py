# -------------------------------------------

# Created by:               jasper
# as part of the project:   Bachelorarbeit
# Date:                     11/7/19

# --------------------------------------------


import traceback

import numpy as np
import scipy as sp
from numpy.polynomial.hermite import hermval
from typing import Union, Iterable
from .waveguides import *




class Beam:
    """ A parentclass for classes providing methods to compute the initial
    electric field for different types of beams propagateable by 2D FD-BPM
    (e.g. Hermite-Gaussian modes, Eigenmodes...)

    Attributes
    ----------
    wavelength : float
        the wavelength of the propagated radiation beam in relative units
    wavenumber : float
        the wavenumber of the propagated radiation beam in relative units,
        calculated from the wavelength

    """

    def __init__(self, wavelength=1.55):
        """initialize a

        Parameters
        ----------
        wavelength : float
            wavelength of light in mircometers
        """
        print("{} {} {}".format(5 * "#", self.__class__.__name__, (95 - len(self.__class__.__name__)) * "#"))
        self.wavelength = wavelength
        self.wavenumber = 2. * np.pi / wavelength
        print("Initializing beam with wavelength l = {}, k = {}".format(self.wavelength, self.wavenumber))


    def calc_initial_field(self, computational_grid=None):
        print("{} {} {}".format(5 * "#", self.__class__.__name__, (95 - len(self.__class__.__name__)) * "#"))
        print("# Started calculating intial field profile")
        E0 = self._calc_initial_field(computational_grid)
        print("# Finished field calculation")
        print(100 * "#")
        return E0
    def _calc_initial_field(self, computational_grid):
        raise NotImplementedError

    def dump_data(self, filepath):
        """ Dump the beam data into the specified filepath"""

        log.info("Dumping Beam Config to {}".format(filepath))
        f = open(filepath + ".dat", "w")
        self._dump_data(f)
        f.close()

    def read_data(self, filepath):
        log.info("Reading beam config from {}".format(filepath))
        f = open(filepath + ".dat", "r")
        self._read_data(f)
        f.close()

    def _read_data(self, fStream):
        fStream.readline()
        line = fStream.readline()
        line = line.split(" ")
        self.wavelength = float(line[0])
        self.wavenumber = float(line[1])

    def _dump_data(self, fStream):
        fStream.write("lambda k\n")
        fStream.write(str(self.wavelength) + " " + str(self.wavenumber) + "\n")


class GaussianBeam(Beam):

    def __init__(self, wavelength: float = 1.55, order: int = 0, coord_mode: str = "relative",
                 x_coord: float = 0.5, offset: float = 0,
                 width: float = 1, phase_shift: float = 0, angle: float = 0,
                 rel_strength: float = 1):
        """Initializes a Hermite-Gaussian mode initial field.

        This Class provides a single Hermite-Gaussian mode as a initial field for the
        beam propagation. The x-coordinate of the center, width, angle to the optical
        axis, phase, relative intensity and order can be specified.

        Parameters
        ----------
        wavelength : float
            The wavelength of the propagated radiation beam in relative units
        order : int
            The order of the Hermite-Gaussian-Modes. Orders can be mixed via the
            GaussianBeams-Class or the SuperPostion-Class
        coord_mode : {"relative", "absolute"}
            Specifies whether the x_coord attribute is given relative to the
            ComputationalGrid instance used in the calc_inital_field-method
        x_coord : float
            The x coordinate of the beams center
        offset : float
            A offset of the beams center in abs. units from the x_coord-attribute.
            Useful in "relative"-coordinate mode
        width : float
            The width of the beam, i.e. the distance from the beams center where the
            intensity has fallen to 1/e relative to the beams center for a beam with
            order = 0
        phase_shift : float
            the phase of the beam
        angle : float
            The angle between the beams direction of propagation and the optical axis.
            In the paraxial approximation, small values should be used. Has a influence
            on the wavelength and wavenumber that is not! updated
        rel_strength : float
            the beams intensity is normalized and can be rescaled through this attribute
        """
        super().__init__(wavelength)
        self.x_coord = x_coord
        self.width = width
        self.angle = angle
        self.phase_shift = phase_shift
        self.offset = offset
        self.rel_strength = rel_strength
        self.coord_mode = coord_mode
        self.order = np.zeros(order + 1)
        self.order[order] = 1

    def _calc_initial_field(self, computational_grid: ComputationalGrid) -> np.ndarray:
        """Calculates the field of the Hermite-Gaussian mode

        Calculates the Hermite-Gaussian Mode according to the formula
        E = Hermite-poly * Normaldistribution * phase_factor * angle_factor * strength

        Paramters
        ---------
        compuational_grid : ComputationalGrid
            A ComputationalGrid instance storing the spatial dimensions of the
            region and the refractive index distribtution

        Returns
        -------
        np.ndarray
            The initial field as a np.ndarray with dtype=np.complex64 and
            the size computational_grid.N_x
        """
        # calculate inital x coord.
        x = computational_grid.x
        if self.coord_mode == "relative":
            x0 = self.x_coord * x[-1] + (1 - self.x_coord) * x[0] + self.offset
        else:
            x0 = self.x_coord
        # calc the hermite-gaussian beam
        E = np.array(hermval(np.sqrt(2) * (x - x0), self.order)
                     * np.exp(-(x - x0) ** 2 / 2 / self.width / self.width),
                     dtype=np.complex64)

        # normalize that gaussian beam
        norm = np.sqrt(np.trapz(np.abs(E) ** 2, dx=x[1] - x[0]))

        # perform phase shift
        phase_shift = np.exp(1j * self.phase_shift)
        E *= phase_shift

        # shift propagattion angle
        if self.angle != 0:
            E *= np.exp(-1j * computational_grid.n_eff
                        * self.wavenumber / np.sqrt(1 + 1 / np.tan(self.angle) ** 2)
                        * (x - x0))

        E = E / norm * self.rel_strength
        return E

    # def _read_data(self, fStream):
    #     fStream.readline()
    #     super()._read_data(fStream)
    #     fStream.readline()
    #     lines = fStream.readline().split(" ")
    #     self.width = float(lines[0])
    #     self.x_coord = float(lines[1])
    #     self.offset = float(lines[2])

    def _dump_data(self, fStream):
        fStream.write("Mode: Gaussian\n")
        super()._dump_data(fStream)
        fStream.write("width x-coord offset\n")
        fStream.write(
            "{} {} {}\n".format(self.width, self.x_coord, self.offset))


class GaussianBeams(Beam):

    def __init__(self, wavelength: float = 1.55, orders: Union[Iterable[float], float] = 0,
                 coord_mode: str = "relative", x_coords: Iterable[float] = [0.5],
                 offsets: Union[Iterable[float], float] = 0, widths: Union[Iterable[float], float] = 1,
                 phase_shifts: Union[Iterable[float], float] = 0, angles: Union[Iterable[float], float] = 0,
                 rel_strengths: Union[Iterable[float], float] = 1):
        """Create a list of seperate, normalized gaussian beams with uniform
        wavelength using a list of coordinates describing the beams center
        (either in relative or absolute coordinates. In the following, the paramters

        Parameters
        ----------
        wavelength : float
            The wavelength of the propagated radiation beam in relative units
        orders : Union[Iterable[float], float]
            The orders of the Hermite-Gaussian-Modes. If float, all Beams are of
            the same order. If Iterable, the length must be equal to
            x_coords.length
        coord_mode : {"relative", "absolute"}
            Specifies whether the x_coord attribute is given relative to the
            ComputationalGrid instance used in the calc_inital_field-method
        x_coords : Iterable[float]
            The x coordinates of the beams center.
        offsets : Union[Iterable[float], float]
            A offset of the beams center in abs. units from the x_coord-attribute.
            Useful in "relative"-coordinate mode. If Iterable, the length must
            be equal to x_coords.length
        widths : Union[Iterable[float], float]
            The width of the beam, i.e. the distance from the beams center where
            the intensity has fallen to 1/e relative to the beams center for a
            beam with order = 0. If Iterable, the length must be equal to
            x_coords.length
        phase_shifts : Union[Iterable[float], float]
            The phase of the beam. If Iterable, the length must be equal to
            x_coords.length
        angles : Union[Iterable[float], float]
            The angle between the beams direction of propagation and the optical
            axis. In the paraxial approximation, small values should be used.
            Has a influence on the wavelength and wavenumber that is not! updated.
            If Iterable, the length must be equal to x_coords.length
        rel_strengths : Union[Iterable[float], float]
            the beams intensity is normalized and can be rescaled through this
            attribute
        """

        self.wavelength = wavelength
        self.wavenumber = 2 * np.pi / wavelength
        self.beams = []

        for j, i in enumerate(x_coords):
            try:
                angles = angles[i]
            except TypeError:
                pass
            try:
                orders = orders[i]
            except TypeError:
                pass
            try:
                widths = widths[i]
            except TypeError:
                pass
            try:
                offsets = offsets[i]
            except TypeError:
                pass
            try:
                phase_shifts = phase_shifts[i]
            except TypeError:
                pass
            try:
                rel_strengths = rel_strengths[i]
            except TypeError:
                pass
            self.beams.append(
                GaussianBeam(wavelength, orders, coord_mode, i, offsets,
                             widths, phase_shifts, angles, rel_strengths))

    # def _read_data(self, fStream):
    #     fStream.readlines()
    #     super()._read_data(fStream)
    #     fStream.readlines()
    #     self.beams = []
    #     for line in fStream:
    #         lines = line.split(" ")
    #         width = float(lines[0])
    #         x_coord = float(lines[1])
    #         offset = float(lines[2])
    #         self.beams.append(GaussianBeam(self.wavelength, x_coord, width, offset))

    def _dump_data(self, fStream):
        fStream.write("Mode: Gaussian\n")
        super()._dump_data(fStream)
        fStream.write("width x-coord offset\n")
        for i in self.beams:
            fStream.write("{} {} {}\n".format(i.width, i.x_coord, i.offset))

    def _calc_initial_field(self, computational_grid: ComputationalGrid) -> np.ndarray:
        """ Performs GaussianBeam._calc_initial_field for each Beam in instance

        Paramters
        ---------
        compuational_grid : ComputationalGrid

        Returns
        -------
        np.ndarray
            The initial field as a np.ndarray with dtype=np.complex64 and
            the size computational_grid.N_x
        """
        beam = np.zeros(computational_grid.N_x, dtype=np.complex64)
        for i in self.beams:
            beam += i.calc_initial_field(computational_grid)

        return beam


class EigenModes(Beam):

    def __init__(self, wavelength: float,
                 waveguide: Union[WaveguideBase, CombinedWaveguide],
                 selected_mode: Union[Iterable[int], int] = 0,
                 rel_strength: Union[Iterable[float], float] = 1,
                 offset: Union[Iterable[float], float] = 0,
                 phase_shift: Union[Iterable[float], float] = 0,
                 angle_offset: Union[Iterable[float], float] = 0):
        """Calculate the Modes supported by the (input) waveguides of the system
        and chose modes for each input guide and a relative intensity of that mode.

        Parameters
        ----------
        wavelength : float
            The wavelength of the beams radiation
        waveguide : WaveguideBase
            The waveguide object that is used as / contains the input waveguides
        selected_mode : Union[Iterable[int], int]
            The selected modes for each input waveguide, if Iterable, its length
            must be the same as the number of inputs.
        rel_strength : Union[Iterable[float], float]
            The relative intensities of the modes at each input. If Iterable, its length
            must be the same as the number of inputs.
        offset : Union[Iterable[float], float]
            Offset of the eigenmodes from the waveguides center. Useful to test
            reaction of system to imperfections. If Iterable, its length
            must be the same as the number of inputs.
        phase_shift : Union[Iterable[float], float]
            The phase of the eigemodes at each input. If Iterable, its length
            must be the same as the number of inputs.
        angle_offset : Union[Iterable[float], float]
            A offset to the angle of each input beam / waveguide. If Iterable, its length
            must be the same as the number of inputs.
        """
        super().__init__(wavelength)
        self.selected_mode = selected_mode
        self.rel_strength = rel_strength
        self.phase_shift = phase_shift
        self.waveguide = waveguide
        self.offset = offset
        self.angle_offset = angle_offset

    def _dump_data(self, fStream):
        super()._dump_data(fStream)
        fStream.write("Mode: Eigenmodes\n")
        fStream.write("modes intensities\n")
        fStream.write("{} {}".format(self.selected_mode, self.rel_strength))

    # def _read_data(self, fStream):
    #     super()._read_data(fStream)
    #     fStream.readlines(2)
    #     line = fStream.readlines().split(" ")
    #     self.selected_mode = [float(i) for i in line[0]]
    #     self.rel_strength = [float(i) for i in line[1]]

    def _calc_initial_field(self, computational_grid: ComputationalGrid) -> np.ndarray:
        """ Calculate the inital field for a beam propagation according to the
        selected properties. This implementation for the calculation of
        Eigenmodes works for single waveguide structures as well as combinded
        waveguide structures with multible input guides.
        A temporary grid is created and the modes are calculated
        for each input guide seperately.

        If the guide does not hold the selected mode you will be prompted to
        select another on.

        Parameters
        ----------
        computational_grid : CombinedWaveguide
            setup of the simulation grid
        """
        print("# Generating temporary computational grid for mode calculation")
        x_params = computational_grid.x_min, computational_grid.x_max, computational_grid.N_x
        E0 = np.zeros(computational_grid.N_x, dtype=np.complex64)

        temp_grid = ComputationalGrid(x_params, (computational_grid.z_min,
                                                 computational_grid.z_max, 1),
                                      0, 0)
        def calc_mode_and_check_existance(temp_grid, offset, mode):
            # calculate the supported modes
            TE_List = self._determine_guided_modes(temp_grid, offset)
            print("# Found {} supported modes".format(len(TE_List)))

            # set the selected mode

            # write the selected mode into the inital beam if supported by guide
            # or ask for another mode if not supported
            try:
                n_eff, E = TE_List[mode]
            except IndexError:
                print("# Selected mode {} not supported by waveguide".format(mode))
                if input(
                        "Do you want to select another mode? [Y/n]") != "n":
                    print("Supported modes: {}".format(
                        [i for i in range(len(TE_List))]))
                    mode = int(input("Enter mode to select: "))
                    n_eff, E = TE_List[mode]
            except:
                print("something went wrong with the guided modes")
                traceback.print_exc()
                exit()
            print(
                "# Selected mode {}".format(
                    mode))
            return E, n_eff

        if self.waveguide.form == "Combined Waveguide":
            # calculate the modes for each input waveguide seperately

            for i, guide in enumerate(self.waveguide.waveguide_input):

                # write the input guide into the temp grid
                self.waveguide.waveguide_base.write_waveguide(temp_grid)
                # if waveguide has angle, set angle 0 temporarily for mode
                # calculation
                try:
                    angle = guide.angle
                    guide.angle = 0
                except AttributeError:
                    angle = 0
                guide.write_waveguide(temp_grid)
                if angle != 0:
                    guide.angle = angle

                try:
                    mode = self.selected_mode[i]
                except TypeError:
                    mode = self.selected_mode
                try:
                    offset = self.offset[i]
                except TypeError:
                    offset = self.offset

                print("# Calculating Modes")
                E, computational_grid.n_eff = calc_mode_and_check_existance(temp_grid, offset, mode)

                print("# Selected mode has effective refractive index n_0 = {}".format(computational_grid.n_eff))

                norm = np.trapz(np.abs(E) ** 2, dx=computational_grid.dx)

                try:
                    strength = self.rel_strength[i]
                except TypeError:
                    strength = self.rel_strength

                try:
                    phases = self.phase_shift[i]
                except TypeError:
                    phases = self.phase_shift
                try:
                    angles = self.angle_offset[i] + angle
                except TypeError:
                    angles = self.angle_offset + angle
                if angles != 0:
                    if angles < 0:
                        fac = -1
                    else:
                        fac = 1
                    E *= np.exp(fac * -1j * computational_grid.n_eff
                                * self.wavenumber
                                / np.sqrt(1 + 1 / np.tan(angles) ** 2)
                                * (computational_grid.x - guide.guide_x_coord))
                E0 += E / norm * strength * np.exp(phases * 1j)
            norm = np.sqrt(np.trapz(np.abs(E0) ** 2, dx=computational_grid.dx))

            return E0 / norm

        else:
            if hasattr(self.waveguide, "angle"):
                angle = self.waveguide.angle + self.angle_offset
                waveguide_angle = self.waveguide.angle
                self.waveguide.angle = 0
            else:
                waveguide_angle = 0
                angle = self.angle_offset
            self.waveguide.write_waveguide(temp_grid)
            if waveguide_angle != 0:
                self.waveguide.angle = waveguide_angle

            # calculate guided mode at input
            print("# Calculating Modes")
            E0, computational_grid.n_eff = calc_mode_and_check_existance(temp_grid, self.offset, self.selected_mode)
            print("# Selected mode has effective refractive index n_0 = {}".format(computational_grid.n_eff))

            norm = np.sqrt(np.trapz(np.abs(E0) ** 2, dx=computational_grid.dx))
            if hasattr(self.waveguide, "angle") and angle != 0:
                if angle < 0:
                    fac = -1
                else:
                    fac = 1
                E0 *= np.exp(fac * -1j * computational_grid.n_eff
                             * self.wavenumber
                             / np.sqrt(1 + 1 / np.tan(angle) ** 2)
                             * (computational_grid.x - self.waveguide.guide_x_coord))
            return E0 / norm * self.rel_strength * np.exp(self.phase_shift * 1j)


    # ---------------------------------------------------------------------------------------
    # THE FOLLOWING CODE WAS WRITTEN BY OLIVER MELCHERT, MODIFIED BY JASPER MARTINS
    # ---------------------------------------------------------------------------------------

    def _determine_guided_modes(self, computational_grid, offset):
        """compute bound states for mode operator

            Implements mode operator (see Ref. [2]) following Ref. [1] and solves its

            eigenvalues and eigenvectors, respectively.

            Notes
            -----
                Mode operator is self-adjoined, therefore its eigenvalues are real.
                Real eigenvalues (i.e. effective refractive indices) mean lossless
                propagation of the respective eigenvectors (i.e. guided modes).

                Effective refractive indices that refer to actual guiding modes need to
                be larger than the refractive index at infinity. This is required for
                the guiding mode to be normalizable. In turn, this gives a criterion
                to filter for allowed eigenvalues/eigenvector pairs.

            Returns
            -------
                TE (list of tuples): sequence of sorted (effectiveRefractiveIndes,
                    guidingMode) pairs, ordered for increasing values of the first
                    argument. Thus, the eigenvalue/eigenvector pair of lowest order
                    can be found at the zeroth array position.
            Refsobject
            ----
                [1] Dielectric Waveguides
                    Hertel, P.
                    Lecture Notes: TEDA Applied Physiscs School

                [2] Modified finite-difference beam-propagation method based on the
                    Douglas scheme
                    Sun, L. and Yip, G. L.
                    Optics Letters, 18 (1993) 1229
            """
        # IMPLEMENTS MODE OPERATOR FOR TE POLARIZED FIELD - EQ. (2.7) OF REF. [1]

        # reduce accuracy for mode calculation and interpolate the resulting array
        max_steps = 400
        steps = computational_grid.N_x / max_steps
        if steps > 1:
            steps = int(steps)
        else:
            steps = 1
        x, dx = computational_grid.x[::steps], computational_grid.dx * steps
        k0 = self.wavenumber
        nx0 = np.real(computational_grid.n_xz[::steps, 0])
        nb = nx0[0]
        n_max = np.amax(nx0)

        diag = -2.0 * np.ones(x.size) / dx / dx / k0 / k0 + nx0 * nx0
        off_diag = np.ones(x.size - 1) / dx / dx / k0 / k0

        # BEARING IN MIND THAT MODE OPERATOR IS SELF-ADJOINT, AN ALGEBRAIC SOLUTION
        # PROCEDURE FOR HERMITIAN MATRICES (NUMPYS EIGH) CAN BE EMPLOYED
        eigVals, eigVecs = sp.linalg.eigh_tridiagonal(diag, off_diag, select='v', select_range=(nb * nb, n_max * n_max))
        # FILTER FOR ALL EIGENVALUES THAT ARE LARGER THAN THE REFRACTIVE INDEX
        # AT INFINITY. OTHERWISE, THE RESPECTIVE EIGENVECT  ORS WOULD BE OF
        # OSCILLATING TYPE AT INFITY, PREVENTING NORMALIZATION
        TEList = []
        myFilter = eigVals > nb * nb
        for i in range(eigVals.size):
            if myFilter[i]:
                print(np.sqrt(eigVals[i]))
                fac = 1. / np.sqrt(dx) if eigVecs[:,
                                          i].sum() > 0 else -1 / np.sqrt(dx)
                TEList.append((np.sqrt(eigVals[i]),
                               np.interp(computational_grid.x, x + offset,
                                         fac * np.array(eigVecs[:, i], dtype=np.complex64))))

        return sorted(TEList, key=lambda x: x[0], reverse=True)

    # --------------------------------------------------------------------------
    # END OF THE CODE WRITTEN BY OLIVER MELCHERT
    # ---------------------------------------------------------------------------


class SuperPosition(Beam):

    def __init__(self, wavelength: float, GaussianBeams: Iterable[Beam] = None, EigenModes: Iterable[Beam] = None):
        """ Combine Gaussian beams and Eigenmodes arbitrarily

        Parameters
        ----------
        wavelength : float
        GaussianBeams : Iterable[beam]
        EigenModes : Iterable[beam]
        """
        super().__init__(wavelength)
        self.GaussianBeams = GaussianBeams
        self.EigenModes = EigenModes

    def _calc_initial_field(self, computational_grid: ComputationalGrid) -> np.ndarray:
        """

        Parameters
        ----------
        computational_grid

        Returns
        -------

        """
        E0 = np.zeros(computational_grid.N_x, dtype=np.complex64)
        if self.GaussianBeams is not None:
            for i in self.GaussianBeams:
                E0 += i.calc_initial_field(computational_grid)
        if self.EigenModes is not None:
            for i in self.EigenModes:
                E0 += i.calc_initial_field(computational_grid)

        return E0



