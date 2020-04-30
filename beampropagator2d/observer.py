import logging
log = logging.getLogger(__name__)

import numpy as np

from .waveguides import *




class Observer:
    """ Class providing methods for monitoring of all properties of interest of the propagated beam """

    def __init__(self, computational_grid):
        """Create a Observer instance for the collection of storage of the field properties of interest.

        Parameters
        ----------
        x_size : int
            number of grid points in x direction
        z_size : int
            number of grid points in z_direction
        """

        self.computational_grid = computational_grid

        self.dx = self.computational_grid.dx

        self.x_size = computational_grid.N_x
        self.z_size = computational_grid.N_z
        self.efield_intensity = np.zeros(self.z_size, dtype=np.longdouble)
        self.efield_intensity_in_guide = np.zeros(self.z_size, dtype=np.longdouble)
        self.efield_profile = np.zeros((self.z_size, self.x_size), dtype=np.clongdouble)
        self.mode_loss_mismatch = np.zeros(self.z_size, dtype=np.double)
        self.P1 = np.zeros(self.z_size, dtype=np.double)
        self.alpha = np.zeros(self.z_size, dtype=np.double)
        self._propation_step = 0

    def measure(self, current_E_field):
        """Frontend handler for the calculation and collection of the
        properties of interest

        Parameters
        ----------
        current_E_field : np.ndarray
            Elektrikfield at current propagation step

        """


        E = current_E_field
        self.efield_profile[self._propation_step] = E
        E0 = self.efield_profile[0]
        self._calc_efield_intensity(E, self.dx)
        self._calc_efield_intensity_in_guide(E, self.dx)
        self._modeMismatchLoss(self.dx, E0, E)
        self._correlationFunction(self.dx, E0, E)
        self._powerAttenuation(self.efield_intensity[self._propation_step])
        self._propation_step += 1

    def _calc_efield_intensity(self, current_E_field, dx):
        b_mask = self.computational_grid.boundary_mask
        self.efield_intensity[self._propation_step] = np.trapz(
            np.abs(current_E_field*b_mask) ** 2, dx=dx)

    def _calc_efield_intensity_in_guide(self, current_E_field, dx):
        w_mask = self.computational_grid.waveguide_mask
        b_mask = self.computational_grid.boundary_mask
        masked_field = current_E_field * w_mask[:,self._propation_step] * b_mask 
        self.efield_intensity_in_guide[self._propation_step] = np.trapz(
                                            np.abs(masked_field) ** 2, dx=dx)

    # code below this marking taken from Oliver Melchert (modified)
    # --------------------------------------------------------------------------
    def _modeMismatchLoss(self, dx,E0, E):
        """mode mismatch loss

        Implements mode mismatch loss for 2D waveguide according to Ref. [1]

        Args:
            dx (float): meshwidth in x direction
            E0 (numpy array, ndim=1): incident field of the fundamental mode
            E (numpy array, ndim=1): propagating field

        Refs:
            [1] Modified Finite-Difference Beam Propagation Method Based on the
                Generalized Douglas Scheme for Variable Coefficients
                Yamauchi, J. and Shibayama, J. and Nakano, H.
                IEEE Photonics Tech. Lett., 7 (1995) 661

        """
        A = np.abs(np.trapz(E0 * np.conj(E), dx=dx)) ** 2
        B = np.abs(np.trapz(np.abs(E0) ** 2, dx=dx)) ** 2
        self.mode_loss_mismatch[self._propation_step] = (10.0 * np.log10(A / B))

    def _correlationFunction(self, dx, E0, E):
        """correlation function

        Implements complex field-amplitude correlation function following [1]

        Args:
            dx (float): meshwidth in x direction
            E0 (numpy array, ndim=1): incident field of the fundamental mode
            E (numpy array, ndim=1): propagating field

        Refs:
            [1] Computation of mode properties in optical fiber waveguides
                by a propagating beam method
                Feit, M. D. and Flec, J. A. Jr.
                Applied Optics, 19 (1980) 1154

        """
        self.P1[self._propation_step] = np.real(np.trapz(E0 * np.conj(E), dx=dx))

    def _powerAttenuation(self, efield_intensity):
        """power attenuation

        Implements power attenuation following Ref. [1]

        Note:
            for lossless straight waveguide, the power attenuation is due
            solely to numerical dissipation

        Refs:
            [1] Modified finite-difference beam-propagation method based on the
                Douglas scheme
                Sun, L. and Yip, G. L.
                Optics Letters, 18 (1993) 1229

        """

        self.alpha[self._propation_step] = (10. * np.log10(efield_intensity
                                          / self.efield_intensity[0]))
    # --------- End of Code by Oliver Melchert ---------------------------------

    def dump_data(self, fName):
        """dump field

        Parameters
        ----------
        fName : str
            filename (including path) to which ascii output
                should be written
        """
        log.info("Dumping field data to {}_field_field.npy".format(fName))
        fName += "_obs"
        np.save(fName + "_field", self.efield_profile)
        np.save(fName + "_mlm", self.mode_loss_mismatch)
        np.save(fName + "_field_intensity", self.efield_intensity)
        np.save(fName + "_field_int_in_guide", self.efield_intensity)
        np.save(fName + "_power_atten", self.P1)
        np.save(fName + "_alpha", self.alpha)

    def read_data(self, filepath):

        self.efield_profile = np.load(filepath + "_field.npy")
        self.mode_loss_mismatch = np.load(filepath + "_mlm.npy")
        self.efield_intensity = np.load(filepath + "_efield_intensity.npy")
        self.efield_intensity_in_guide = np.load(filepath + "_calc_efield_intensity_in_guide.npy")
        self.P1 = np.load(filepath + "_power_atten.npy")
        self.alpha = np.load(filepath + "_alpha.npy")

    # TODO: Write function saving observables to text file
