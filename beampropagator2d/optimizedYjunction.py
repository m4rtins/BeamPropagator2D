import numpy as np

from .beam import *
from .waveguides import *
from .indexcalculator import *
from .beampropagater import *
from .observer import *


class OptimizedYJunction(CombinedWaveguide):

    def __init__(self, x_params, z_params, wavelength,
                 refractive_index_guide=1.52, refractive_index_medium=1.5,
                 half_angle=3, branching_region_x=(-15, 15),
                 branching_region_z=(50, 200), guide_x_coord=0,
                 x_resolution=200,
                 z_resolution=200, guide_seperation=4, width=3,
                 index_levels=np.array([1.5, 1.505, 1.52]),
                 xp=10, use_ideal_distribution=False,
                 cutoff=0.5e-2,
                 effective_width=2, old_version=False):
        """A y-splitter structure consisting of linear waveguides with an
        branching region optimized for a certain wavelength and specific
        discrete refractive index levels. The procedure specified in [1] is
        used for the calculation of said structure. By default, a trimmed index
        distribution with zero imaginary part is used, however it is also
        possible to use the ideal, continouus but also complex index
        distribution resoluting from the procedure.
        The parameters xp and index_levels can be used for the optimization of
        the optical performance.

        Parameters
        ----------
        x_params : iterable
            spacial dimensions of transversal direction (x_min, x_max, N_x)
        z_params : iterable
            spacial dimensions of longitudinal direction (z_min, z_max, N_z)
        wavelength : float
            self explanatory
        refractive_index_guide : float
            refractive index of waveguide (center)
        refractive_index_medium : float
            refractive index of medium
        half_angle : float
            half the angle between the output waveguides
        branching_region_x : iterable
            the x region considered in the optimization process
        branching_region_z : iterable
            the z region considered in the optimization process
        guide_x_coord : float
            the x coordinate of the structures center
        x_resolution : int
            the x resolution used in the optimization procedure
        z_resolution : int
            the z resolution used in the optimization procedure
        guide_seperation : float
            the distance between the waveguide boundaries at the end of the
            branching region
        width : float
            the width of the linear waveguides
        effective_width : float
            the effective width of the input and output waveguides, ie the region
            around the guides considered for the calculation of the guided
            intensity
        index_levels : np.ndarray
            the refractive indices used trimming procedure
        xp : float
            the x coordinate where the relative phase of the field at the
            input and the output of the branching region is 180°
        use_ideal_distribution : bool
            whether the ideal (complex) distribution in the branching region
            should be used. Defaults to False
        Notes
        -----
            In contrast to the other waveguide objects all spacial parameters
            are to be entered as absolute coordinates.

        References
        ----------
            [1] New Design Method for Low-Loss Y-Branch Waveguides
            Yabu, T. and Geshiro, M. and Sawa, S.
            Journal of Lightwave Technologie,
                Vol. 19, No. 9, Sep 2001
        """

        # ----------------------------------------------------------------------
        self.form                   = "Combined Waveguide"
        self.refractive_index_guide = refractive_index_guide
        self.refractive_index_medium = refractive_index_medium

        # transversal dimensions -----------------------------------------------
        self.x_params               = x_params
        self.branching_region_x     = branching_region_x
        self.guide_x_coord          = guide_x_coord
        self.x_resolution           = x_resolution

        # longitudinal dimensions ----------------------------------------------
        self.z_params               = z_params
        self.branching_region_z     = branching_region_z
        self.z_resolution           = z_resolution

        # additional properties ------------------------------------------------
        self.output_angle           = np.pi / 180 * half_angle
        self.guide_seperation       = guide_seperation
        self.width                  = width
        self.index_levels           = index_levels
        self.xp                     = xp
        self.use_ideal_distribution = use_ideal_distribution
        self.cutoff                 = cutoff
        self.old_version            = old_version

        # x coordinate of junction outputs -------------------------------------
        self.output_x_offset = \
            (z_params[1] - self.branching_region_z[1]) * \
            np.tan(self.output_angle)

        # beam instance for ideal field computation ----------------------------
        self.beam = EigenModes(wavelength=wavelength, waveguide=self)

        # Linear waveguides forming the y-junction ---------------------------------
        self.waveguide_input = LinearWaveguide(
            refractive_index_medium=0,
            refractive_index_guide=refractive_index_guide,
            width=self.width,
            z_start=z_params[0],
            z_end=self.branching_region_z[0],
            guide_x_coord=self.guide_x_coord,
            angle=0,
            coord_mode="absolute",
            effective_width=effective_width)
        self.waveguide_input = [self.waveguide_input]

        self.output_1 = LinearWaveguide(
            refractive_index_guide=refractive_index_guide,
            refractive_index_medium=0,
            width=self.width,
            z_start=self.branching_region_z[1],
            z_end=z_params[1],
            guide_x_coord=self.guide_x_coord + \
                          1/np.cos(self.output_angle) * self.width / 2 + \
                          self.guide_seperation / 2,
            angle=self.output_angle,
            coord_mode="absolute",
            effective_width=effective_width)

        self.output_2 = LinearWaveguide(
            refractive_index_guide=refractive_index_guide,
            refractive_index_medium=0,
            width=self.width,
            z_start=self.branching_region_z[1],
            z_end=z_params[1],
            guide_x_coord=self.guide_x_coord - \
                          1/np.cos(self.output_angle) * self.width / 2 - \
                          self.guide_seperation / 2,
            angle=-self.output_angle,
            coord_mode="absolute",
            effective_width=effective_width)

        self.waveguide_base = WaveguideBase(
            refractive_index_medium=self.refractive_index_medium)

        self.waveguide_additional = [self.output_1, self.output_2]


        x, dx = np.linspace(*self.branching_region_x, self.x_resolution,
                            retstep=True)
        z, dz = np.linspace(*self.branching_region_z, self.z_resolution,
                            retstep=True)

        output_field = self.compute_output_field(x)
        input_field, n_eff = self.compute_input_field(x)

        angles_in = np.angle(input_field)
        angles_out = np.angle(output_field)

        relative_angles = angles_out - angles_in

        angle = np.pi - relative_angles[np.argmin(
            np.abs(x - xp))]
        output_field = output_field * np.exp(1j * angle)


        self.index_calc = IndexCalculator(input_field=input_field, output_field=output_field, x=x, dx=dx, z=z, dz=dz,
                                     index_levels=self.index_levels,
                                     xp=self.xp, use_ideal_distribution= \
                                        use_ideal_distribution,
                                     cutoff=cutoff)

        self.index_calc.find_index_distribution(self.beam.wavenumber,
                                                   n_eff)

    def compute_output_field(self, x):
        """Compute the desired field at the output of the branching region.
        This is done by propagating the base mode of the waveguide in the
        reversed z direction in order to create a clean version of the field.

        Paramters
        ---------
        x : np.ndarray
            the x-coordinates of the field distribution.
            The Result is interpolated.
        """

        logging.info("Starting computation of ideal field at junction output")
        if self.old_version:
            # temporal grid for propagation ----------------------------------------
            temp_grid = ComputationalGrid(
                x_params=(
                self.guide_x_coord - self.output_x_offset - 5 * self.width,
                self.guide_x_coord + self.output_x_offset + 5 * self.width,
                self.x_params[2]),
                z_params=(self.branching_region_z[1], self.z_params[1], self.z_params[2]),
                k_max=0.00)

            # waveguides for propagation in inverse z direction --------------------
            waveguide_base = WaveguideBase(
                refractive_index_medium=self.refractive_index_medium)

            waveguide_output_1 = LinearWaveguide(
                width=self.width,
                refractive_index_guide=self.refractive_index_guide,
                refractive_index_medium=0,
                guide_x_coord=self.guide_x_coord + \
                              self.output_x_offset + \
                              np.cos(self.output_angle) * self.width / 2 + \
                              self.guide_seperation / 2,
                z_end=self.z_params[1], angle=-self.output_angle,
                z_start=self.branching_region_z[1],
                coord_mode="absolute")

            waveguide_output_2 = LinearWaveguide(
                width=self.width,
                refractive_index_guide=self.refractive_index_guide,
                refractive_index_medium=0,
                guide_x_coord=self.guide_x_coord - \
                              self.output_x_offset - \
                              np.cos(self.output_angle) * self.width / 2 - \
                              self.guide_seperation / 2,
                z_end=self.z_params[1],
                z_start=self.branching_region_z[1],
                angle=self.output_angle,
                coord_mode="absolute")

            guide = CombinedWaveguide(
                waveguide_base=waveguide_base, waveguide_additional=[],
                waveguide_input=[waveguide_output_1, waveguide_output_2])

            # beam for reversed propagation ----------------------------------------
            beam = EigenModes(wavelength=self.beam.wavelength, waveguide=guide,
                              rel_strength=[0.5, 0.5],
                              phase_shift=[0, 0])

            # propagate in reversed direction --------------------------------------
            solver = Chung1990Solver(computational_grid=temp_grid,
                                     waveguide=guide, beam=beam)
            solver.run_simulation()

            # conjugate and interpolate the electric field at the output -----------
            e_field = np.array(solver.observer.efield_profile, dtype=np.complex128)
            output_field =  np.conj(np.interp(x, temp_grid.x, e_field[-1]))

        else:
            N_x = int(2 * (self.guide_x_coord + self.output_x_offset + 3 * self.width) / 0.1)

            temp_grid = ComputationalGrid(
                x_params=(
                    self.guide_x_coord - self.output_x_offset - 3 * self.width,
                    self.guide_x_coord + self.output_x_offset + 3 * self.width,
                    N_x),
                z_params=(self.branching_region_z[1], self.branching_region_z[1]+1, 1),
                k_max=0.00)

            guide = CombinedWaveguide(self.waveguide_base, [self.output_1, self.output_2], waveguide_additional=[])

            beam = EigenModes(wavelength=self.beam.wavelength, waveguide=guide)
            E1 = beam.calc_initial_field(temp_grid)

            output_field = np.interp(x, temp_grid.x, E1)



        return output_field

    def compute_input_field(self, x):

        temp_grid = ComputationalGrid(
            x_params=(
                self.guide_x_coord - self.output_x_offset - 5 * self.width,
                self.guide_x_coord + self.output_x_offset + 5 * self.width,
                self.x_params[2]),
            z_params=(0, 1, 1),
            k_max=0.00)
        E0 = self.beam.calc_initial_field(temp_grid)
        input_field = np.interp(x, temp_grid.x, E0)
        return input_field, temp_grid.n_eff

    def write_waveguide(self, computational_grid):
        """Write the waveguide structure into the grid and calculate the index
        distribution in the branching region

        Parameters
        ----------
        computational_grid : ComputationalGrid
        """

        super().write_waveguide(computational_grid)
        self.index_calc.write_waveguide(computational_grid)


