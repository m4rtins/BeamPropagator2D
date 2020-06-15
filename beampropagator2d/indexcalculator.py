import numpy as np


class IndexCalculator():

    def __init__(self, input_field, output_field, x, dx, z, dz,
                 index_levels, xp, cutoff=0.5e-1, use_ideal_distribution=False):
        """
        Implements the functionality described in [1] for the
        determination of a refrective index profile that creates a
        electric field distribution. That distribution is trimmed in the
        process and its imaginary part disregarded.

        Paramters
        ---------
        input_field : np.ndarray
            array containing the electric field distribution at the
            input of the branching region
        output_field : np.ndarray
            array containing the electric field distribution at the
            output of the branching region
        x : np.ndarray
            x coordinates of the branching region
        dx : float
            spacing of x coordinates
        z : np.ndarray
            z coordinates of the branching region
        dz : float
            spacing of z coordinates
        index_levels : np.ndarray
            array of the refractive indices used in the trimming
            procedure
        xp : float
            coordinate where the relative phase of the input field and the
            output field
        cutoff : float
            field strength cutoff value

        Notes
        -----

        References
        ----------

            [1] New Design Method for Low-Loss Y-Branch Waveguides
            Yabu, T. and Geshiro, M. and Sawa, S.
            Journal of Lightwave Technologie,
                Vol. 19, No. 9, Sep 2001
        """

        self.x, self.dx = x, dx
        self.z, self.dz = z, dz

        self.input_field = input_field
        self.output_field = output_field

        self.output_field *= np.sqrt(np.trapz(np.abs(input_field) ** 2, dx=self.dx)) / np.sqrt(np.trapz(np.abs(output_field) ** 2, dx=self.dx))
        # compute the relative phase angles of input field and output field


        self.index_levels = index_levels
        self.cutoff = cutoff

        self.use_ideal_distribution = use_ideal_distribution

        self.trimmed_index_distribution = np.zeros((len(self.x) - 2, len(self.z) - 2))
        self.ideal_index_distribution   = np.zeros(self.trimmed_index_distribution.shape, dtype=np.complex64)
        self.interpolated_field         = np.zeros((len(self.x), len(self.z)))


    def _interpol(self):
        """Interpolate the field in the branching region and normalize the field
        at each step"""
        # interpolation parameter
        t = (self.z - self.z[0]) / (self.z[-1] - self.z[0])
        self.interpolated_field = np.array([self.input_field * (1-i) +
                                            self.output_field * i for i in t])
        # normalize the interpolated field
        self.intensities = \
                np.array([[np.sqrt(np.trapz(np.abs(i) ** 2, dx=self.dx))
                           for i in
                           self.interpolated_field]]).transpose()

        self.interpolated_field = self.interpolated_field / \
            self.intensities * self.intensities[0]
        # bring the field into the shape (x, z)
        self.interpolated_field = self.interpolated_field.transpose()

    def _calculate_derivatives(self, field, dx, dz):
        """Calculate the necessary derivates at each gridpoint using finite
        differences.

        Paramters
        ---------
        field : np.ndarray
            the field distribution as an (x times z) array
        dx : float
            spacing of the x coordinates
        dz : float
            spacing of the z coordinates

        Returns
        -------
        The derivatives at each (x-z)-gridpoint.

        Notes
        -----
        Since the field is interpolated in the z direction, it is possible to
        compute a (mostly) analytical derivative in that direction. It is however
        necessary to pay intention to the factor b(z) used no normalize the
        field intensity at each z step. This saves a bit of computational
        efford.
        """
        dev_b_z = (1 / self.intensities[1:-1] - 1 / self.intensities[:-2]) / dz
        dev_b_z_2 = ( 1 / self.intensities[2:] + 1 / self.intensities[:-2] -
                     2 / self.intensities[1:-1])/dz**2
        fac = 1 / (self.z[-1]-self.z[0]) * (self.output_field[1:-1] -
                                            self.input_field[1:-1])
        dev_z = np.transpose(fac * 1 / self.intensities[1:-1]) + \
                field[1:-1, 1:-1] * np.transpose(dev_b_z)
        dev_z_2 = np.transpose(2 * fac * dev_b_z) + \
                field[1:-1, 1:-1] * np.transpose(dev_b_z_2)
        dev_x_2 = (field[2:, 1:-1] +
                   field[:-2, 1:-1] - 2 * field[1:-1, 1:-1])/dx**2
        return dev_z, dev_z_2, dev_x_2

    def _calculate_index_distribution(self, wavenumber, n_eff):
        """Calculate the ideal index distribution following eq. (7)
        from [1] and return only the real part

        Paramters
        ---------
        wavenumber : float
        n_eff : np.ndarray or float
            the reference refractive indices at each z location

        Returns
        -------
        The Ideal index distribution
        """

        field = self.interpolated_field
        dev_z, dev_z_2, dev_x_2 = \
                self._calculate_derivatives(field, self.dx, self.dz)
        index = np.sqrt((dev_x_2 + dev_z_2
                        - 2j * wavenumber * n_eff * dev_z
                         - wavenumber**2 * n_eff**2 * field[1:-1, 1:-1])
                        * -1 / wavenumber**2 / field[1:-1, 1:-1])
        index[np.abs(field[1:-1, 1:-1]) < self.cutoff] = self.index_levels[0]
        return index

    def _error_function(self):
        """Compute the errors made with the trimming procedure at each
        propagation step following eq (9) from [1]
        """
        ideal = np.real(self.ideal_index_distribution)
        trimmed = self.trimmed_index_distribution
        errors = np.trapz(np.abs((ideal - trimmed))**2 *
                          np.abs(self.interpolated_field[1:-1, 1:-1]),
                          dx=self.dx, axis=0)
        return errors

    def find_index_distribution(self, wavenumber, n_eff):
        """Comute a trimmed index distribution that leads to the desired
        interpolated field as close as possible minimizing eq(9) from [1]

        Paramters
        ---------
        wavenumber : float
        n_eff : float
            reference refractive index
        Returns
        -------
        Trimmed index distribution
        """

        # make the linear field interpolation ----------------------------------
        self._interpol()
        # calculate the ideal index distribution -------------------------------
        self.ideal_index_distribution = \
                self._calculate_index_distribution(wavenumber, n_eff)

        if self.use_ideal_distribution:
            return self.ideal_index_distribution

        # prepare trimming process ---------------------------------------------
        level_borders = (self.index_levels[:-1] + self.index_levels[1:]) / 2
        level_borders = np.array([-np.infty, *level_borders, np.infty])
        n_eff = np.ones(len(self.z)-2) * n_eff

        def calculate_distri_and_errors(n_eff):
            # helper function for trimming procedure
            index = np.real(self._calculate_index_distribution(wavenumber, n_eff))
            self.trimmed_index_distribution = np.piecewise(index,
                                            [np.logical_and(
                                                index < level_borders[i+1],
                                                index >= level_borders[i])
                                        for i in range(len(level_borders) - 1)],
                                        self.index_levels)
            errors = self._error_function()

            return errors

        # minimize the error between trimmed and indeal distribution
        # through a brute-force method -----------------------------------------
        print(np.amin(np.real(self.ideal_index_distribution)))
        print(np.amax(np.real(self.ideal_index_distribution)))

        ref_indices = np.linspace(np.amin(np.real(self.ideal_index_distribution))-0.03, np.amax(np.real(self.ideal_index_distribution)) + 0.03, 1000)

#        ref_indices = np.linspace(self.index_levels[0]-0.03, self.index_levels[-1] + 0.03, 1000)
        y = []
        for i in ref_indices:
            errors = calculate_distri_and_errors(i)
            y.append(errors)
        y = np.array(y).transpose()
        n_eff = []
        for i in y:
            n_eff.append(ref_indices[np.argmin(i)])
        n_eff = np.array(n_eff)

        calculate_distri_and_errors(n_eff)

        return self.trimmed_index_distribution

    def dump_data(self, fName):
        fName += "_optYjunc"
        np.save(fName + "_branching_x", self.x)
        np.save(fName + "_branching_z", self.z)
        np.save(fName + "_interp_field", self.interpolated_field)
        np.save(fName + "_ideal_index_distribution", self.ideal_index_distribution)
        if not self.use_ideal_distribution:
            np.save(fName + "_ideal_index_distribution",
                    self.ideal_index_distribution)

    def read_data(self, fName):
        fName += "_optYjunc"
        self.x = np.read(fName + "_branching_x.npy")
        self.z = np.read(fName + "_branching_z.npy")
        self.interpolated_field = np.read(fName + "_interp_field.npy")
        self.ideal_index_distribution = np.read(fName + "_ideal_index_distribution.npy")
        if not self.use_ideal_distribution:
            self.use_ideal_distribution = np.read(fName + "_ideal_index_distribution.npy")

    def write_waveguide(self, computational_grid):
        """write the index distribution in a grid of arbitrary resolution
        according to the grid and distribution coordinates.

        Paramters
        ---------
        computational_grid : ComputationalGrid
            the grid the structure is to be written to

        Notes
        -----
            This Code does employ a sort of interpolation scheme that is custom
            made because the trimmed structure of the grid should be preserved
        """
        # TODO: Limit the scope of the following code to the region of interest
        xv, zv = computational_grid.xz_mesh
        x = computational_grid.x
        # a boolean array specifying the nearest x coordinate of the distro
        # for each x coordinate in x
        condlist = [np.logical_and(x <= i + self.dx/1.9, x > i - self.dx/1.9)
                                                     for i in self.x[1:-1]]
        z = self.z
        dz = self.dz
        dx = self.dx
        n_xz = computational_grid.n_xz
        mask = computational_grid.waveguide_mask
        index_distro = self.ideal_index_distribution if self.use_ideal_distribution else \
                self.trimmed_index_distribution

        for zdex, z in enumerate(computational_grid.z):

            if self.z[0] <= z <= self.z[-1]:
                # calculate the index of the nearest z coordinate
                index = np.array(np.argmin(np.abs(self.z[1:-1] - z)))

                n_xz[:, zdex] = np.piecewise(n_xz[:, zdex], condlist,
                    np.append(index_distro[:,index],
                              lambda x: x))
                mask[:, zdex] = np.piecewise(mask[:, zdex], np.logical_and(x <= self.x[-1],
                                                                           x >= self.x[0]),
                                             [1, 0])

