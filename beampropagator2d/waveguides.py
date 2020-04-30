# -------------------------------------------

# Created by:               jasper
# as part of the project:   Bachelorarbeit
# Date:                     11/6/19

# --------------------------------------------

import numpy as np
from scipy.special import erf
from typing import Union, Iterable


class ComputationalGrid:
    """The linear spaced grid in x and z direction consired for
    the beam propagation simulation

    Attributes
    ----------
    x_min, x_max, N_x, z_min, z_max, N_z : float
        The endpoints of the spatial region considered in the beam propagation
        and the respective number of grid points for each dimenion.
    x, z : np.ndarray
        Arrays containing the x and z coordinates of the grid points. The coordinates
        are linearly spaced.
    dx, dz : float
        The gridspacings in each direction
    xz_mesh : np.ndarray
        A meshgrid generated from x. z
    waveguide_mask : np.ndarray
        A (N_x x N_z)-sized array used to store a mask of waveguide structures in
        the refractive index distribution. This can be useful to restrain calculations
        to only those parts of the grid overlayed by waveguides
    boundary_mask : np.ndarray
        A (N_x x N_z)-sized array marking the absorbing regions of the grid
    n_eff : float
        The effective refractive index used in the paraxial approximation
    k_max : float
        The max. imaginary part of the absorging layer at the boundaries.
    delta : float
        The width of the absorbing layer and the boundaries
    k : np.ndarray
        A N_x-sized array containing the imaginary part of the index distribution.
        At the boundaries, the imaginary part has the shape of a Gaussian-Error Funktion.
    """

    def __init__(self, x_params: tuple = (0, 60, 1000), z_params: tuple = (0, 500, 4000),
                 k_max: float = 0.05, delta: float = 5):
        """Initalizes a ComputationalGrid storing the properties of the refractive index distribution.

        A empty (constant zero) grid of the size (N_x x N_z)  and arrays containing
        the coordinates of each grid point are intialized, as well as mask-arrays
        used to identify reagions of the region with waveguide structures and the
        absorbing regions at the boundaries. The absorbing regions are written.

        Parameters
        ----------
        x_params : tuple
            The dimensions and grid-point number for the transversal x-direction
            (x_min, x_max, N_x)
        z_params : tuple
            The dimensions and grid-point number for the transversal z-direction
            (z_min, z_max, N_z)
        k_max : float
            Amplitude of the absorbing layer
        delta : float
            Width of the absorbing layer
        """
        self.x_min, self.x_max, self.N_x = x_params
        self.z_min, self.z_max, self.N_z = z_params
        self.x, self.dx = np.linspace(*x_params, endpoint=False, retstep=True)
        self.z, self.dz = np.linspace(*z_params, endpoint=False, retstep=True)
        self.xz_mesh = np.meshgrid(self.x, self.z, indexing='ij')
        self.n_xz = np.zeros((self.N_x, self.N_z), dtype=np.complex64)
        self.waveguide_mask = np.zeros((self.N_x, self.N_z), dtype=int)
        self.boundary_mask = np.zeros(self.N_x, dtype=int)
        self.n_eff = 0

        # boundary-condition shape: (err-func)
        self.k_max = k_max
        self.delta = delta
        self.write_absorbing_boundary()
        print("{} {} {}".format(5 * "#", "ComputationalGrid", (95 - len("ComputationalGrid")) * "#"))
        print("# Created computational grid with the setup: \n \t \t " + \
              self.__str__())

    def __str__(self):
        x_str = "x: min = {} max = {} N = {}, dx = {}".format(
            self.x_min, self.x_max, self.N_x, self.dx)
        z_str = "z: min = {} max = {} N = {}, dz = {}".format(
            self.z_min, self.z_max, self.N_z, self.dz)
        return x_str + "\t" + z_str

    def write_absorbing_boundary(self):
        self.k = np.piecewise(self.x,
                              [self.x < self.delta + self.x_min,
                               np.logical_and(self.x >= self.delta + self.x_min,
                                              self.x <= self.x_max - self.delta),
                               self.x > self.x_max - self.delta],
                              [lambda x: -self.k_max * (erf(
                                  ((x - self.x_min) * 2.2 / self.delta)) - erf(
                                  2.2)),
                               0,
                               lambda x: self.k_max * (-erf(
                                   ((self.x_max - x) * 2.2 / self.delta)) + erf(
                                   2.2))])
        self.boundary_mask[self.k == 0] = 1

    def dump_data(self, filepath):
        """Save data necessary to reconstructed the class-instance in the npy-file format

        Parameters
        ----------
        filepath : str
        """
        filepath += "_index_grid"
        params = np.array([self.n_eff, self.k_max, self.delta])
        np.save(filepath + "_params", params)
        np.save(filepath + "_x", self.x)
        np.save(filepath + "_z", self.z)
        np.save(filepath + "_n_xz", self.n_xz)
        np.save(filepath + "_waveguide_mask", self.waveguide_mask)
        print("# Saved computational grid setup to: {}".format(filepath))

    def read(self, filepath):
        """ Reconstruct a ComputationalGrid-instance from a saved instance

        Parameters
        ----------
        filepath : str
        """
        filepath += "_index_grid"
        print("# Reading a saved computational grid setup from: {}".format(filepath))
        self.x = np.load(filepath + "_x.npy")
        self.z = np.load(filepath + "_z.npy")
        self.n_xz = np.load(filepath + "_n_xz.npy")
        self.waveguide_mask = np.load(filepath + "_n_xz.npy")
        self.n_eff, self.k_max, self.delta = np.load(filepath + "_params.npy")

        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]

        self.x_min, self.x_max, self.N_x = self.x[0], self.x[-1], len(self.x)
        self.z_min, self.z_max, self.N_z = self.z[0], self.z[-1], len(self.z)

        self.xz_mesh = np.meshgrid(self.x, self.z, indexing='ij')
        self.write_absorbing_boundary()

    def dump_str(self, filepath):
        """ Save the important properties of a class-instance as a text-file

        Parameters
        ----------
        filepath
        """
        with open(filepath, 'w') as fStream:
            fStream.write("x_min x_max, N_x z_min z_max N_z k_max delta \n")
            fStream.write("{} {} {} {} {} {} {} {}\n".format(
                self.x_min, self.x_max, self.N_x,
                self.z_min, self.z_max, self.N_z, self.k_max, self.delta))
            fStream.write("{} ".format(self.n_eff))
            np.savetxt(fStream, [self.x], delimiter=" ")
            for j in range(self.N_z):
                np.savetxt(fStream, [self.z[j]], fmt="%.3e", delimiter=" ", newline=" ")
                np.savetxt(fStream, [self.n_xz[:, j]], delimiter=" ")

        print("# Saved computational grid setup to: {}".format(filepath))


class WaveguideBase:
    """Stores the parameters and provides methods for writing an
    empty waveguide structure into a computational grid.
    """

    def __init__(self, rel_index_profile=None,
                 refractive_index_medium=3.3, refractive_index_guide=3.33,
                 coord_mode="relative", effective_width=1):
        """Stores the parameters and provides methods for
        writing a general waveguide structure into a
        computational grid.

        Parameters
        ----------
        rel_index_profile : function, optional
            Function that accepts 2D ndarrays and returns a scalar for each
            element, e.i. a 2D ndarray.
            Should describe the index profile as a function from the
            relative (norm = 1) distance from the guide core.
            Must describe the refractive index of the medium surrounding
            the waveguide. May describe arbitrary guide structures e.g.
            step profiles or continuous profile. If not specified a one
            step profile is used.
        refractive_index_guide : float, optional
            refractive index of the waveguide, used if no rel_index_profile
            is specified. Defaults to n = 3.30.
        refractive_index_medium : float, optional
            refractive index of the medium surrounding the waveguide.
            Used if no rel_index_profile is specified. Defaults to n = 3.33.
        """
        print("{} {} {}".format(5 * "#", self.__class__.__name__, (95 - len(self.__class__.__name__)) * "#"))
        print("# Initalizing waveguide structure")
        if rel_index_profile is None:
            print("# Structure has one step profile with core refractive index {}".format(refractive_index_guide))
            if refractive_index_medium != 0:
                print("# The substrat has a refractive index of {}".format(refractive_index_medium))
        else:
            print("# Structure has a custom refractive index profile")
        # OPTICAL PROPERTIES OF SYSTEM
        self.effective_width = effective_width
        self.write_mask = 1
        self.coord_mode = coord_mode
        self.refractive_index_medium = refractive_index_medium
        self.refractive_index_guide = refractive_index_guide
        # Default function for index profile if not specified at class initialization
        if rel_index_profile is None:
            self.rel_index_profile = \
                lambda x: np.piecewise(x, [np.logical_and(x <= 1, x >= -1), ],
                                       [refractive_index_guide,
                                        refractive_index_medium])
        else:
            self.rel_index_profile = rel_index_profile

        self.profile = ""
        self.form = "Free Space"

    def write_waveguide(self, computational_grid):
        """Write the structure specified in the class instance into a
        computational grid

        Paramters
        ---------
        ComputationalGrid : ComputationalGrid
            The grid the structure is to be written into
        """
        print("# Writing {} into computational grid".format(
            self.__class__.__name__))
        # compute the distances of each grid point from the waveguide core
        distances_from_center = self._compute_distance_from_center(
            computational_grid)

        # and a write a mask for the waveguide used for various purposes
        if self.write_mask != 0:
            computational_grid.waveguide_mask[np.abs(distances_from_center) <= \
                                              self.effective_width] = self.write_mask

        # change only the part of the grid we actually want to write a structure into
        grid = self.rel_index_profile(distances_from_center)
        computational_grid.n_xz[grid != 0] = grid[grid != 0]
        # select the refractive index at "infinity" as the reference refractive index
        # used in the propagation simulation
        computational_grid.n_eff = computational_grid.n_xz[0, 0]
        return computational_grid

    def _compute_distance_from_center(self, computational_grid):
        """Gives the distance from the waveguide core.
        Since this is the case of free space, a grid filled with inf
        is passed to the rel_index_profile function.
        """
        return computational_grid.n_xz + np.infty


class Substrat(WaveguideBase):

    def __init__(self, refractive_index_medium=3.3):
        super().__init__(refractive_index_medium=refractive_index_medium)


class LinearWaveguide(WaveguideBase):
    """Stores the parameters and provides methods for writing a
    linear waveguide structure into a computational grid.
    """

    def __init__(self, rel_index_profile=None, refractive_index_guide=3.33,
                 refractive_index_medium=3.3,
                 width=1, effective_width=1, guide_x_coord=0.5, z_start=0, z_end=1,
                 coord_mode="relative",
                 profile='step profile', angle=0):
        """
        Create a linear waveguide structure with given width and angle with
        the z-direction in a 2D space domain.
        The Waveguide profile may be described by a function of the relative
        displacement from the guide center or by two refractive indices, one for
        the guide and one for the medium.

        Parameters
        ----------
        rel_index_profile : function, optional
            Function that accepts 2D ndarrays and returns a scalar for each
            element, e.i. a 2D ndarray.
            Should describe the index profile as a function from the
            relative (norm = 1) distance from the guide core.
            Must describe the refractive index of the medium surrounding the
            waveguide. May describe arbitrary guide structures e.g. step
            profiles or continuous profile. If not specified a one
            step profile is used.
        refractive_index_guide : float, optional
            refractive index of the waveguide, used if no rel_index_profile is
            specified. Defaults to n = 3.30.
        refractive_index_medium : float, optional
            refractive index of the medium surrounding the waveguide.
            Used if no rel_index_profile is specified. Defaults to n = 3.33.
        guide_x_coord : float
            starting x coordinate of the waveguide.
            Defaults to the structure center.
        width : float
            width of the waveguide in question.
        angle : float
            Angle of the guide with the z-axis. Needs to be in the Interval
            (-pi/4, pi/4), although the limits of the paraxial approximation
            need to be considered.
        z_start : float
            the z-coordinate the waveguide starts at. Defaults to the
            lower boundary of the comp. grid.
        z_end : float
            the z-coordinate the waveguide ends at. Defaults the the
            upper boundary of the comp. grid
        profile : str
            Type of the Waveguide. Possible types are "step profile",
            "continuous profile",
            "custom profile".
        coord_mode : str
            Choose whether the given coordinates for the waveguide structures
            should be handled as absolute lengths or relative lengths
        """

        super().__init__(rel_index_profile=rel_index_profile,
                         refractive_index_medium=refractive_index_medium,
                         refractive_index_guide=refractive_index_guide,
                         coord_mode=coord_mode, effective_width=effective_width)

        # structure identifiers
        self.form = "Linear Wave Guide"
        self.profile = profile

        # spacial structure parameters
        self.width = width
        self.guide_x_coord = guide_x_coord
        self.z_region = np.array([z_start, z_end])

        # the paraxial limit should not be overstepped
        if angle < -np.pi / 4 or angle > np.pi / 4:
            print(
                "# WARINING: Invalid angle alpha = {} of waveguide (to big). Defaulted to alpha = 0.".format(
                    angle))
            self.angle = 0
        else:
            self.angle = angle

    def _compute_distance_from_center(self, computational_grid):
        """Compute the !closest! distance to the Waveguide core for linear waveguides with arbitrary angle to the
        z-axis.

        Parameters
        ----------
        computational_grid : ComputationalGrid
            Needed to provide grid-structure of spacedomain considered.

        Returns
        -------
        distances : np.ndarray
            2D-ndarray containing the distances from the guide core for everey gridpoint in the space domain.

        """

        xv, zv = computational_grid.xz_mesh

        # calculate the absolute coordinates of the structure if relative coords are used
        if self.coord_mode == "relative":
            z_region = self.z_region * computational_grid.z_max + \
                       (1 - self.z_region) * computational_grid.z_min
            x0 = self.guide_x_coord * computational_grid.x_max + \
                 (1 - self.guide_x_coord) * computational_grid.x_min
        elif self.coord_mode == "absolute":
            z_region = self.z_region
            x0 = self.guide_x_coord
        else:
            exit()

        # calculate the x coordinates of the waveguide center for each z step
        x0 = np.piecewise(zv, [
            np.logical_and(zv >= z_region[0], zv <= z_region[1]), ],
                          [lambda zv: x0 + np.tan(self.angle) * (
                                  zv - z_region[0]), np.infty])
        # calculate the shortest distance for each gridpoint from the center of the waveguide
        # relative to the width of the waveguide
        return 2 * (xv - x0) / self.width * np.cos(self.angle)


class Triangle(WaveguideBase):
    """ Define a triangle shaped waveguide with constant refractive index. """

    def __init__(self, refractive_index_guide=3.33,
                 refractive_index_medium=3.3,
                 coord_mode="relative",
                 p1=(0.2, 0.2), p2=(0.2, 0.6), p3=(0.4, 0.4)):
        """ Define a triangle shaped waveguide with constant refractive index by
        specifying the corners of the triange, the refractive index of guide and
        medium and whether the coordinates should be treated as relative or
        absolute.

        Parameters
        ---------
        refractive_index_guide : float
            refractive index of the guide material
        refractive_index_medium : float
            refractive index of the cladding material
        coord_mode : str
            whether the coordinates should be treated as relative or absolute
        p1 : tuple of float
            coordinates of first corner
        p2 : tuple of float
            coordinates of second corner
        p3 : tuple of float
            coordinates of third corner
        """
        super().__init__(refractive_index_medium=refractive_index_medium, refractive_index_guide=refractive_index_guide,
                         coord_mode=coord_mode)
        self.write_mask = 0
        self.profile = "step profile"
        self.x_coords = np.array([p1[0], p2[0], p3[0]])
        self.z_coords = np.array([p1[1], p2[1], p3[1]])

        # create a box around the triangle to save a bit of calculation time when
        # writing the waveguide

        self.x_min = np.amin(self.x_coords)
        self.z_min = np.amin(self.z_coords)

        self.x_max = np.amax(self.x_coords)
        self.z_max = np.amax(self.z_coords)

    def _compute_distance_from_center(self, computational_grid):
        """ check for all grid points if they are inside the triangle, return 0 if
        and infinity if not. A box surrounding the whole triangle is used to
        save computation time.

        Parameters
        ----------
        computational_grid : ComputationalGrid
            self explanatoy

        Notes
        -----

        The computation uses the following condition: If one consideres the vectors
        connecting a grid point with all the triangle vertices and the angle these
        vectors enclose, these vectors will only ever enclse and combined angle of
        360° if the point lies inside the triangle.
        Hence this routine computes the three vectors for each gridpoint (inside
        the box), the corresponding angles and checks whether the condition is
        fullfilled.
        """

        if self.coord_mode == 'relative':
            self.x_coords = self.x_coords * computational_grid.x_max + \
                            (1 - self.x_coords) * computational_grid.x_min
            self.z_coords = self.z_coords * computational_grid.z_max + \
                            (1 - self.z_coords) * computational_grid.z_min

        xv_temp, zv_temp = computational_grid.xz_mesh
        # write the mask of the box the triangle is placed inside
        mask = np.logical_and(xv_temp <= self.x_max, xv_temp >= self.x_min) * np.logical_and(zv_temp <= self.z_max,
                                                                                             zv_temp >= self.z_min)

        xv = xv_temp[mask]
        zv = zv_temp[mask]
        # compute the vectors connection the points with the vertices
        x_distances = np.array([i - xv for i in self.x_coords])
        z_distances = np.array([i - zv for i in self.z_coords])
        # compute the products of the coordiantes
        x_prods_1 = x_distances[0] * x_distances[1]
        x_prods_2 = x_distances[1] * x_distances[2]
        x_prods_3 = x_distances[2] * x_distances[0]

        z_prods_1 = z_distances[0] * z_distances[1]
        z_prods_2 = z_distances[1] * z_distances[2]
        z_prods_3 = z_distances[2] * z_distances[0]

        prods_1 = x_prods_1 + z_prods_1
        prods_2 = x_prods_2 + z_prods_2
        prods_3 = x_prods_3 + z_prods_3

        # compute the norms of the vectors and their products
        x = np.array([i ** 2 for i in x_distances])
        z = np.array([i ** 2 for i in z_distances])
        norms = np.sqrt(x + z)

        norms_1 = norms[0] * norms[1]
        norms_2 = norms[1] * norms[2]
        norms_3 = norms[2] * norms[0]

        # compute angles and their sum
        angles_1 = np.arccos(prods_1 / norms_1)
        angles_2 = np.arccos(prods_2 / norms_2)
        angles_3 = np.arccos(prods_3 / norms_3)

        angle_sum = angles_1 + angles_2 + angles_3

        # write a boolean mask for the grid
        in_triangle_temp = np.where(np.abs(angle_sum - 2 * np.pi) < 1e-2, 0, np.infty)

        in_triangle = np.infty * np.ones(xv_temp.shape)
        # reincorporate that mask into the whole picture
        in_triangle[mask] = in_triangle_temp

        return in_triangle


class BendedWaveguide(WaveguideBase):

    def __init__(self, rel_index_profile=None, refractive_index_guide=3.33,
                 refractive_index_medium=3.3,
                 width=1, profile='step profile', x_start=0.5, x_end=0.65,
                 z_start=0, z_end=1, bend_start=.25,
                 bend_end=.75, coord_mode="relative"):
        """
        Create a bended waveguide structure with given width and endpoints of the bended region
        in a 2D space domain. The Waveguide profile may be described by a function of the
        relative displacement from the guide center or by two refractive indices, one for
        the guide and one for the medium.

        Parameters
        ----------
        rel_index_profile : function, optional
            Function that accepts 2D ndarrays and returns a scalar for each element, e.i.
            a 2D ndarray. Should describe the index profile as a function from the
            relative (norm = 1) distance from the guide core. Must describe the refractive
            index of the medium surrounding the waveguide. May describe arbitrary guide
            structures e.g. step profiles or continuous profile. If not specified a one
            step profile is used.
        refractive_index_guide : float, optional
            refractive index of the waveguide, used if no rel_index_profile is specified.
            Defaults to n = 3.30.
        refractive_index_medium : float, optional
            refractive index of the medium surrounding the waveguide.
            Used if no rel_index_profile is specified. Defaults to n = 3.33.
        x_start : float
            starting x coordinate of the waveguide. Defaults to the structure center.
        x_end : float
            ending x coordinate of the waveguide. Defaults .65 of the x direction.
        bend_start : float
            starting z coordinate of the bended region
        bend_end : float
            ending z coordinate of the bended region
        width : float width of the waveguide in question.
        profile : str
            Type of the Waveguide. Possible types are "step profile", "continuous profile",
            "custom profile".
        coord_mode : str
            Choose whether the given coordinates for the waveguide structures should be handled as
            absolute lengths or relative lengths
        """

        super().__init__(rel_index_profile=rel_index_profile,
                         refractive_index_medium=refractive_index_medium,
                         coord_mode=coord_mode)
        # structure identifiers
        self.form = "BendedWaveguide"
        self.profile = profile
        # spacial structure parameters
        self.x_start = x_start
        self.x_end = x_end
        self.refractive_index_guide = refractive_index_guide
        self.width = width
        self.bended_region = np.array([bend_start, bend_end])
        self.z_start = z_start
        self.z_end = z_end

    def form_func(self, z, z_bended_region, max):
        """Computes a scaled cosine corresponding to the center of the bended region of the waveguide

        Parameters
        ----------
        z : np.ndarray
            space grid points the function is evaluated upon
        z_bended_region : np.arraylike
            the z region the bend is confined to
        max : float
            distance of waveguide input and output in x direction

        Returns
        -------
        np.ndarray
            new x coordinates for waveguide center
        """
        z = (z - z_bended_region[0]) / (z_bended_region[1] - z_bended_region[0])
        max = 1 / 2 * max

        return (1 - np.cos(np.pi * z)) * max

    def form_func_dev(self, z, z_bended_region, x_max):
        """Compute the tangent of the waveguide structure at each z step.
        This is used to scale the bended region to approximately correct the
        thinning of the structure due to the calculation of the distance only
        along the x - direction

        Parameters
        ----------
        z : np.ndarray
            gridpoint coordinates in z direction
        z_bended_region : np.ndarray
            the boundaries of the bended region to correctly scale the cosine describing the bend in z direction
        x_max : float
            the maximal offset of the bend in x direction. Used to scale the cosine accordingly

        Returns
        -------
        The tangent of the waveguide for each z coordinate
        """
        max = 1 / 2 * x_max
        return np.piecewise(z, [
            np.logical_and(z >= z_bended_region[0], z <= z_bended_region[1]), ],
                            [lambda z: 1 + np.sin(
                                np.pi * (z - z_bended_region[0]) / (
                                        z_bended_region[1] - z_bended_region[
                                    0])) * np.pi * max / (
                                               z_bended_region[1] -
                                               z_bended_region[0]),
                             lambda z: 1])

    def _compute_distance_from_center(self, computational_grid):
        """Compute the linear distance to the Waveguide core for a bended waveguide.
        The derivative of the bend is used to approximate the actual closest
        distance (in order to handle thinning of the guide in a bended region)

        Parameters
        ----------
        computational_grid : ComputationalGrid
            Needed to provide grid-structure of spacedomain considered.

        Returns
        -------
        distances : np.ndarray
            2D-ndarray containing the distances from the guide core for every gridpoint in the space domain.

        """

        # Determine the region in z direction the bend is occupying

        xv, zv = computational_grid.xz_mesh

        # turn the coordinates relative if necessary
        if self.coord_mode == "relative":
            x_in = self.x_start * computational_grid.x_max + \
                   (1 - self.x_start) * computational_grid.x_min
            x_out = self.x_end * computational_grid.x_max + \
                    (1 - self.x_end) * computational_grid.x_min

            z_bended_region = self.bended_region * computational_grid.z_max + \
                              (1 - self.bended_region) * computational_grid.z_min

            z_start = self.z_start * computational_grid.z_max + \
                      (1 - self.z_start) * computational_grid.z_min
            z_end = self.z_end * computational_grid.z_max + \
                    (1 - self.z_end) * computational_grid.z_min
        else:
            x_in = self.x_start
            x_out = self.x_end
            z_bended_region = self.bended_region
            z_start = self.z_start
            z_end = self.z_end

        # compute the waveguide center along the z direction

        x = np.piecewise(zv,
                         [np.logical_or(zv < z_start, zv > z_end),
                          np.logical_and(zv < z_bended_region[0],
                                         zv >= z_start),
                          np.logical_and(zv >= z_bended_region[0],
                                         zv <= z_bended_region[1]),
                          np.logical_and(zv > z_bended_region[1], zv <= z_end)],
                         [np.infty,
                          x_in,
                          lambda zv: x_in + self.form_func(zv, z_bended_region,
                                                           x_out - x_in),
                          x_out])
        # compute and scale the distance from the waveguide center relative to the width of the waveguide and the
        # bend of the waveguide (we want to know the closest distance, not the horizontal distance
        return 2 * (xv - x) / self.width / self.form_func_dev(zv,
                                                              z_bended_region,
                                                              np.abs(
                                                                  x_out - x_in))


class CombinedWaveguide:
    """ParentClass any structure combined from a unrestrained count of waveguides. Provides all the Methods associated with
    waveguide - class instances.
    """

    def __init__(self, waveguide_base: WaveguideBase,
                 waveguide_input: Iterable[WaveguideBase],
                 waveguide_additional: Iterable[WaveguideBase]):
        """Combine Several Waveguides to a new waveguide object.

        The Waveguides listed in waveguide_input will be considered for the calc.
        of guided modes. IMPORTANT: In order to prevent waveguide structures from
        overwriting each other, the refractive index of the medium must be zero
        for all waveguides listed in waveguide_input and waveguide_additional.

        Parameters
        ----------
        waveguide_base : WaveguideBase
            A Waveguide object that is supposed to describe the background, i.e.
            the substrat of the system.
        waveguide_input : Iterable[WaveguideBase]
            Waveguide-objects used as Inputs. These are the ones considered for
            the calculation of guided modes
        waveguide_additional : Iterable[WaveguideBase]
            Additional waveguide structures that should not be considered for the
            calculation of guided modes
        """
        self.waveguide_base = waveguide_base
        self.waveguide_input = waveguide_input
        self.waveguide_additional = waveguide_additional
        self.form = "Combined Waveguide"

    def write_waveguide(self, computational_grid):
        self.waveguide_base.write_waveguide(computational_grid)
        for i in self.waveguide_input:
            i.write_waveguide(computational_grid)
        for i in self.waveguide_additional:
            i.write_waveguide(computational_grid)


class Coupler(CombinedWaveguide):
    """ A rectangular  Waveguide-Coupler with a arbitrary count of inputs and outputs."""

    def __init__(self, input_guides=2, output_guides=8, in_width=1, out_width=None,
                 refractive_index_guide=1.555,
                 refractive_index_medium=1.537, coupler_length=20,
                 coupler_width=20, coupler_x_coord=20,
                 coupler_z_coord=20, input_spacing=5, output_spacing=.5,
                 angle_scale=0.5):
        """
        Parameters
        ----------
        input_guides : int
            number of guides used as input into the coupler
        output_guides : int
            number of guides used as output from the coupler
        width : float
            width of the input and output guides
        refractive_index_guide : float
            refractive index of the entire guide structure (input, output and coupler)
        refractive_index_medium : float
            refractive index of the medium
        coupler_length : float
            abs. length of the coupler
        coupler_width : float
            abs. width of the coupler
        coupler_x_coord : float
            abs. x coordinate of the center of the coupler
        coupler_z_coord : float
            abs. z. coordinate of the start of the coupler
        input_spacing : float
            distance between the input waveguides
        output_spacing : float
            distance between the output waveguides
        angle_scale : float
        """

        self.input_guides = input_guides
        self.output_guides = output_guides
        self.in_width = in_width
        self.out_width = in_width if out_width is None else out_width
        self.refractive_index_guide = refractive_index_guide
        self.refractive_index_medium = refractive_index_medium
        self.coupler_width = coupler_width
        self.coupler_length = coupler_length

        input = []
        additional_guides = []

        input_spacing = (input_spacing + self.in_width)
        output_spacing = (output_spacing + self.out_width)

        # write the refractive index of the medium as a background of the structure
        base = WaveguideBase(refractive_index_medium=refractive_index_medium,
                             refractive_index_guide=0)
        # write the coupler structure

        additional_guides.append(
            LinearWaveguide(coord_mode="absolute",
                            guide_x_coord=coupler_x_coord,
                            z_start=coupler_z_coord,
                            z_end=coupler_z_coord + coupler_length,
                            width=coupler_width,
                            refractive_index_guide=refractive_index_guide,
                            refractive_index_medium=0))
        additional_guides[0].effective_width = 1
        # write each input waveguide
        for i in range(input_guides):
            input.append(
                LinearWaveguide(coord_mode="absolute",
                                guide_x_coord=coupler_x_coord + i * input_spacing / 2 - (
                                        input_guides - 1 - i) * input_spacing / 2,
                                z_start=0, z_end=coupler_z_coord,
                                width=self.in_width,
                                refractive_index_guide=refractive_index_guide,
                                refractive_index_medium=0))
        # write each output waveguide
        for i in range(output_guides):
            additional_guides.append(
                LinearWaveguide(
                    angle=angle_scale * (i - (output_guides - 1) / 2) / 30,
                    coord_mode="absolute",
                    guide_x_coord=coupler_x_coord + i * output_spacing / 2 - (
                            output_guides - 1 - i) * output_spacing / 2,
                    z_start=coupler_z_coord + coupler_length,
                    z_end=10000, width=self.out_width,
                    refractive_index_guide=refractive_index_guide,
                    refractive_index_medium=0))

        super().__init__(base, input, additional_guides)

    def __str__(self):
        return "coupler_{}_in_{}_out_{}_width_{}_ndiff".format(
            self.input_guides, self.output_guides, self.width,
            self.refractive_index_guide - self.refractive_index_medium)


# a quick and dirty implementation of a y - shaped waveguide splitter
class YSplitter(CombinedWaveguide):

    def __init__(self):
        a = BendedWaveguide(x_end=0.55, )
        b = BendedWaveguide(x_end=0.45, refractive_index_medium=0)
        super().__init__([a, b])
