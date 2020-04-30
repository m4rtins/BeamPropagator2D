import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .beampropagater import *
#from .indexcalculator import *
#from .optimizedYjunction import *


class Plotter:
    """Class that handles all plotting of propagation data"""

    def __init__(self, filepath=None, use_tex=True):
        self.filepath = "test" if filepath is None else filepath
        if use_tex:
            self.x_label = r"Transversal direction $x$ in $\si{\micro\meter}$"
            self.z_label = r"Propagation direction $z$ in $\si{\micro\meter}$"
        else:
            self.x_label = r"Transversal direction $x$"
            self.z_label = r"Propagation direction $z$"
        self.use_tex = use_tex
        if use_tex:
            self.tex_setup()

    def tex_setup(self):

        import matplotlib as mpl

        mpl.use('pgf')

        pgf_with_latex = {  # setup matplotlib to use latex for output

            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex

            "text.usetex": True,  # use LaTeX to write all text

            "font.family": "serif",

            "font.serif": [],  # blank entries should cause plots to inherit fonts from the document

            "font.sans-serif": [],

            "font.monospace": [],

            "axes.labelsize": 8,  # LaTeX default is 10pt font.

            "font.size": 8,

            "legend.fontsize": 6,  # Make the legend/label fonts a little smaller

            "xtick.labelsize": 8,

            "ytick.labelsize": 8,

            "axes.titlesize":8,

            "figure.figsize": self.figsize(1,2),  # default fig size of 0.9 textwidth

            "pgf.preamble": [

                r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)

                r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
                
                r"\usepackage{siunitx}"

            ]

        }

        mpl.rcParams.update(pgf_with_latex)


    def backend(self, filename):

        if self.use_tex:
            self.savefig(filename)
        else:
            plt.show()

    def newfig(self, nrows=1, ncols=1, width=1, ratio=2, gridspec_kw=None):

        #plt.clf()
    

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=self.figsize(width, ratio), gridspec_kw=gridspec_kw)

        return fig, ax

    def savefig(self, filename):
        plt.savefig('{}.pgf'.format(filename), dpi=300, #bbox_inches='tight', pad_inches=1)
                )
        plt.savefig('{}.pdf'.format(filename), dpi=300, #bbox_inches='tight',pad_inches=1
                )

    def figsize(self, scale, ratio):
        fig_width_pt = 418.25555 # Get this from LaTeX using \the\textwidth

        inches_per_pt = 1.0 / 72.27  # Convert pt to inch

        golden_mean = (np.sqrt(5.0) - 1.0) /ratio # Aesthetic ratio (you could change this)

        fig_width = fig_width_pt * inches_per_pt * scale  # width in inches

        fig_height = fig_width * golden_mean  # height in inches

        fig_size = [fig_width, fig_height]

        return fig_size

    def _plot_mode_loss_mismath(self, z, lm):
        plt.plot(z, lm)
        plt.show()

    def _plot_power_attenuation(self, z, alpha):
        fig,ax = plt.subplots()

        ax.plot(z, alpha)

        ax.set_xlabel(r"Propagation direction $z$ in $\mu m$")
        ax.set_ylabel(r"Power attenuation $\frac{P_0}{P_z}$ in $db$")

        ax.set_title(r"Relative power transmitted by waveguide")
        ax.grid(which="angle", linestyle='dashed')
        ax.grid(which='minor', linestyle='dotted')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        plt.show()

    def plot_index_distribution(self, obj):
        if isinstance(obj, BeamPropagator2D):
            computational_grid = obj.computational_grid
        elif isinstance(obj, ComputationalGrid):
            computational_grid = obj

        fig, ax = self.newfig(1,1,1,2)

        im = self._plot_index_profile(ax, computational_grid)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.z_label)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(r'Refractive index')
        ax.legend(loc=4)
        self.backend(self.filepath + "_grid_i_d")



    def plot_power_attenuation(self, beam_propagator):

        fig, ax = plt.subplots()

        z = beam_propagator.computational_grid.z
        alpha = beam_propagator.observer.alpha

        ax.plot(z, alpha)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(r'Power attenuation $\frac{P_0}{P_z}$ in $db$')

        self.backend(self.filepath + "_power_atten")


    def plot_crosssection(self, beam_propagator, z):

        fig, ax = self.newfig(1,1,1,2)

        field = beam_propagator.observer.efield_profile[np.logical_and(
            beam_propagator.computational_grid.z < z + beam_propagator.computational_grid.dz,
            beam_propagator.computational_grid.z >= z)].flatten()

        x = beam_propagator.computational_grid.x


        mask = np.where(np.abs(field)**2 > 0.0005)

        x = x[mask]
        field = field[mask]

        ax.plot(x, np.abs(field)**2, label=r'Field intensity')

        ax1 = ax.twinx()
        angle = np.angle(field)
        ax1.plot(x, np.unwrap(angle), color='orange',linestyle='dashed', label=r"Unwrapped phase angle")

        ax.set_ylabel(r"Field intensity")
        ax1.set_ylabel(r"Unwrapped phase angle")
        ax.set_xlabel(r"Transerval x-direction in $\mu m$")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=0)
        self.backend(self.filepath + "_c_s_z_{}".format(z))


    def _plot_index_profile(self, ax, computational_grid):
        x_min, x_max = computational_grid.x_min, computational_grid.x_max
        z_min, z_max = computational_grid.z_min, computational_grid.z_max
        im = ax.pcolorfast((x_min, x_max), (z_min, z_max),
                           np.real(computational_grid.n_xz), cmap='inferno')

        return im

    def _waveguide_overlay(self, ax, computational_grid):
        x = computational_grid.x
        z = computational_grid.z
        n_xz = computational_grid.n_xz
        im = ax.contour(x, z, np.transpose(n_xz), cmap='Greys', levels=0,
                        linestyles='-')

        return im

    def _waveguide_mask_overlay(self, ax, computational_grid):
        x = computational_grid.x
        z = computational_grid.z
        mask = computational_grid.waveguide_mask
        im = ax.contour(x, z, np.transpose(mask), cmap='Greys', levels=0,
                        linestyles='dashed', linewidths=0.5)

        for i in im.collections:
            i.set_label(r'Waveguide mask')

        return im
        
    def _boundary_mask_overlay(self, ax, computational_grid):
        x = computational_grid.x
        z = computational_grid.z
        mask = np.ones((x.size, z.size))
        mask = np.transpose(np.tile(computational_grid.boundary_mask, (len(z), 1)))
        im = ax.contour(x, z, np.transpose(mask), cmap='Greys', levels=0,
                        linestyles='dotted', linewidths=0.5)

        for i in im.collections:
            i.set_label(r'Waveguide mask')

        return im

    def plot_field_waveguide_overlay(self, beam_propagator, colorbar=True, x_label=True, z_label=True, index_plot=True):

        #gridspec_kw={'height_ratios':[5, 1]}

        fig, ax = self.newfig(1, 1, 0.5, 1.25)


        im_field = self._plot_field(ax, beam_propagator)
        #im_index = self._waveguide_overlay(ax, beam_propagator.computational_grid)
        im_w_mask = self._waveguide_mask_overlay(ax, beam_propagator.computational_grid)
        im_b_mask = self._boundary_mask_overlay(ax, beam_propagator.computational_grid)
        divider = make_axes_locatable(ax)

        if index_plot:
            ax2 = divider.append_axes("top", size=0.2, pad=0.1)
            re = beam_propagator.computational_grid.n_xz[:,0]
            ax2.plot(beam_propagator.computational_grid.x, beam_propagator.computational_grid.n_xz[:,-1], label=r"$\text{Re}(n)$", linewidth=1)
            ax2.set_xlim(beam_propagator.computational_grid.x_min, beam_propagator.computational_grid.x_max)
            ax2.set_ylim(np.amin(re) - 0.1*(np.amax(re) - np.amin(re)), np.amax(re) + 0.1*(np.amax(re) - np.amin(re)))
            ax2.set_xticklabels([])
            ax2.set_yticks([])

            ax2.legend()

            #ax.legend(loc=4)
        if colorbar:
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cbar = fig.colorbar(im_field, cax=cax)
            cbar.ax.set_ylabel(r'Field Intensity')




        if x_label:
            ax.set_xlabel(self.x_label)
        if z_label:
            ax.set_ylabel(self.z_label)
        plt.tight_layout()
        self.backend(self.filepath + "_p_b_w_o")

    def plot_interpolated_field(self, obj):
        if isinstance(obj, optimizedYjunction):
            indexcalculator = obj.index_calc
        elif isinstance(obj, IndexCalculator):
            indexcalculator = obj
        fig, ax = self.newfig(1,1,1,2)

        x_min, x_max = indexcalculator.x[0], indexcalculator.x[-1]
        z_min, z_max = indexcalculator.z[0], indexcalculator.z[-1]
        field = np.abs(indexcalculator.interpolated_field)**2
        im_field = ax.pcolorfast((x_min, x_max), (z_min, z_max), field,
                           #norm=colors.SymLogNorm(linthresh=0.03, linscale=0.5,
                           #),
                           cmap='inferno')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im_field, cax=cax)
        cbar.ax.set_ylabel(r'Field Intensity')
        
        self.backend(self.filepath + "interp_field")

    def plot_ideal_index_distribution(self, obj):

        if isinstance(obj, optimizedYjunction):
            index_calculator = obj.index_calculator
        elif isinstance(obj, IndexCalculator):
            index_calculator = obj

        fig, (ax1, ax2) = self.newfig(1,2,1,2)

        x_min, x_max = index_calculator.x[1], index_calculator.x[-2]
        z_min, z_max = index_calculator.z[1], index_calculator.z[-2]

        index_real = np.transpose(np.real(
                                    index_calculator.ideal_index_distribution))
        index_imag = np.transpose(np.imag(
                                    index_calculator.ideal_index_distribution))
        im_index = ax1.pcolorfast((x_min, x_max), (z_min, z_max),
                                  index_real, cmap='magma')

        im_index_imag = ax2.pcolorfast((x_min, x_max), (z_min, z_max),
                                       index_imag, cmap='seismic', vmin=index_imag.min(), vmax=-index_imag.min())

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im_index, cax=cax)
        cbar.ax.set_ylabel(r'$Re(n)$')

        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="3%", pad=0.05)
        cbar2 = fig.colorbar(im_index_imag, cax=cax2)
        cbar2.ax.set_ylabel(r'$Im(n)$')

        ax1.set_xlabel(self.x_label)
        ax1.set_ylabel(self.z_label)

        ax2.set_xlabel(self.x_label)
        ax2.set_ylabel(self.z_label)
        ax1.set_title(r'Real part of the index distribution')
        ax2.set_title(r'Imaginary part of the index distribution')
        plt.tight_layout()
        self.backend(self.filepath + "_ideal_dist")

    def plot_trimmed_index_distribution(self, obj):

        if isinstance(obj, optimizedYjunction):
            index_calculator = obj.index_calculator
        elif isinstance(obj, IndexCalculator):
            index_calculator = obj

        fig, ax = plt.subplots()

        x_min, x_max = index_calculator.x[1], index_calculator.x[-2]
        z_min, z_max = index_calculator.z[1], index_calculator.z[-2]


        im_index = ax.pcolorfast((x_min, x_max), (z_min, z_max),
                                 index_calculator.trimmed_index_distribution,
                                cmap='magma')

        cbar = fig.colorbar(im_index, ax=ax)
        cbar.ax.set_ylabel(r'Refractive index $n$')

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.z_label)

        self.backend(self.filepath + "_trimmed_dist")

    def _plot_field(self, ax, beam_propagator):
        x_min, x_max = beam_propagator.computational_grid.x[0], beam_propagator.computational_grid.x[-1]
        z_min, z_max = beam_propagator.computational_grid.z[0], beam_propagator.computational_grid.z[-1]
        field = np.abs(beam_propagator.observer.efield_profile)**2
        im = ax.pcolorfast((x_min, x_max), (z_min, z_max), field,
                           #norm=colors.SymLogNorm(linthresh=0.03, linscale=0.5,
                           #),
                           cmap='inferno')
        return im

