### Installation and Setup

The source code is available at GitHub and other sources

#### Dependencies

The *BeamPropagator.py* software was written using version 3.8. of the
Python-programming-language and, thus, requires a Python-3.8
installation or newer. Other Python-3 versions were not tested but might
be usable. The usage of Python-2 is *not* supported. The list of
dependencies apart from the Python installation is:

Numpy : The Python package *NumPy* provides the functionality for all
    array-operations performed when using the software.

Scipy : The Python package *SciPy* provides routines for the efficient
    solution of several essential linear systems.

Matplotlib : The Python package *Matplotlib* is used for all plotting
    functionality provided by BeamPropagator.py.

LaTeX : (optional) Graphs and images produced by the BPM-software can be
    exported to the file formats *pdf* and *pgf* using the
    LaTeX-typesetting system. This requires a working LaTeXinstallation
    as well as *dvipng* and *Ghostscript*, which may already be included
    in the LaTeX-installation. If not, they must be installed seperatly.
    Further information can be found in the Matplotlib documentation .

On the *Linux*-operating system, Python and all other dependencies can
commonly be installed with the package-manager used by the specific
Linux-distribution. The Python-packages can also be installed with the
native Python package-manager *pip*. On *Windows*, Python and the Python
packages can be installed using *Anaconda* . The LaTeXtypesetting system
can be installed through *TeXlive* or *MikTeX*. All dependencies must be
located in the systems PATH-environment.

### Usage

The software provides no graphic user interface (gui) or command line
interface (cli). It must be imported as a package and set up in a python
script in the following steps and order. 

1.  Import of the *BeamPropagator.py*-package.

2.  Initialize a grid for the propagation simulation through the
    *ComputationalGrid*-class. The Intialization requires the
    specification of the dimensions of the region of interest and the
    grid-spacing.

3.  Initialize a Waveguide-instance. Various types of waveguides are
    available natively and can be combinded to arbitrary structures. 

4.  Initialize a *Beam*-class instance. The base-mode of Gaussian Beams
    and Eigenmodes supported by the structure can be selected and
    arbitrarily combinded.

5.  Initialize a *BeamPropagator2D* child-class which provides a
    *propagate*-method. The parent-class *BeamPropagator2D* does not
    provide this method. If no instances of the
    *ComputationalGrid*-class, the *Waveguide*-class or the *Beam*-class
    are provided, instances of these clases with default values are
    used.

6.  Run the propagation simulation using the *run\_simulation()*-method
    of the initialized *BeamPropagator2D*-instance.

7.  (optional) Initialize a *Plotter*-class instance. The class provides
    several methods for visualization

8.  (optional) Initialize a *ioHandler*-class instance. The class allows
    the storage of the ComputationalGrid-class and the Observer-class.
    
A simple, exempalary script
using the BPM-software might take the following form

``` {.python breaklines="true" breakafter="-"}
from BeamPropagator import ComputationalGrid, WaveguideBase, GaussianBeam, Chung1990Solver, Plotter

grid = ComputationalGrid(x_params=(-20, 20, 200), z_params=(0, 200, 1000))
guide = WaveguideBase(refractive_index_medium = 1.5)
beam = GaussianBeam(wavelength=1)

solver = Chung1990Solver(grid, guide, beam)
solver.run_simulation()

plotter = Plotter()
plotter.plot_propageted_beam(solver)
```
