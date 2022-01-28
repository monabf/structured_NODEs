Code for the ICML 2022 submission: "Using what you know: Learning dynamics 
from partial observations with structured neural ODEs".
\
Contains tests with simulation and real data. The experimental data from the 
robotic exoskeleton is property of Wandercraft.
\
\
To start working:

- create a directory for this repo, further named dir
- create dir/Figures/Logs
- clone the repo in dir/src
- unzip the Data.zip file
- create a python3 virtual environment for this repo in dir/venv, source it
- install all requirements (`pip install -r src/requirements.txt`)
- install interpolation repo: `git clone https://github.com/aliutkus/torchinterp1d` into dir, `cd torchinterp1d`, `pip install -e .`
- if any problems occur during the installation of required packages, see
  src/requirements.txt for possible fixes
- `cd ../src`: you can now run the scripts.



To reproduce the results of the paper:

- Benchmark of recognition models: run `python 
  benchmark_recognition_models/earthquake_difftraj.py 1 KKL_u0T_back` for 
  the earthquake model with fixed dynamical parameters (Fig. 7, top row) and 
  `python benchmark_recognition_models/earthquake_difftraj_paramid.py 1 
  KKL_u0T_back` for the earthquake model with joint optimization of the 
  dynamical parameters (Fig. 7, bottom row). Similarly for the 
  FitzHugh-Nagumo model (Fig. 9), run `python benchmark_recognition_models/FitzHugh_Nagumo_ODE_difftraj(_paramid).py 1 
  KKL_u0T_back`. Options for the recognition model (second argument of the 
  python command, the first being the process number) are: `KKL_u0T_back` for 
  backward KKL, `KKL_u0T` for forward KKL, `y0T_u0T` for direct, `y0_u0` for 
  direct with t_c = 0.
- Harmonic oscillator: run `python 
  harmonic_oscillator_testcase/HO_back_physical_coords_NN_difftraj.py 1 
  KKL_u0T_back` for no structure (Fig.3 a), 
  `HO_back_physical_coords_NN_hamiltonian.py` for Hamiltonian (Fig.3 (b)), 
  `HO_back_physical_coords_NN_hamiltonian_x1dotx2.py` for a particular 
  Hamiltonian (Fig.3 (c)), `HO_back_physical_coords_NN_paramid_linobs.py` for 
  a parametric model (Fig.3 (d)), and `HO_back_physical_coords_NN_only_recog.
  py` for the extended state-space model (Fig.3 (e)). The options for the 
  recognition method (second argument) are again: `KKL_u0T_back`, `KKL_u0T`, 
  `y0T_u0T`, `y0_u0`.
- Robotic exoskeleton: run `python wandercraft_id/wandercraft_id_difftraj.py 
  1 KKL_u0T_back` for no structure (Fig.4 (b)), 
  `wandercraft_id_difftraj_x1dotx2.py` for x1dot = x2, x3dot = x4 (Fig.4 (c))
  , `wandercraft_id_difftraj.py` for the residuals of the linear prior on 
  top of this constraint. The options for the recognition method (second 
  argument) are: `KKL_u0T_back`, `KKL_u0T`, `y0T_u0T`, `y0_u0`, `KKLu_back`, 
  `KKLu` (KKLu recognition method backward and forward also possible for this 
  nonautonomous system).

The code runs in a few minutes on a regular laptop for the first two cases, 
but will need about a day for the robotics dataset.

 
