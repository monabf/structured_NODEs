Code for the paper: "Recognition Models to Learn Dynamics from Partial 
Observations with Neural ODEs".
\
Contains tests with simulation and real data. The experimental data from the 
robotic exoskeleton is property of Wandercraft.
\
\
To start working:

- create a directory for this repo, further named dir
- create dir/Figures/Logs
- clone the repo in dir/src
- unzip the Data.zip file in dir/src to create dir/src/Data
- create a python3 virtual environment for this repo in dir/venv, source it
- install all requirements (`pip install -r src/requirements.txt`)
- install interpolation repo: `git clone https://github.com/aliutkus/torchinterp1d` into dir, `cd torchinterp1d`, `pip install -e .`
- if any problems occur during the installation of required packages, see
  src/requirements.txt for possible fixes
- `cd ../src`: you can now run the scripts.



To reproduce the results of the paper:

- Benchmark of recognition models: run `python 
  benchmark_recognition_models/earthquake_difftraj_fullNODE.py 1 
  KKL_u0T_back` for the earthquake model with joint optimization of the 
  dynamical parameters (Fig. 2, left). Similarly for the 
  FitzHugh-Nagumo model (Fig. 2, middle), run `python 
  benchmark_recognition_models/FitzHugh_Nagumo_ODE_difftraj_fullNODE.py 1 
  KKL_u0T_back`, and for the Van der Pol model (Fig. 2, right), run `python 
  benchmark_recognition_models/vanderpol_difftraj_fullNODE.py 1 
  KKLu_back`. Options for the recognition model (second argument of the 
  python command, the first being the process number) are: `KKL_u0T_back` for 
  backward KKL (`KKL_u0T` for forward), `KKLu_back` for 
  backward KKLu (`KKLu` for forward), `y0T_u0T` for direct, `y0_u0` for 
  direct with t_c = 0, `y0T_u0T_RNN_outNN_back` for backward RNN+ 
  (`y0T_u0T_RNN_outNN` for forward). Modify the other parameters (mostly 
  `init_state_obs_T` and `true_meas_noise_var`) to reproduce the ablation 
  studies.
- Harmonic oscillator: run `python 
  harmonic_oscillator_testcase/HO_back_physical_coords_NN_difftraj.py 1 
  KKL_u0T_back` for no structure (Fig. 5 a), 
  `HO_back_physical_coords_NN_hamiltonian.py` for Hamiltonian (Fig. 5 (b)), 
  `HO_back_physical_coords_NN_hamiltonian_x1dotx2.py` for a particular 
  Hamiltonian (Fig. 5 (c)), `HO_back_physical_coords_NN_paramid_linobs.py` for 
  a parametric model (Fig. 5 (d)), and `HO_back_physical_coords_NN_only_recog.
  py` for the extended state-space model (Fig. 5 (e)). The main options for the 
  recognition method (second argument) are again: `KKL_u0T_back`, `y0T_u0T_RNN_outNN_back`, 
  `y0T_u0T`, `y0_u0`.
- Robotic exoskeleton: run `python wandercraft_id/wandercraft_id_difftraj.py 
  1 KKL_u0T_back` for no structure (Fig. 7 (b)), 
  `wandercraft_id_difftraj_x1dotx2.py` for x1dot = x2, x3dot = x4 (Fig. 7 (b))
  , `wandercraft_id_difftraj_x1dotx2_residuals.py` for the residuals of the linear prior on 
  top of this constraint (Fig. 7 (c)). The options for the recognition 
  method (second argument) are: `KKL_u0T_back`, `y0T_u0T_RNN_outNN_back`, 
  `y0T_u0T`, `y0_u0`, `KKLu_back`.

The code runs in a few minutes on a regular laptop for the first two cases, 
but will need about a day for the robotics dataset.

 
