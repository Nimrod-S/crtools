# UHECR analysis tools
A collection of useful tools for simulating ultra-high-energy cosmic ray propagation and analyzing the results.

## About
This code was written for and alongside the paper 
[TODO: add arxiv citation].

It is available for anyone who would like to reproduce our results, use our semi-analytic propagation simulation, or apply our analysis methods to existing UHECR data. If you use any of this code in your research, please cite the aforementioned paper.

For any issues or questions, contact nimrod.strasman@weizmann.ac.il.

## Primary usage
### Creating simulated UHECR arrival maps
```
randomize_rays.py -n NSIDE -e EXPOSURE -sd SOURCE_DENSITY -b BIAS -o OUTPUT_DIRECTORY --gmf M [--lenses] [--justs]
```
`randomize_rays.py` can be used to create arrival maps of UHECRs. `SOURCE_DENSITY` corresponds to $\log(s_0)$ and `BIAS` corresponds to $b_1$ as described in the paper. `EXPOSURE` is the name of the observatory exposure pattern (defined in `exposure.py`). `M` is the index of the UF23 GMF model (0-7) or -1 (default) in the case of no coherent GMF.

The file has two operating modes. Specify `--justs` to generate 10000 (by default) maps of random sources sampled from the LSS. Each generated map is a 2D numpy matrix where `axis=0` denotes redshift and `axis=1` denotes the angular coordinate. This assumes the existence of a source density field file in `<OUTPUT_DIRECTORY>/lss/` (see [here](#creating-the-lss-density-map)), and the results are saved in `<OUTPUT_DIRECTORY>/sourcemaps/`.

Do not specify the flag to generate 10000 (by default) arrival maps of cosmic rays in different observed energy bins. Each generated map is a 2D numpy array, where `axis=0` denotes the energy bin and `axis=1` denotes the angular coordinate. This assumes the existence of as many random source maps in `<OUTPUT_DIRECTORY>/sourcemaps/`, and $\psi$ vectors for the propagation calculation in `<OUTPUT_DIRECTORY>/flux/` (see [here](#semi-analytic-propagation-calculation)). The results are saved in `<OUTPUT_DIRECTORY>/hitmaps/<EXPOSURE>/`.

At least once, this file has to be called with the flag `--lenses`, which creates a "smear-map" for the random GMF and a "lens" for the coherent GMF. To create a lens for the coherent GMF, the code uses the [CRPropa3](https://crpropa.desy.de/) framework.

### Calculating test statistics for existing/simulated data
```
summarize_data.py -n NSIDE -f DATA_PATH
```
`summarize_data.py` calculates and prints the observable statistics discussed in the paper (large scale correlation measure, entropy, inter-energy correlation, reconstructed dipole) for a list of cosmic rays. It is written to parse a csv file in the format made available by Auger in the path `<DATA_PATH>/AugerApJS2022_Yr_JD_UTC_Th_Ph_RA_Dec_E_Expo.dat`. It constructs a healpy map out of the results, then calculates the relevant quantities and prints them - in the same way, the classes in `analysis.py` can be invoked directly to calculate the observables for any general UHECR map.

## Secondary usage
### Creating the LSS density map
```
lss.py -n NSIDE -o OUTPUT_DIRECTORY -m MRS_DIRECTORY
```
`lss.py` creates a 3D density map using the 2MRS catalog and saves it in `<OUTPUT_DIRECTORY>/lss/`. The output is a 2D numpy matrix where `axis=0` denotes redshift and `axis=1` denotes the angular coordinate. A "correction" file can be supplied in `<MRS_DIRECTORY>/CORRECTIONS/nearby.txt` that includes distances to nearby galaxies where the 2MRS redshift measurements are unreliable (created using `catalog.py`).

### Semi-analytic propagation calculation
The two most useful (for UHECR simulations) quantities derived in the paper are $\psi(z;E_o)$ (Equation 28), the volumetric emission rate of rays with observed energy above some threshold, and $\frac{d\phi(R;E_o)}{dR}$ (Equation 27), the rigidity distribution of rays with observed energy above some threshold. These quantities are calculated by the functions `calc_cosmic_ray_rate_density` and `get_r_dist` respectively in `propagation.py`. Running
```
propagation.py -o OUTPUT_DIRECTORY
```
generates the arrays `<OUTPUT_DIRECTORY>/flux/flux_nuc_v2.npy` and `<OUTPUT_DIRECTORY>/flux/flux_pro_v2.npy` that contain the values of $\psi$ for nuclei and protons.

The parameters of the heavy composition source spectrum model (Table 2) are defined in the function `get_source_parameters`.