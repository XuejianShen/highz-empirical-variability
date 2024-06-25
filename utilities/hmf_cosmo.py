import hmf
from hmf import mass_function
import numpy as np

## halo mass functions
# here the kmax of CAMB transfer function is fixed to 1e3, one would get almost identical results with extrapolations (Eisenstein & Hu 1998)
# also identical results if one use output files from CAMB calculations done elsewhere
def calculate_hmf_kmaxfixed(z, Mmin, Mmax, dlog10m, cosmo, parameters):
        halo_mass_function = mass_function.hmf.MassFunction(z=z, Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m, 
                cosmo_model=cosmo, n = parameters['ns'], sigma_8 = parameters['sigma8'], delta_c=1.686, 
                transfer_model = hmf.density_field.transfer_models.CAMB, transfer_params = {"kmax":1e3},
                hmf_model      = hmf.mass_function.fitting_functions.Behroozi, hmf_params=None, 
                mdef_model     = hmf.halos.mass_definitions.SOVirial, mdef_params=None, 
                filter_model   = hmf.density_field.filters.TopHat, filter_params=None)

        hubble = cosmo.H0.value/100 
        mhalo_arr    = np.arange(Mmin, Mmax, dlog10m) - np.log10(hubble)    # log10(M[Msun])
        phi_halo_arr = halo_mass_function.dndlog10m  * hubble**3            # dn/dlog10M [1/Mpc^3/dex]
        return mhalo_arr, phi_halo_arr

# for EDE, we used the tabulated transfer function output from CAMB calculations
def calculate_hmf_ede(z, Mmin, Mmax, dlog10m, cosmo, parameters):
        halo_mass_function = mass_function.hmf.MassFunction(z=z, Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m, 
                cosmo_model=cosmo, n = parameters['ns'], sigma_8 = parameters['sigma8'], delta_c=1.686, 
                # transfer function defaults to CAMB
                transfer_model = hmf.density_field.transfer_models.FromFile, 
                transfer_params = {"fname": "./transfer_function_output/transfer_ede.dat"},
                hmf_model      = hmf.mass_function.fitting_functions.Behroozi, hmf_params=None, 
                mdef_model     = hmf.halos.mass_definitions.SOVirial, mdef_params=None, 
                filter_model   = hmf.density_field.filters.TopHat, filter_params=None)

        hubble = cosmo.H0.value/100 
        mhalo_arr    = np.arange(Mmin, Mmax, dlog10m) - np.log10(hubble)    # log10(M[Msun])
        phi_halo_arr = halo_mass_function.dndlog10m  * hubble**3            # dn/dlog10M [1/Mpc^3/dex]
        return mhalo_arr, phi_halo_arr

## cosmology 
def mag_correct_cosmo(redshift, cosmo1, cosmo2):
    # correction from cosmo1 to cosmo2
    # e.g. true luminosity of an object in cosmo2 but interpreted by observer assuming cosmo1
    d1 = cosmo1.luminosity_distance(redshift).value
    d2 = cosmo2.luminosity_distance(redshift).value
    return - 2.5 * np.log10((d2/d1)**2)

def phi_correct_cosmo(redshift, cosmo1, cosmo2):
    # correction from cosmo1 to cosmo2
    vol1 = cosmo1.differential_comoving_volume(redshift).value
    vol2 = cosmo2.differential_comoving_volume(redshift).value
    volume_corr_factor = vol1 / vol2
    return np.log10(volume_corr_factor)