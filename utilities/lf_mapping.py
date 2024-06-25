import numpy as np
import utilities.star_formation as sf

def mapfunc_mhalo_to_muv(log_mhalo, parameters, cosmo, epsilon, redshift, include_dust=True, imf="Chabrier"):
    '''
    mapping from halo mass to UV magnitude (without scatter)
    muv: UV magnitude
    log_mhalo: log10 of halo mass in Msun
    '''
    if type(epsilon) == str:
        if epsilon == "H22":
            epsilon = sf.star_formation_efficiency_H22(10**log_mhalo)
        elif epsilon == "H22_z7":
            epsilon = sf.star_formation_efficiency_H22_zdep(10**log_mhalo)
        elif epsilon == "this_work":
            epsilon = sf.star_formation_efficiency_fiducial(10**log_mhalo)
        elif epsilon == "T18":
            epsilon = sf.star_formation_efficiency_T18(10**log_mhalo)
        elif epsilon == "FIREbox":
            epsilon = sf.star_formation_efficiency_firebox(log_mhalo, mode='SFR100') / parameters['fbaryon']
            #print(epsilon)
        elif "vary" in epsilon: # e.g. "vary_0.01"
            fcorr = float(epsilon.split("_")[1])
            epsilon = sf.star_formation_efficiency_fiducial(10**log_mhalo, fsfe = fcorr)
        else:
            raise ValueError("epsilon not recognized")

    sfr = epsilon * parameters['fbaryon'] * sf.halo_accretion_rate_RP16(10**log_mhalo, redshift=redshift, cosmo=cosmo)
    
    if imf == "Chabrier":
        muv_raw = sf.convert_sfr_to_Muv(sfr)
    elif imf == "Salpeter":
        muv_raw = sf.convert_sfr_to_Muv(sfr, model='Madau2014_Salpeter')
    else:
        raise ValueError("IMF not recognized")

    if include_dust:
        muv = sf.dust_attenuation(muv_raw, redshift)
    else:
        muv = muv_raw
    return muv

def mapfunc_jacobian_numeric(log_mhalo, *args, **kwargs):
    # return dMuv/dlogMhalo numerically
    # compute the derivative of muv with respect to mhalo
    dlogm=0.001
    muv_plus = mapfunc_mhalo_to_muv(log_mhalo + dlogm, *args, **kwargs)
    muv_minus = mapfunc_mhalo_to_muv(log_mhalo - dlogm, *args, **kwargs)
    dmuv_dlogm = (muv_plus - muv_minus) / (2*dlogm)
    return np.abs(dmuv_dlogm)

def archived_version_convolve_on_grid(input_grid, input_weight, sigma_uv_input, normalization_check=True):
    # convolve with a gaussian kernel
    # over the UV luminosity function
    grid_binsize  = np.abs(input_grid[1] - input_grid[0])
    minimum_sigma = grid_binsize/4. # set to the binsize divided by a constant
    #print(minimum_sigma)
    if np.isscalar(sigma_uv_input):
        sigma_uv = max(sigma_uv_input, minimum_sigma) # regulate the miminum sigma to be of order the binsize (~ 0.01 dex)
    else:
        sigma_uv = sigma_uv_input.copy()
        sigma_uv[sigma_uv < minimum_sigma] = minimum_sigma

    output_weight = np.zeros(len(input_grid))
    for i, mapfrom in enumerate(input_grid):
        raw_output    = np.zeros(len(input_grid))
        for j, mapto in enumerate(input_grid):
            if np.isscalar(sigma_uv):
                raw_output[j] += 1./np.sqrt(2*np.pi*sigma_uv**2) * np.exp( -0.5 * (mapto - mapfrom)**2 / sigma_uv**2 ) * grid_binsize
            else:
                raw_output[j] += 1./np.sqrt(2*np.pi*sigma_uv[i]**2) * np.exp( -0.5 * (mapto - mapfrom)**2 / sigma_uv[i]**2 ) * grid_binsize
        
        if normalization_check: # redundant normalization
            sum_raw_output = np.sum(raw_output)
            if np.abs(sum_raw_output - 1) > 1e-2:
                print("Warning: the convolution is not perfectly normalized", sum_raw_output, 'at', input_grid[i])
            if sum_raw_output > 0:
                raw_output = raw_output/sum_raw_output
            else:
                raw_output = np.zeros(len(input_grid))
        output_weight += input_weight[i] * raw_output 
    return output_weight


def convolve_on_grid(input_grid, input_weight, sigma_uv_input, normalization_check=False, verbose=False):
    # convolve with a gaussian kernel
    # over the UV luminosity function
    grid_binsize  = np.abs(input_grid[2:] - input_grid[:-2])/2.
    grid_binsize  = np.append(grid_binsize, grid_binsize[-1])
    grid_binsize  = np.append(grid_binsize[0], grid_binsize)

    minimum_sigma = np.median(grid_binsize)/4. # set to the binsize divided by a constant
    #print(minimum_sigma)
    if np.isscalar(sigma_uv_input):
        sigma_uv = max(sigma_uv_input, minimum_sigma) # regulate the miminum sigma to be of order the binsize (~ 0.01 dex)
    else:
        sigma_uv = sigma_uv_input.copy()
        sigma_uv[sigma_uv < minimum_sigma] = minimum_sigma

    output_weight = np.zeros(len(input_grid))
    for i, mapfrom in enumerate(input_grid):
        raw_output    = np.zeros(len(input_grid))
        for j, mapto in enumerate(input_grid):
            if np.isscalar(sigma_uv):
                raw_output[j] += 1./np.sqrt(2*np.pi*sigma_uv**2) * np.exp( -0.5 * (mapto - mapfrom)**2 / sigma_uv**2 ) * grid_binsize[j]
            else:
                raw_output[j] += 1./np.sqrt(2*np.pi*sigma_uv[i]**2) * np.exp( -0.5 * (mapto - mapfrom)**2 / sigma_uv[i]**2 ) * grid_binsize[j]
        
        if normalization_check: # redundant normalization
            sum_raw_output = np.sum(raw_output)
            if np.abs(sum_raw_output - 1) > 1e-2:
                if verbose: print("Warning: the convolution is not perfectly normalized", sum_raw_output, 'at', input_grid[i])
            if sum_raw_output > 0:
                raw_output = raw_output/sum_raw_output
            else:
                raw_output = np.zeros(len(input_grid))

        output_weight += raw_output * (input_weight[i] * grid_binsize[i])
    return output_weight/grid_binsize

#########
def compute_uv_luminosity_function(mhalo_arr, phi_halo_arr, sigma_uv, parameters, cosmo, 
                                   epsilon, redshift, include_dust=True, imf="Chabrier", variable_sigma_uv=None,
                                   normalization_check=False):
    # mhalo_arr: log10(Mhalo/Msun)
    # phi_halo_arr: dn/dlog10Mhalo
    # compute the UV luminosity function from the halo mass function
    muv_arr    = mapfunc_mhalo_to_muv(     mhalo_arr, parameters, cosmo, epsilon=epsilon, 
                                           redshift=redshift, include_dust=include_dust, imf=imf)
    dmuv_dlogm = mapfunc_jacobian_numeric( mhalo_arr, parameters, cosmo, epsilon=epsilon, 
                                           redshift=redshift, include_dust=include_dust, imf=imf)
    
    if variable_sigma_uv is not None:
        sigma_uv_arr = variable_sigma_uv(mhalo_arr)
        phi_uv_arr = convolve_on_grid(muv_arr, phi_halo_arr/dmuv_dlogm , sigma_uv_input = sigma_uv_arr, normalization_check=normalization_check)
    else:
        if sigma_uv > 0: phi_uv_arr = convolve_on_grid(muv_arr, phi_halo_arr/dmuv_dlogm , sigma_uv_input = sigma_uv, normalization_check=normalization_check)
        else:            phi_uv_arr = phi_halo_arr/dmuv_dlogm
    #print(phi_uv_arr)
    return muv_arr, phi_uv_arr


def compute_linking_matrix(input_grid, input_weight, sigma_uv_input):
    grid_binsize  = np.abs(input_grid[2:] - input_grid[:-2])/2.
    grid_binsize  = np.append(grid_binsize, grid_binsize[-1])
    grid_binsize  = np.append(grid_binsize[0], grid_binsize)

    minimum_sigma = np.median(grid_binsize)/4.  # set to the binsize divided by a constant
    if np.isscalar(sigma_uv_input):
        sigma_uv = max(sigma_uv_input, minimum_sigma) # regulate the miminum sigma to be of order the binsize (~ 0.01 dex)
    else:
        sigma_uv = sigma_uv_input.copy()
        sigma_uv[sigma_uv < minimum_sigma] = minimum_sigma

    linking_matrix = np.zeros((len(input_grid), len(input_grid)))
    for i, mapfrom in enumerate(input_grid):
        for j, mapto in enumerate(input_grid):
            if np.isscalar(sigma_uv):
                linking_matrix[i,j] = 1./np.sqrt(2*np.pi*sigma_uv**2) * np.exp( -0.5 * (mapto - mapfrom)**2 / sigma_uv**2 ) \
                    * grid_binsize[j] * (input_weight[i] * grid_binsize[i])
            else:
                linking_matrix[i,j] = 1./np.sqrt(2*np.pi*sigma_uv[i]**2) * np.exp( -0.5 * (mapto - mapfrom)**2 / sigma_uv[i]**2 ) \
                    * grid_binsize[j] * (input_weight[i] * grid_binsize[i])
    return linking_matrix 

#########
def compute_mhalo_muv_relation(mhalo_arr, phi_halo_arr, sigma_uv, parameters, cosmo, 
                               epsilon, redshift, include_dust=True, imf="Chabrier", variable_sigma_uv=None):
    
    muv_arr    = mapfunc_mhalo_to_muv(     mhalo_arr, parameters, cosmo, epsilon=epsilon, 
                                           redshift=redshift, include_dust=include_dust, imf=imf)
    dmuv_dlogm = mapfunc_jacobian_numeric( mhalo_arr, parameters, cosmo, epsilon=epsilon, 
                                           redshift=redshift, include_dust=include_dust, imf=imf)
    phi_uv_arr_noscatter = phi_halo_arr /dmuv_dlogm

    if variable_sigma_uv is not None:
        sigma_uv_arr = variable_sigma_uv(mhalo_arr)
        linking_matrix = compute_linking_matrix(muv_arr, phi_uv_arr_noscatter, sigma_uv_input=sigma_uv_arr)
        return mhalo_arr, muv_arr, linking_matrix
    else:
        if sigma_uv > 0: 
            linking_matrix = compute_linking_matrix(muv_arr, phi_uv_arr_noscatter, sigma_uv)
            return mhalo_arr, muv_arr, linking_matrix
        else:            
            return mhalo_arr, muv_arr, None
        
#########
def weighted_percentile(values, weights=None, percentile=[50]):
    '''
    Calculate weighted percentiles.
    values, weights -- Numpy ndarrays with the same shape.
    mapping from Pr(x<=x0) to x0
    '''
    if weights is None:
        weights = np.ones(len(values))

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    percentile = np.array(percentile)/100.
    weighted_percentile = (np.cumsum(weights)) / np.sum(weights)
    return np.interp(percentile, weighted_percentile, values)