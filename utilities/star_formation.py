import numpy as np
import astropy.constants as con

# Universe Machine model in Behroozi+2019
# parameters taken from Appendix H
def star_formation_rate_B19(Mhalo, redshift):
    z = redshift
    gamma_0, gamma_a, gamma_z = -1.699, 4.206, -0.809
    alpha_0, alpha_a, alpha_la, alpha_z = -5.598, -20.731, 13.455, -1.321
    beta_0, beta_a, beta_z = -1.911, 0.395, 0.747
    V_0, V_a, V_la, V_z = 2.151, -1.658, 1.680, -0.233
    eps_0, eps_a, eps_la, eps_z = 0.109, -3.441, 5.079, -0.781
    delta0 = 0.055

    a = 1./(1+z)
    log10_gamma = gamma_0 + gamma_a * (1-a) + gamma_z * z
    beta        = beta_0  + beta_a  * (1-a) + beta_z  * z
    alpha       = alpha_0 + alpha_a * (1-a) + alpha_z * z + alpha_la * np.log(1+z)
    log10_eps   = eps_0   + eps_a   * (1-a) + eps_z   * z + eps_la   * np.log(1+z)
    log10_V     = V_0     + V_a     * (1-a) + V_z     * z + V_la     * np.log(1+z)

    # vmpeak - Mhalo relation taken from Appendix E2
    M200kms = 1.64e12 / ( (a/0.378)**(-0.142) + (a/0.378)**(-1.79) )  # Msun 
    Vmpeak = 200 * (Mhalo / M200kms)**(1./3) # km/s       # the powerlaw index in B19 is 3. We correct it here to 1/3.

    v = Vmpeak / 10**log10_V 
    SFR = 10**log10_eps *  ( 1./(v**alpha + v**beta) + 10**log10_gamma * np.exp( - np.log10(v)**2/2./delta0**2)  ) # Msun/yr
    return SFR

from scipy.optimize import fsolve
# Behroozi 2015 halo growth model
def halo_accretion_rate_B15(mhalo, redshift, cosmo):
    def __M13(z): 
        return 10**13.276 * (1+z)**3.00 * (1+z/2.)**(-6.11) * np.exp(-0.503 * z)
    
    def __a0(M0):
        return 0.205 - np.log10( (10**9.649 / M0)**0.18 + 1 )

    def __g(M0, a):
        return 1 + np.exp(-4.651 * (a - __a0(M0)))

    def __f(M0, z):    
        return np.log10(M0/ __M13(0)) * __g(M0, 1) / __g(M0, 1./(1+z))

    def Mmed(M0, z):
        return __M13(z) * 10**__f(M0, z) 

    def Mmed_dot(M0, z):
        dz = 0.01
        dt = cosmo.age(z+dz).value - cosmo.age(z-dz).value
        return np.abs((Mmed(M0, z+dz) - Mmed(M0, z-dz))/dt/1e9)

    def M0_of_Mpeak(Mpeak, z):
        return fsolve(lambda x: np.log10(Mmed(x, z)) - np.log10(Mpeak), np.log10(Mpeak))[0]

    def Mpeak_dot(Mpeak, z):
        M0 = M0_of_Mpeak(Mpeak, z)
        a = 1./(1+z)
        fac = (0.22*(a-0.4)**2 + 0.85) * (Mpeak/1e12)**(0.05*a) /  (1.0 + 10**(11 - 0.5*a)/Mpeak)**(0.04 + 0.1*a)
        return Mmed_dot(M0, z) * fac
    
    if np.isscalar(mhalo):
        return Mpeak_dot(mhalo, redshift)
    else:
        return np.array([Mpeak_dot(m, redshift) for m in mhalo])

# Fakhouri 2010 halo accretion model
def halo_accretion_rate_F10(mhalo, redshift, cosmo):
    # Mhalo in Msun
    # mean rates
    #mhalo_dot = 46.1 * (1 + 1.11*redshift) * np.sqrt(cosmo.Om0*(1+redshift)**3 + (1-cosmo.Om0))  \
    # * (mhalo / 1e12)**(1.1)

    # median rates
    mhalo_dot = 25.3 * (1 + 1.65*redshift) * np.sqrt(cosmo.Om0*(1+redshift)**3 + (1-cosmo.Om0))  \
     * (mhalo / 1e12)**(1.1)
    #corr = 10**(-0.1) # down 0.1 dex consider the drop of sigma8
    return mhalo_dot 

# Yung+2024 GUREFT halo growth rate
def halo_accretion_rate_Y23(mhalo, redshift, cosmo):
    Ez = cosmo.H(redshift).value/cosmo.H0.value
    a = 1./(1+redshift)
    alpha = 0.858 + 1.554 * a - 1.176 * a**2
    beta  = 10**(2.578 - 0.989*a - 1.545*a**2)
    mhalo_dot = beta * (mhalo/1e12 * Ez)**alpha
    return mhalo_dot

# RP16 halo growth rate
def halo_accretion_rate_RP16(mhalo, redshift, cosmo):
    Ez = cosmo.H(redshift).value/cosmo.H0.value
    a = 1./(1+redshift)

    # instantaneous
    #alpha = 0.975 + 0.300*a - 0.224*a**2 
    #beta  = 10**(2.677 - 1.708*a + 0.661*a**2)

    # average over dynamical time
    alpha = 1.000 + 0.329*a - 0.206*a**2 
    beta  = 10**(2.730 - 1.828*a + 0.654*a**2)
    mhalo_dot = beta * (mhalo/1e12*(cosmo.H0.value/100))**alpha * Ez / (cosmo.H0.value/100)
    return mhalo_dot

# star formation efficiency models

def star_formation_efficiency_H22(mhalo):
    # Harikane+2022   SFR/Mhalo_dot
    # who assumed the Salpeter IMF in calculating the SFR
    # that means SFR = 1.15e-28 * Luv [erg/s/Hz] which is corr larger than the one in Kennicutt & Evans 2012 assuming Kroupa IMF
    # corr = 1.15*1e-28 / 10**( - 43.35 + np.log10(con.c.value/(1500*1e-10)))  

    return 6.4e-2 / ( (mhalo/10**(11.5))**0.5 + (mhalo/10**(11.5))**(-1.2) ) / (0.049/0.3089)

def star_formation_efficiency_H22_zdep(mhalo):
    # Harikane+2022 (z-dependent) at z=7
    z = 7
    corr = 0.53 * np.tanh(0.54*(2.9-z)) + 1.53
    return 6.4e-2 / ( (mhalo/10**(11.5))**0.5 + (mhalo/10**(11.5))**(-1.2) )  * corr / (0.049/0.3089)

def star_formation_efficiency_fiducial(mhalo, fsfe=1.):
    # our adaptation of the double power law model 
    #return fsfe * 0.5 * 6.4e-2 / ( (mhalo/10**(12.0))**0.5 + (mhalo/10**(12.0))**(-0.6) ) / parameters['fbaryon']
    return fsfe * 2 * 0.1 / ( (mhalo/10**(12.0))**0.5 + (mhalo/10**(12.0))**(-0.6) )

def star_formation_efficiency_T18(mhalo):
    # Tacchella+2018, Z-const model  (SFR/Mgas_dot)
    #eps0, Mcrit, beta, gamma = 0.25, 6.89e10, 1.14, 0.35 
    eps0, Mcrit, beta, gamma = 0.26, 7.10e10, 1.09, 0.36
    return 2 * eps0 / ( (mhalo/Mcrit)**(-beta) + (mhalo/Mcrit)**gamma )

def star_formation_efficiency_M18(mhalo):
    # Moster+2018
    z = 4 
    eps0 = 0.005 + 0.689 * z/(1.+z)  
    Mcrit= 10**(11.339 + 0.692 * z/(1.+z))
    beta = 3.344 + 2.079 * z/(1.+z)
    gamma= 0.966
    return 2 * eps0 / ( (mhalo/Mcrit)**(-beta) + (mhalo/Mcrit)**gamma )

def star_formation_efficiency_firebox(logmhalo, mode='UVint'):
    # Feldmann+2024
    # SFR/Mhalo_dot
    if mode == "UVint":
        A = -2.351;  xb = 9.4;  Delta = 0.160
        alpha1 = 0.939; alpha2 = 0.352
    elif mode == "UVobs":
        A = -2.329;  xb = 9.465;  Delta = 0.194
        alpha1 = 0.945; alpha2 = 0.127
    elif mode == "SFR100":
        A = -2.855;  xb = 9.192;  Delta = 0.124
        alpha1 = 1.321; alpha2 = 0.371
    elif mode == "SFR20":
        A = -2.631;  xb = 9.386;  Delta = 0.144
        alpha1 = 0.935; alpha2 = 0.326
    return 10** (A + alpha1*(logmhalo-xb) + (alpha2 - alpha1) * Delta * (np.log(1+np.exp((logmhalo-xb)/Delta)) - np.log(2)) )

def return_Vmax_RP16(mhalo, redshift, cosmo):
    # RP16 
    Ez = cosmo.H(redshift).value/cosmo.H0.value
    a = 1./(1+redshift)
    alpha = 0.346 - 0.059*a + 0.025*a**2
    beta  = 10**(2.209 + 0.060*a - 0.021*a**2)
    Vmax = beta * (mhalo/1e12*(cosmo.H0.value/100) * Ez)**alpha 
    return Vmax

def return_Vvir(mhalo, redshift, cosmo):
    rho_c = cosmo.critical_density(redshift).to('Msun/kpc^3').value
    rvir = (mhalo / (178 * rho_c) / (4*np.pi/3.))**(1./3) 
    return np.sqrt(con.G.value * mhalo * con.M_sun.value / rvir / 1e3 / con.pc.value) / 1e3


# original code from F23
# Halo virial velocity (mass in solar masses; vvir output in km/s)
def return_Vvir_F23(M, z, cosmo):
    omz          = cosmo.Om0 * (1+z)**3/ (cosmo.Om0*(1+z)**3+ (1-cosmo.Om0))
    d            = omz - 1
    Deltac       = 18*np.pi**2 + 82*d - 39*d**2
    h            = cosmo.h
    vvir         = 23.4*(M*h/1e8)**(1./3.)*(cosmo.Om0/omz*Deltac/18/np.pi**2)**(1./6.)*np.sqrt((1.+z)/10.)
    return vvir

def star_formation_rate_F23(mhalo, redshift, cosmo, parameters):
    fw = 0.1
    eps0 = 0.02
    vs = 975
    vc = return_Vvir_F23(mhalo, redshift, cosmo)
    eps = eps0 * vc**2 / (vc**2 + fw * vs**2) 
    # correct for different UV to SFR conversion factors 
    fcorr = 0.587e10 * con.L_sun.value * 1e7 /  (con.c.value / (1500*1e-10))  / (1/0.72 * 1e28)
    return eps/0.06  * cosmo.H(redshift).value * (1e3/1e6/con.pc.value) * (365.25 * 24 * 3600) * mhalo * parameters['fbaryon'] / fcorr

# SFR and Luv conversion models
def convert_sfr_to_Muv(sfr, model='Madau2014'):
    if model == 'Kennicutt2012':
        logCx = 43.35 # Kennicutt & Evans 2012 (assuming Kroupa IMF; using STARBURST99; solar metallicity)
        logLx = np.log10(sfr) + logCx  # log Lx in erg/s
        fnu = 10**logLx / (4*np.pi* (10*con.pc.value*100)**2 ) / (con.c.value/(1500*1e-10))
        Muv = -2.5 * np.log10(fnu) - 48.6 # AB mag
    elif model == "Madau2014_Salpeter":
        fnu = (sfr / 1.15e-28)/ (4*np.pi* (10*con.pc.value*100)**2 ) # erg/s/Hz/cm^2
        Muv = -2.5 * np.log10(fnu) - 48.6 # AB mag
    elif model == "Madau2014":  # assuming a Chabrier IMF
        fnu = (sfr / (1.15e-28 * 0.63))/ (4*np.pi* (10*con.pc.value*100)**2 ) # erg/s/Hz/cm^2
        Muv = -2.5 * np.log10(fnu) - 48.6 # AB mag
    else:
        raise ValueError("Unknown model of SFR to Muv conversion")
    
    return Muv

# dust attenuation models
def dust_attenuation(muv, redshift):
    # muv: intrinsic UV magnitude
    k_softplus = 10   # allow smooth transition when Muv_obs < Muv
    if redshift <= 10:
        C0, C1 = 4.43, 1.99  # IRX beta relation, M99
        if redshift >= 8:
            slope = -0.17; Mref = -19.5; intercept = -2.085 # Cullen 2023
        elif redshift == 7:
            slope = -0.20; Mref = -19.5; intercept = -2.05 # Bouwens 2014
        elif redshift == 6:
            slope = -0.20; Mref = -19.5; intercept = -2.00
        elif redshift == 5:
            slope = -0.14; Mref = -19.5; intercept = -1.91
        elif redshift == 4:
            slope = -0.11; Mref = -19.5; intercept = -1.85

        scatter = 0 # for a median relation
        #scatter=0.35 # for a mean relation

        prefactor = 1/(1 - C1 * slope)
        muv_obs = prefactor * (  muv  + C0 + C1 * intercept - C1 * slope * Mref + 0.2 * np.log(10) * C1**2 * scatter**2  )    # Vogelsberger 2020
        #return muv_obs * (muv_obs >= muv) + muv * (muv_obs < muv)
        return 1/k_softplus * np.log(1 + np.exp( k_softplus *( muv_obs - muv) )) + muv 
    else:
        return muv