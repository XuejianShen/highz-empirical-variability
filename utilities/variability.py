import numpy as np

def sigma_uv_vs_mhalo_alt(log_mhalo, yfloor):
    y = 1.1 - 0.34 * (log_mhalo - 10)
    y[y<yfloor] = yfloor
    if yfloor < 2.0:
        y[y>2.00] = 2.00
    else:
        y[y>yfloor] = yfloor
    return y

def sigma_uv_vs_mhalo_alt2(log_mhalo, yfloor, delta, ymax=2.0):
    y = 1.1 - 0.34 * (log_mhalo - 10) + delta

    if yfloor + delta > ymax:
        return np.ones_like(log_mhalo) * (yfloor + delta)

    yfloor = max(yfloor, 0)
    y[y<yfloor] = yfloor
    y[y>ymax] = ymax
    return y

def sigma_uv_vs_mhalo_alt3(log_mhalo, log_mbreak, y1, y2):
    # a step-like function
    out = np.zeros_like(log_mhalo)
    out[log_mhalo < log_mbreak] = y1
    out[log_mhalo >= log_mbreak] = y2
    return out

## literature values
def sigma_uv_vs_mhalo_G24(log_mhalo):
    # taken from Gelli+2024
    y = 4.5 - 0.34 * log_mhalo
    yfloor = 0.
    y[y<yfloor] = yfloor
    return y

def sigma_uv_vs_mhalo_firebox(log_mhalo, mode='UVint'):
    # taken from Feldmann+2024
    if mode == "UVint":
        xb = 9.42; Delta = 0.166
        c1 = 0.646; c2 = 0.341
    elif mode == "UVobs":
        xb = 9.476; Delta = 0.196
        c1 = 0.646; c2 = 0.281
    elif mode == "SFR100":
        xb = 9.855; Delta = 0.081
        c1 = 0.479; c2 = 0.390
    elif mode == "SFR20":
        xb = 8.719; Delta = 0.569
        c1 = 0.248; c2 = 0.797

    sig_log_sfr = c1 + (c2 - c1) / (1 + np.exp(-(log_mhalo - xb)/Delta))
    return 2.5 * sig_log_sfr
