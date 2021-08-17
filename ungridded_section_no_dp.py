import numpy as np
import numpy.ma as ma
import pyart

def quality_control_no_dp(radar1,n,calibration):
    #Inputs,
    #radar1: Raw volume data
    #n: Radar scan counter
    #calibration: Differential Reflectivity (Zdr) calibration value -- not used
    print('Pre-grid Organization Section')
    #Pulling apart radar sweeps and creating ungridded data arrays
    ni = 0
    radar = radar1
    n = n+1
    range_i = radar1.range['data']
    ref_ungridded_base = radar1.fields['reflectivity']['data']

    #Get 2d ranges at lowest tilt
    range_2d = np.zeros((ref_ungridded_base.shape[0], ref_ungridded_base.shape[1]))
    for i in range(ref_ungridded_base.shape[0]):
        range_2d[i,:]=range_i
        
    #Get stuff for QC control rings
    ungrid_lons = radar1.gate_longitude['data']
    ungrid_lats = radar1.gate_latitude['data']
    gate_altitude = radar1.gate_altitude['data'][:]

    #Returning variables,
    #radar: Quality-controlled volume data
    #n: Radar scan counter
    #range_2d: Range array of lowest tilt used to define outer limit of effective data
    #ungrid_lons,ungrid_lats: Longitude and Latitude arrays at the lowest tilt
    return radar,n,range_2d,ungrid_lons,ungrid_lats