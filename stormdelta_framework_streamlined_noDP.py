import matplotlib.pyplot as plt
import pyart
import numpy as np
import numpy.ma as ma
import math
from metpy.units import check_units, concatenate, units
#from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from siphon.radarserver import RadarServer
#rs = RadarServer('http://thredds-aws.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
#rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')
from datetime import datetime, timedelta
from siphon.cdmr import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from metpy.units import check_units, concatenate, units
#from metpy.units import atleast_1d, check_units, concatenate, units
from shapely.geometry import polygon as sp
import pyproj 
import shapely.ops as ops
from shapely.ops import transform
from shapely.geometry.polygon import Polygon
from functools import partial
from shapely import geometry
import netCDF4
from scipy import ndimage as ndi
from pyproj import Geod
from metpy.calc import wind_direction, wind_speed, wind_components
import matplotlib.lines as mlines
import pandas as pd
import scipy.stats as stats
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
import nexradaws
import os
from grid_section_no_dp import gridding_no_dp
from ungridded_section_no_dp import quality_control_no_dp
from stormid_section_xtrap import storm_objects_new

def storm_motion_deltas_algorithm(REFlev, REFlev1, big_storm, numsecs, zero_z_trigger, storm_to_track, year, month, day, hour, start_min, duration, calibration, offhodoshear, station, Bunkers_s, Bunkers_m, meanw_s, meanw_m, shear_dir, track_dis, GR_mins=1.0):
    #Set storm motion
    Bunkers_m = Bunkers_m
    #Set reflectivity thresholds for storm tracking algorithm
    REFlev = [REFlev]
    REFlev1 = [REFlev1]
    #Set storm size threshold that triggers subdivision of big storms
    big_storm = big_storm #km^2
    Outer_r = 30 #km
    Inner_r = 6 #km
    #Set trigger to ignore strangely formatted files right before 00Z
    #Pre-SAILS #: 17
    #SAILS #: 25
    zero_z_trigger = zero_z_trigger
    storm_to_track = storm_to_track
    #Here, set the initial time of the archived radar loop you want.
    #Our specified time
    dt = datetime(year,month, day, hour, start_min)
    station = station
    track_dis = track_dis
    end_dt = dt + timedelta(hours=duration)

    #Set up nexrad interface
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans_in_range(dt,end_dt,station)
    results = conn.download(scans, 'RadarFolder')

    #Setting counters for figures and Pandas indices
    f = 27
    n = 1
    storm_index = 0
    scan_index = 0
    tracking_index = 0
    #Create geod object for later distance and area calculations
    g = Geod(ellps='sphere')

    #Actual algorithm code starts here
    #Create a list for the lists of arc outlines
    tracks_dataframe = []
    for i,scan in enumerate(results.iter_success(),start=1):
    #Local file option:
        #Loop over all files in the dataset and pull out each 0.5 degree tilt for analysis
        try:
            radar1 = scan.open_pyart()
        except:
            print('bad radar file')
            continue
        #Local file option
        print('File Reading')
        #Make sure the file isn't a strange format
        if radar1.nsweeps > zero_z_trigger:
            continue
            
        for i in range(radar1.nsweeps):
            print('in loop')
            print(radar1.nsweeps)
            try:
                radar4 = radar1.extract_sweeps([i])
            except:
                print('bad file')
            #Checking to make sure the tilt in question has all needed data and is the right elevation
            #if ((np.mean(radar4.elevation['data']) < .65) and (np.max(np.asarray(radar4.fields['reflectivity']['data'])) != np.min(np.asarray(radar4.fields['reflectivity']['data'])))):
            if ((np.mean(radar4.elevation['data']) < .65) and (np.max(np.asarray(radar4.fields['velocity']['data'])) == np.min(np.asarray(radar4.fields['velocity']['data'])))):
                n = n+1

                #Calling ungridded_section; Pulling apart radar sweeps and creating ungridded data arrays
                [radar,n,range_2d,ungrid_lons,ungrid_lats] = quality_control_no_dp(radar4,n,calibration)

                time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
                object_number=0.0
                month = time_start.month
                if month < 10:
                    month = '0'+str(month)
                hour = time_start.hour
                if hour < 10:
                    hour = '0'+str(hour)
                minute = time_start.minute
                if minute < 10:
                    minute = '0'+str(minute)
                day = time_start.day
                if day < 10:
                    day = '0'+str(day)
                time_beg = time_start - timedelta(minutes=0.1)
                time_end = time_start + timedelta(minutes=GR_mins)
                sec_beg = time_beg.second
                sec_end = time_end.second
                min_beg = time_beg.minute
                min_end = time_end.minute
                h_beg = time_beg.hour
                h_end = time_end.hour
                d_beg = time_beg.day
                d_end = time_end.day
                if sec_beg < 10:
                    sec_beg = '0'+str(sec_beg)
                if sec_end < 10:
                    sec_end = '0'+str(sec_end)
                if min_beg < 10:
                    min_beg = '0'+str(min_beg)
                if min_end < 10:
                    min_end = '0'+str(min_end)
                if h_beg < 10:
                    h_beg = '0'+str(h_beg)
                if h_end < 10:
                    h_end = '0'+str(h_end)
                if d_beg < 10:
                    d_beg = '0'+str(d_beg)
                if d_end < 10:
                    d_end = '0'+str(d_end)


                #Calling grid_section; Now let's grid the data on a ~250 m x 250 m grid
                [REF,REFmasked,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon] = gridding_no_dp(radar)

                #Let's set up the map projection!
                crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

                #Set up our array of latitude and longitude values and transform our data to the desired projection.
                tlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),rlons[0,:,:],rlats[0,:,:])
                tlons = tlatlons[:,:,0]
                tlats = tlatlons[:,:,1]

                #Limit the extent of the map area, must convert to proper coords.
                LL = (cenlon-2.00,cenlat-1.55,ccrs.PlateCarree())
                UR = (cenlon+2.00,cenlat+1.55,ccrs.PlateCarree())
                print(LL)

                #Get data to plot state and province boundaries
                states_provinces = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces_lakes',
                        scale='50m',
                        facecolor='none')
                #Make sure these shapefiles are in the same directory as the script
                fname = 'cb_2017_us_county_20m/cb_2017_us_county_20m.shp'
                fname2 = 'cb_2017_us_state_20m/cb_2017_us_state_20m.shp'
                counties = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
                states = ShapelyFeature(Reader(fname2).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')

                #Create a figure and plot up the initial data and contours for the algorithm
                #fig=plt.figure(n,figsize=(30.,25.))
                fig=plt.figure(n,figsize=(22.,20.))
                ax = plt.subplot(111,projection=ccrs.PlateCarree())
                ax.coastlines('50m',edgecolor='black',linewidth=0.75)
                ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5, linestyle='--')
                ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
                ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
                REFlevels = np.arange(20,73,2)

                #Options for Z backgrounds/contours
                #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_c, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 73)
                #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_ungridded_base, cmap='HomeyerRainbow', vmin = 10, vmax = 73)
                #refp = ax.pcolormesh(rlons_2d, rlats_2d, REFrmasked, cmap=pyart.graph.cm_colorblind.HomeyerRainbow, vmin = 10, vmax = 73)
                refp2 = ax.contour(rlons_2d, rlats_2d, REFmasked, [40], colors='dodgerblue', linewidths=5, zorder=1)
                #refp3 = ax.contour(rlons_2d, rlats_2d, REFmasked, [45], color='r')
                #plt.contourf(rlons_2d, rlats_2d, ZDR_sum_stuff, depth_levels, cmap=plt.cm.viridis)

                #Storm tracking algorithm starts here
                #Reflectivity smoothed for storm tracker
                smoothed_ref = ndi.gaussian_filter(REFmasked, sigma = 3, order = 0)
                #1st Z contour plotted
                refc = ax.contour(rlons[0,:,:],rlats[0,:,:],smoothed_ref,REFlev, alpha=.01)

                #Set up projection for area calculations
                proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                           pyproj.Proj("+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55"))

                #Main part of storm tracking algorithm starts by looping through all contours looking for Z centroids
                #This method for breaking contours into polygons based on this stack overflow tutorial:
                #https://gis.stackexchange.com/questions/99917/converting-matplotlib-contour-objects-to-shapely-objects
                #Calling stormid_section
                [storm_ids,max_lons_c,max_lats_c,ref_areas,storm_index, alg_speeds, alg_directions] = storm_objects_new(refc,proj,REFlev,REFlev1,big_storm,numsecs,smoothed_ref,ax,rlons,rlats,storm_index,tracking_index,scan_index,tracks_dataframe, track_dis, time_start)

                #Setup tracking index for storm of interest
                tracking_ind=np.where(np.asarray(storm_ids)==storm_to_track)[0]
                max_lons_c = np.asarray(max_lons_c)
                max_lats_c = np.asarray(max_lats_c)
                ref_areas = np.asarray(ref_areas)

                plt.contour(ungrid_lons, ungrid_lats, range_2d, levels=[50000, 100000, 150000], linewidths=6, colors='gold', zorder=-1)
                plt.savefig('testfig.png', bbox_inches='tight')
                print('Testfig Saved')

                if len(max_lons_c) > 0:
                    storm_times = []
                    for l in range(len(max_lons_c)):
                        storm_times.append((time_start))
                    tracking_index = tracking_index + 1
                    #Get storm motion deltas:
                    u_B, v_B = wind_components(Bunkers_s*units('m/s'), Bunkers_m*units('degree'))
                    u_alg, v_alg = wind_components(alg_speeds*units('m/s'), alg_directions*units('degree'))
                    print(u_B, v_B, 'Bunkers motion components')
                    print(u_alg, v_alg, 'Observed motion components')
                    u_diff = u_alg-u_B
                    v_diff = v_alg-v_B
                    motion_delta = np.sqrt(u_diff**2 + v_diff**2).magnitude
                    #Create code to calculate the deviation from the mean wind
                    #Calculate the difference between observed motion and the mean wind
                    meanw_u, meanw_v = wind_components(meanw_s*units('m/s'), meanw_m*units('degree'))
                    shear_dir1 = shear_dir + 90
                    if shear_dir1 > 360:
                        shear_dir1 = shear_dir1 - 360
                    shear_u, shear_v = wind_components(1*units('m/s'), shear_dir1*units('degree'))
                    meand_u = u_alg - meanw_u
                    meand_v = v_alg - meanw_v
                    #Calculate the scalar projection of the difference between the mean wind and observed storm motion 
                    #onto the vector orthogonal to the bulk shear vector (the off-hodograph deviation)
                    offhodo_devs = (meand_u * shear_u) + (meand_v * shear_v)
                    offhodo_devs = offhodo_devs.magnitude
                #If there are no storms, set everything to empty arrays!
                else:
                    storm_ids = []
                    storm_ids = []
                    alg_speeds = []
                    alg_directions = []
                    motion_delta = []
                    max_lons_c = []
                    max_lats_c = []
                    offhodo_devs = []
                    storm_times = time_start
                #Now record all data in a Pandas dataframe.
                new_cells = pd.DataFrame({
                    'scan': scan_index,
                    'storm_id' : storm_ids,
                    'storm speed' : alg_speeds,
                    'storm direction' : alg_directions,
                    'motion_deltas' : motion_delta,
                    'off-hodograph deviation' : offhodo_devs,
                    'storm_id1' : storm_ids,
                    'storm_lon' : max_lons_c,
                    'storm_lat' : max_lats_c,
                    'times' : storm_times
                })
                new_cells.set_index(['scan', 'storm_id'], inplace=True)
                if scan_index == 0:
                    tracks_dataframe = new_cells
                else:
                    tracks_dataframe = tracks_dataframe.append(new_cells)
                n = n+1
                scan_index = scan_index + 1

                #Plot the consolidated stuff!
                title_plot = plt.title(station+' Radar Reflectivity '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
                                           ' '+str(hour)+':'+str(minute)+' UTC', size = 25)

                ref_centroid_lon = max_lons_c
                ref_centroid_lat = max_lats_c
                if len(max_lons_c) > 0:
                    ax.scatter(max_lons_c,max_lats_c, marker = "o", color = 'k', s = 500, alpha = .6)
                    for i in enumerate(ref_centroid_lon):
                        j = i[0]
                        print("offhodo_devs[j] =", offhodo_devs[j])
                        if math.isnan(offhodo_devs[j]):
                            plt.text(ref_centroid_lon[i[0]]-.012, ref_centroid_lat[i[0]]+.035, "%.1f" %(storm_ids[i[0]]), color='black', size = 15, fontweight='bold')
                        elif (offhodo_devs[j]-7.5) > 0:
                            offhodo_devs_size = int((offhodo_devs[j]-7.5)+20)
                            if offhodo_devs_size > 35:
                                offhodo_devs_size = 35
                            print("offhodo_devs_size =", offhodo_devs_size)
                            plt.text(ref_centroid_lon[i[0]]-.012, ref_centroid_lat[i[0]]+.035, "%.1f" %(storm_ids[i[0]]), color='red', size = offhodo_devs_size, fontweight='bold')
                        else:
                            offhodo_devs_size = int((offhodo_devs[j]-7.5)+20)
                            if offhodo_devs_size < 5:
                                offhodo_devs_size = 5
                            print("offhodo_devs_size =", offhodo_devs_size)
                            plt.text(ref_centroid_lon[i[0]]-.012, ref_centroid_lat[i[0]]+.035, "%.1f" %(storm_ids[i[0]]), color='darkgreen', size = offhodo_devs_size, fontweight='bold')
				#plt.legend(handles=[zdr_outline, kdp_outline, separation_vector, elevation], loc = 3, fontsize = 25)
                plt.savefig('Machine_Learning/DELTA_dev'+station+str(time_start.year)+'-'+str(time_start.month)+'-'+str(day)+'-'+str(hour)+str(minute)+'z_'+str(int(meanw_m))+'degMW_'+str(track_dis)+'km_'+str(numsecs)+'sec_'+str(REFlev)+'dBZ_'+str(REFlev1)+'dBZ.png', bbox_inches='tight')
                print('Figure Saved')
                plt.close()
    plt.show()
    print('Finished running the algorithm!')
    return tracks_dataframe