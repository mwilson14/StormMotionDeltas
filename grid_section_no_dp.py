import pyart
import numpy as np
import numpy.ma as ma

def gridding_no_dp(radar):
	#Inputs variables,
	#radar: Quality-controlled volume data	
    print('Grid Section')
    #Create grid of data on a 493m x 493m grid (500x500 array)
	#Right now it's set up to grid the radar data onto a 492 km by 492 km box 
	#centered on the radar site with a 1000 x 1000 - point grid with a grid 
	#spacing of 493 m. The arrays in grid_shape and grid_limits are set up in 
	#(z, y, x) order, and the grid_limits settings in x and y are for half of 
	#the actual grid dimension on either side of the radar. Modifying the extent 
	#of the area gridded can get a bit messy since you'll want to keep the same 
	#ratio between the dimensions in grid_limits and the number of points in 
	#grid_shape to keep the grid spacing the same. (1000,1000) (-246, 246)
    gatefilter1 = pyart.filters.GateFilter(radar)
    gatefilter1.exclude_below('reflectivity', 10.0)
    grid = pyart.map.grid_from_radars(
    	(radar,),
    	grid_shape=(1, 750, 750),
    	grid_limits=((200, 200), (-184000.0, 184000.0), (-184000.0, 184000.0)),
    	fields=['reflectivity'],
    	weighting_function='Barnes',
        gatefilters=gatefilter1)
    
    REF = grid.fields['reflectivity']['data'][0,:,:]
    REFmasked = ma.masked_where(REF < 20, REF)

    #Create 2D coordinate arrays used for tracking/place files
    rlons = grid.point_longitude['data']
    rlats = grid.point_latitude['data']
    rlons_2d = rlons[0,:,:]
    rlats_2d = rlats[0,:,:]
    cenlat = radar.latitude['data'][0]
    cenlon = radar.longitude['data'][0]

    #Returning variables,
    #REF: 1km Reflectivity grid
    #REFmasked: REF masked below 20 dBz
    #rlons,rlats: Full volume geographic coordinates, longitude and latitude respectively
    #rlons_2d,rlats_2d: Single layer slice of rlons,rlats
    #cenlat,cenlon: Radar latitude and longitude
    return REF,REFmasked,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon