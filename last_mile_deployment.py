import numpy 
import pylab 
import random 
import scipy
import fiona 
import pandas as pd 
import geopandas as gp 
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, box ,MultiPoint,MultiLineString 
import seaborn as sns 
from datetime import datetime
from shapely.ops import nearest_points, split, snap
import shapely.wkt
import shapely.wkb
import requests
from shapely import wkb 
import sys 
import fiona
import mplleaflet as mp 
from random import randint
from time import sleep
from osgeo import ogr
# good tile 545660991 611124804
import os 
import seaborn
from fiona.crs import from_epsg
import xml.etree.ElementTree as et
import gpxpy 
import gpxpy.gpx 
from skimage import data
import numpy as np
from skimage import io
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import feature
from sklearn.cluster import KMeans
import geoplot
from scipy.spatial import distance
from time import sleep
from shapely.ops import split

sphere_of_influence = 0.001
weight=0.1
delta=0.01



def get_shortest_osm(s_lat,s_lon,d_lat,d_lon,delta=0.01):

    destination=Point(d_lat,d_lon)
    source=Point(s_lat,s_lon)
    
    minx=min(d_lon,s_lon) 
    maxx=max(d_lon,s_lon) 
    miny=min(d_lat,s_lat) 
    maxy=max(d_lat,s_lat) 
    URL = r"https://api.openstreetmap.org/api/0.6/trackpoints"
    bbox= str(minx) + "," + str(miny) + "," + str(maxx) + "," + str(maxy)
    PARAMS = {
                'bbox':bbox,# str(minx) + "," + str(maxy) + "," + str(maxx) + "," + str(miny),
            'page':0        
        } 
    
    r = requests.get(url = URL, params = PARAMS)
    
    if r.status_code == 200:
        print('')
    else:
        print('end point unreachable')
        sys.exit()
    
    filename=str(destination) + str(source) + '.gpx'
    text_file = open(filename, "w")
    text_file.write(r.text) 
    text_file.close()
    gpx_file = open(filename, 'r') 
    gpx = gpxpy.parse(gpx_file)
    
    
    osm=gp.GeoDataFrame(crs = {'init': 'epsg:4326'})
    for track in gpx.tracks:
        for segment in track.segments: 
            pointlist=[];    
            for point in segment.points:
                pointlist.append(Point(point.latitude,point.longitude))
            if len(pointlist) > 2  and len(pointlist) < 500 :            
                loo =  LineString(pointlist)
                loo  =  gp.GeoDataFrame(geometry=pd.Series([loo]))
                loo['pointcount']=len(pointlist)
                osm=osm.append(loo,sort=False)
                
    osm['dismetric'] = weight * (osm.distance(source) + osm.distance(destination))  + (1-weight) * osm.length 

    bestroute=gp.GeoDataFrame([osm.iloc[osm.dismetric.idxmin()]])
    #bestroute_df =  gp.GeoDataFrame(geometry=pd.Series([bestroute.geometry]))
    
    #
    if not bestroute.intersects(gp.GeoDataFrame(geometry=pd.Series([LineString(list(source.buffer(sphere_of_influence).exterior.coords))])))[0] or bestroute.intersects(gp.GeoDataFrame(geometry=pd.Series([LineString(list(source.buffer(sphere_of_influence).exterior.coords))])))[0]:
        return osm,bestroute,source,destination
    else:
        return osm,False,source,destination



    
def gen_random_paths(s_lat,s_lon,d_lat,d_lon,delta=0.01,maxwalks=50):
    startx,starty=s_lat,s_lon  
    number_of_walks = random.randint(4,maxwalks)
    incr=0.0001
    random_paths =gp.GeoDataFrame(crs = {'init': 'epsg:4326'})        
    for _ in range(0,number_of_walks):        
        points_in_each_walk = random.randint(10,100)
        x = numpy.zeros(points_in_each_walk) 
        y = numpy.zeros(points_in_each_walk) 
        x[0]=startx
        y[0]=starty  
        # filling the coordinates with random variables 
        for i in range(1, points_in_each_walk):
            
                tempx = x[i - 1] + random.uniform(-1,1)  * incr
                tempy = y[i - 1] + random.uniform(-1,1) * incr
            
                if random.uniform(0,1) > 0.1:
                    if distance.euclidean((x[i - 1] , y[i - 1]),(d_lat,d_lon)) < distance.euclidean((d_lat,d_lon),(tempx,tempy)):
                        tempx=x[i - 1]
                        tempy=y[i - 1]                                                
                x[i] = tempx
                y[i] = tempy
        foo = LineString([Point(a) for a in zip(x,y)])
        foo  =  gp.GeoDataFrame(geometry=pd.Series([foo]))
        foo['timetaken']=random.randint(0,1000)
        foo['counts']=points_in_each_walk
        random_paths= random_paths.append(foo)
    return random_paths


def get_shortest_random(s_lat,s_lon,d_lat,d_lon):
    
    destination=Point(d_lat,d_lon)
    source=Point(s_lat,s_lon)
    random_paths_all=gen_random_paths(s_lat,s_lon,d_lat,d_lon,delta,100)
    #random_paths.to_file('good_rand_path.json')
    random_paths =  random_paths_all[random_paths_all.distance(source) < sphere_of_influence ]
    random_paths =  random_paths[random_paths.distance(destination) < sphere_of_influence ]
    
    #random_paths.plot(column='counts')
    
    
    if random_paths.shape[0] == 0:
        print('')
        return False,False,False,False
        
    random_paths['dismetric'] = weight * (random_paths.distance(source) + random_paths.distance(destination))  + (1-weight) * random_paths.length 
    #random_paths.dismetric.idxmin()
    bestroute = gp.GeoDataFrame([random_paths.iloc[random_paths.dismetric.idxmin()]])
    
    
    src_intersection = bestroute.intersection(source.buffer(sphere_of_influence))
    src_intersection_c=gp.GeoDataFrame(geometry=pd.Series(src_intersection.centroid[0]))
    
    dest_intersection = bestroute.intersection(destination.buffer(sphere_of_influence))
    dest_intersection_c=gp.GeoDataFrame(geometry=pd.Series(dest_intersection.centroid[0]))
    
    #handle = gp.GeoDataFrame(geometry=pd.Series([LineString(list(source.buffer(sphere_of_influence).exterior.coords))])).plot()
    #src_intersection_c.plot(ax=handle)
    #
    #handle = gp.GeoDataFrame(geometry=pd.Series([LineString(list(destination.buffer(sphere_of_influence).exterior.coords))])).plot()
    #dest_intersection_c.plot(ax=handle)
    
    
    
    minx,miny,maxx,maxy  = LineString([src_intersection_c.geometry[0],  dest_intersection_c.geometry[0]]).bounds
    cutter  = box(minx,miny,maxx,maxy)
    cutter   =  gp.GeoDataFrame(geometry=[cutter])
    
    
    finalpath  = gp.GeoDataFrame(geometry=[bestroute.intersection(cutter).geometry[0]])
    
    #handle=bestroute.plot()
    #src_intersection_c.plot(ax=handle, color='green')
    #dest_intersection_c.plot(ax=handle, color='red')
    return random_paths,bestroute,src_intersection_c,dest_intersection_c


def smooth_linestring(linestring, smooth_sigma):
    smooth_x = np.array(scipy.ndimage.filters.gaussian_filter1d(
        linestring.xy[0],
        smooth_sigma)
        )
    smooth_y = np.array(scipy.ndimage.filters.gaussian_filter1d(
        linestring.xy[1],
        smooth_sigma)
        )
    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)
    linestring_smoothed = LineString(smoothed_coords)
    return linestring_smoothed


if __name__ == '__main__':
    
    if len(sys.argv) < 5:
        print('enter lat1 lon1 lat2 lon2 ')
        sys.exit()     
        
    d_lat,d_lon,s_lat,s_lon=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
#    d_lat,d_lon=19.154257, 72.856066
#    s_lat,s_lon=19.15284496, 72.85811249
    allroutes,bestroute,src_intersection_c,dest_intersection_c  =  get_shortest_osm(s_lat,s_lon,d_lat,d_lon)
    if not bestroute.__class__ == bool:
        allroutes,bestroute,src_intersection_c,dest_intersection_c  =  get_shortest_random(float(s_lat),float(s_lon),float(d_lat),float(d_lon))
    if not bestroute.__class__ == bool:
        bestroute.geometry[0] = smooth_linestring(bestroute.geometry[0],2)
        #bestroute.plot()
        print(bestroute.to_json())
        foo  =  open('output.json','w')
        foo.write(bestroute.to_json())
        foo.close()
    else:
         print('{no route found}')
        
        
#    allroutes,bestroute,src_intersection_c,dest_intersection_c  =  get_shortest_random(float(s_lat),float(s_lon),float(d_lat),float(d_lon))
#    if not allroutes.__class__ == bool:
#        bestroute.geometry[0] = smooth_linestring(bestroute.geometry[0],2)
#        bestroute.plot()
#        
#    else:
#        print('{no route found}')
    
            #handle=allroutes.plot(column='counts')
            #src_intersection_c.plot(ax=handle, color='green')
            #dest_intersection_c.plot(ax=handle, color='red')
    
    


#    handle=gp.GeoDataFrame([]).plot()
#    while True:
#        sleep(randint(1,2))
#        allroutes,bestroute,src_intersection_c,dest_intersection_c  =  get_shortest_random(s_lat,s_lon,d_lat,d_lon)
#        if not allroutes.__class__  == bool:
#            bestroute.geometry[0] = smooth_linestring(bestroute.geometry[0],2)
#            bestroute.plot(ax=handle,column='counts')
#            print('route found',bestroute.counts)
#        else:
#            print('no route found')
#            #handle=allroutes.plot(column='counts')
#            #src_intersection_c.plot(ax=handle, color='green')
#            #dest_intersection_c.plot(ax=handle, color='red')
#        
##        sleep(randint(1,5))
##        print('findind new route from source : ',s_lat,s_lon,' to destination :',d_lat,d_lon)
##        bestroute,src_intersection_c,dest_intersection_c  =  get_shortest_osm(s_lat,s_lon,d_lat,d_lon)
##        if not bestroute:
##            print('generating random path')
##            bestroute,src_intersection_c,dest_intersection_c  =  get_shortest_random(s_lat,s_lon,d_lat,d_lon)
##        
#        #handle=bestroute.plot()
#        #src_intersection_c.plot(ax=handle, color='green')
#        #dest_intersection_c.plot(ax=handle, color='red')
#        
#        d_lat= d_lat + random.uniform(-1,1)* 0.0001
#        s_lat= s_lat + random.uniform(-1,1)* 0.0001
#        
#        d_lon= d_lon + random.uniform(-1,1)* 0.0001
#        d_lon= d_lon + random.uniform(-1,1)* 0.0001


#gp.GeoDataFrame(geometry=pd.Series([newgeom])).plot(ax=handle,color='red')
# smooth_linestring(bestroute.geometry[0],2)
        
