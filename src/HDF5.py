#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:07:10 2019

@author: rainfall
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:25:48 2019

@author: rainfall
"""
	import os
	import glob
	import numpy as np
	import h5py
	import pandas as pd

def read_hdf5_2BCMB(self):
	
	#start of the program
	print("starting the conversion from hdf5 to DatFrame (CSV):")
	print("")
	
	#list of methods 
	#this method will print all of the names of the hdf internal files
	print("defining methods")
	def printname(name):
		print(name)
	print("method definitions complete")
	print("")
	
	#assign current working directory
	dir=os.getcwd()
	print("the current directory is: "+dir)
	print("")

	#make directory folder (if it does not already exist) and directory variable for output text files
	print("creating a directory for output CSV files")
	if not os.path.exists(dir+"/"+"2BCMB_csv/"):
		os.makedirs(dir+"/"+"2BCMB_csv/")
	csvdir=dir+"/"+"2BCMB_csv"
	print("text file directory created")
	print("")
	
	#list of hdf files to be converted
	print("list of hdf files")
	hdflist=glob.glob(os.path.join('*.HDF5'))
	print(hdflist)
	print("")
	
	#available datasets in hdf files
	print("available datasets in HDF5 files: ")
	singlehdflist=hdflist[0]
	insidehdffile=h5py.File(singlehdflist,"r+")
	insidehdffile.visit(printname)
	insidehdffile.close()
	print("")
	
	#datatype conversion 
	#this loop outputs the indvidual lat long and precip datasets available within the hdf file as indivdual text files 
	for hdffile in hdflist:
		#read and write hdf file
		print("reading the hdf file: "+hdffile)
		currenthdffile=h5py.File(hdffile,"r+")
		print("reading hdf file complete")
		print("")
		
		#data retrieval 
		#This is where you extract the datasets you want to output as text files
		#you can add more variables if you would like
		#this is done in the format varible=hdffilename['dataset']
		print("Creating DF with lat, lon, sfccode, TBs(simulated GMI)!")
		
		lat=currenthdffile['NS/Latitude/']
		lon=currenthdffile['NS/Longitude']
		sfccode=currenthdffile['NS/Input/surfaceType']
		sfcprcp=currenthdffile['NS/surfPrecipTotRate']
		sim_TB=currenthdffile['NS/simulatedBrightTemp']
		
		lat=np.ravel(lat)
		lon=np.ravel(lon)
		sfccode=np.ravel(sfccode)
		sfcprcp=np.ravel(sfcprcp)
		
		TBS=np.array(sim_TB)
		TBS_2D=TBS.transpose(2,0,1).reshape(13,-1)
		TBS=TBS_2D.transpose()
		
		df=pd.DataFrame()
		df['lat']=lat
		df['lon']=lon
		df['sfccode']=sfccode
		df['sfcprcp']=sfcprcp
		df['10V']=TBS[:,0]
		df['10H']=TBS[:,1]
		df['18V']=TBS[:,2]
		df['18H']=TBS[:,3]
		df['23V']=TBS[:,4]
		df['36V']=TBS[:,5]
		df['36H']=TBS[:,6]
		df['89V']=TBS[:,7]
		df['89H']=TBS[:,8]
		df['166V']=TBS[:,9]
		df['166H']=TBS[:,10]
		df['186V']=TBS[:,11]
		df['190V']=TBS[:,11:12]
	
		print("Dataframe created!")
		print("")

		file_name =hdffile[0:60]+".csv"
		df.to_csv(os.path.join(dir, file_name), index=False, sep=",", decimal='.')
		print("Dataframe saved!")
				
		#converting to text file
	#	print("converting arrays to text files")
	#	outputlat=txtdir+"/"+hdffile[:-5]+"_lat.txt"
	#	outputlon=txtdir+"/"+hdffile[:-5]+"_lon.txt"
	#	outputprecip=txtdir+"/"+hdffile[:-5]+"_precip.txt"
	#	outputsfccode=txtdir+"/"+hdffile[:-5]+ "_sfccode.txt"
	#	outputTB=txtdir+"/"+hdffile[:-5]+ "_TB.txt"
	#	np.savetxt(outputlat,latitude,fmt='%f')
	#	np.savetxt(outputlon,longitude,fmt='%f')
	#	np.savetxt(outputprecip,precipitation,fmt='%f')
	#	print("")






def PrepareGMItoML(df_1C,df_2A):
    
    path_csv = '/media/DATA/tmp/git-repositories/validation/clip/'
    file_csv = 'teste_manual_195.csv'
    df_pred = pd.read_csv(os.path.join(path_csv, file_csv), sep=',', decimal='.')
    
    df1.columns = ['lat_s1','lon_s1','10V','10H','18V','18H','23V','36V','36H','89V','89H','lat_s2','lon_s2','166V','166H','186V','190V']
    df2.columns = ['lat_s1','lon_s1','most_like_precip_s1','prob_precip_s1','sfcprcp_s1','sfccode','temp2m','tcwv']
    df3.columns = ['lat_s1','lon_s1','10V','10H','18V','18H','23V','36V','36H','89V','89H','lat_s2','lon_s2','166V','166H','186V','190V']
    
    df1_lat_sa2 = df1.pop('lat_s2')
    df1_lon_s2 = df1.pop('lon_s2')
    df2_lat_s1 = df2.pop('lat_s1')
    df2_lon_s1 = df2.pop('lon_s1')
    
    df=pd.DataFrame()
    df=df1.join(df2, how='right')
    
    df['10VH'] = df['10V'] - df['10H']
    df['18VH'] = df['18V'] - df['18H']
    df['36VH'] = df['36V'] - df['36H']
    df['89VH'] = df['89V'] - df['89H']
    df['166VH'] = df['166V'] - df['166H']
    df['183VH'] = df['186V'] - df['190V']
    df['SSI'] = df['18V'] - df['36V']
    df['delta_neg'] = df['18V'] - df['18H']
    df['delta_pos'] = df['18V'] + df['18H']
    df['MPDI'] = np.divide(df['delta_neg'], df['delta_pos'])
    df['MPDI_scaled'] = df['MPDI']*600
    
    # Inclugin the PCT formulae: PCTf= (1+alfa)*TBfv - alfa*TBfh
    alfa10 = 1.5
    alfa18 = 1.4
    alfa36 = 1.15
    alfa89 = 0.7
    
    df['PCT10'] = (1+ alfa10)*df['10V'] - alfa10*df['10H']
    df['PCT18'] = (1+ alfa18)*df['18V'] - alfa18*df['18H']
    df['PCT36'] = (1+ alfa36)*df['36V'] - alfa36*df['36H']
    df['PCT89'] = (1+ alfa89)*df['89V'] - alfa89*df['89H']
    
    rain_pixels = np.where((df['sfcprcp_s1'] >= 0.1))
    norain_pixels = np.where((df['sfcprcp_s1'] < 0.1))
    df['TagRain'] =""
    df['TagRain'].iloc[rain_pixels] = 1
    df['TagRain'].iloc[norain_pixels] = 0
         
            
            file_name = "teste_manual_195.csv"
            df.to_csv(os.path.join(path, file_name), index=False, sep=",", decimal='.')
            print("The file ", file_name, " was genetared!")
        if not df.empty:
            return df
        else:
            print("Unexpected file format: {} - Skipping...".format(file))
            return None

