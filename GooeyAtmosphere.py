from tkinter import *
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LogNorm
import seaborn as sns

from testAtmosphere import RandomBackscatter
from LidarModel import *

######################################################################################################################################################################################################
# Setup

root = Tk()
root.title('Gooey Atmosphere')
# root.geometry('1920x1080')

mainframe = ttk.Frame(root, padding=10)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

######################################################################################################################################################################################################
# Function to switch which parameters are visible
def swapParams(*args):
	if selection.get() == 'lidar':
		lidar_frame.grid(column=1, row=3, columnspan=9)
		radar_frame.grid_forget()
		atmos_frame.grid_forget()

	elif selection.get() == 'radar':
		radar_frame.grid(column=1, row=3, columnspan=9)
		lidar_frame.grid_forget()
		atmos_frame.grid_forget()

	elif selection.get() == 'atmosphere':
		atmos_frame.grid(column=1, row=3, columnspan=9)
		lidar_frame.grid_forget()
		radar_frame.grid_forget()

######################################################################################################################################################################################################
# Creating the frames and the corresponding parameters

# LIDAR ######################################################################################44444
lidar_frame = ttk.Frame(mainframe, borderwidth=5, relief='ridge', padding=10)
lidar_info1_label = ttk.Label(lidar_frame, text='High Resolution Spectral Lidar (HRSL)').grid(column=1, row=3)
lidar_info2_label = ttk.Label(lidar_frame, text='Wavelength: 532nm').grid(column=1, row=4)
lidar_info3_label = ttk.Label(lidar_frame, text='See README for More Details').grid(column=1, row=5)

# RADAR ######################################################################################44444
radar_frame = ttk.Frame(mainframe, borderwidth=5, relief='ridge', padding=10)
radar_scattering = StringVar(value='mie')
rayleigh = ttk.Radiobutton(radar_frame, text='Rayleigh Scattering', variable=radar_scattering, value='rayleigh', command=swapParams).grid(column=1, row=3, sticky=E, padx=(0,50))
mie = ttk.Radiobutton(radar_frame, text='Mie Scattering', variable=radar_scattering, value='mie', command=swapParams).grid(column=2, row=3, sticky=W)
radar_info1_label = ttk.Label(radar_frame, text='Millimeter-Wave Cloud Radar (MWCR)').grid(column=1, row=4, columnspan=2)
radar_info2_label = ttk.Label(radar_frame, text='Wavelength: 8.6mm').grid(column=1, row=5, columnspan=2)
radar_info3_label = ttk.Label(radar_frame, text='See README for More Details').grid(column=1, row=6, columnspan=2)

# ATMOSHPERE ######################################################################################
atmos_frame = ttk.Frame(mainframe, borderwidth=5, relief='ridge', padding=10)

atmos_type = StringVar(value='cloud')
cloud = ttk.Radiobutton(atmos_frame, text='Water Cloud', variable=atmos_type, value='cloud').grid(column=1, row=2, columnspan=2, sticky=E)
aerosol = ttk.Radiobutton(atmos_frame, text='Aerosol Cloud (Methane)', variable=atmos_type, value='aerosol').grid(column=5, row=2, columnspan=2, sticky=W)

atmos_maxheight = StringVar(value=10000)
atmos_maxheight_label = ttk.Label(atmos_frame, text='Max Atmosphere Height [m]: ').grid(column=1, row=3, rowspan=2, sticky=E)
atmos_maxheight_entry = ttk.Entry(atmos_frame, width=7, textvariable=atmos_maxheight).grid(column=2, row=3, rowspan=2, sticky=W, padx=(0,50))

atmos_peakheight = StringVar(value=2000)
atmos_peakheight_label = ttk.Label(atmos_frame, text='Mean Initial Cloud Height [m]: ').grid(column=3, row=3, sticky=E)
atmos_peakheight_entry = ttk.Entry(atmos_frame, width=7, textvariable=atmos_peakheight).grid(column=4, row=3, sticky=W, padx=(0,50))

atmos_initstd = StringVar(value=200)
atmos_initstd_label = ttk.Label(atmos_frame, text='Inital Height Standard Deviation [m]: ').grid(column=5, row=3, sticky=E)
atmos_initstd_entry = ttk.Entry(atmos_frame, width=7, textvariable=atmos_initstd).grid(column=6, row=3, sticky=W, padx=(0,50))

atmos_varymean = StringVar(value=200)
atmos_varymean_label = ttk.Label(atmos_frame, text='Time Variation of Height [m]: ').grid(column=3, row=4, sticky=E)
atmos_varymean_entry = ttk.Entry(atmos_frame, width=7, textvariable=atmos_varymean).grid(column=4, row=4, sticky=W, padx=(0,50))

atmos_varystd = StringVar(value=50)
atmos_varystd_label = ttk.Label(atmos_frame, text='Time Variation of Standard Deviation [m]: ').grid(column=5, row=4, sticky=E)
atmos_varystd_entry = ttk.Entry(atmos_frame, width=7, textvariable=atmos_varystd).grid(column=6, row=4, sticky=W, padx=(0,50))

atmos_newatmos = BooleanVar(value=True)
check = ttk.Checkbutton(atmos_frame, text='Create New Atmosphere', variable=atmos_newatmos, onvalue=True, offvalue=False).grid(column=1, row=5, columnspan=6, pady=(10,0))

######################################################################################################################################################################################################
# Buttons to change between each set of parameters
testlabel = ttk.Label(mainframe, text='Select the parameters you wish to change:').grid(column=1, row=1, columnspan=9)
selection = StringVar()
lidar = ttk.Radiobutton(mainframe, text='Lidar Parameters', variable=selection, value='lidar', command=swapParams).grid(column=1, row=2, columnspan=3, sticky=E)
radar = ttk.Radiobutton(mainframe, text='Radar Parameters', variable=selection, value='radar', command=swapParams).grid(column=4, row=2, columnspan=3)
atmosphere = ttk.Radiobutton(mainframe, text='Atmosphere Parameters', variable=selection, value='atmosphere', command=swapParams).grid(column=7, row=2, columnspan=3, sticky=W)

######################################################################################################################################################################################################
# Initial empty plots displayed

fig1 = plt.figure(1, figsize=(6.4, 4.8))
plt.plot()
plt.title('Lidar')

fig2 = plt.figure(2, figsize=(6.4, 4.8))
plt.plot()
plt.title('Radar')

fig3 = plt.figure(3, figsize=(6.4, 4.8))
plt.plot()
plt.title('Colour Ratio')

######################################################################################################################################################################################################

num_time_bins = 1440
height_binsize = 5
num_height_bins = int(round(float(atmos_maxheight.get()) / height_binsize))
refractive_lidar = 1.33
refractive_radar = 5 + 2.5j
lidar_wavelength = 5.32e-7
radar_wavelength = 8.60e-3
mean_of_scale = 4e5
std_of_scale = 50
mean_of_std = float(atmos_initstd.get())
std_of_std = float(atmos_varystd.get())
mean_of_mean = float(atmos_peakheight.get())
std_of_mean = float(atmos_varymean.get())
random_atmosphere = RandomBackscatter(int(atmos_maxheight.get()), num_height_bins, mean_of_scale, std_of_scale, mean_of_std, std_of_std, mean_of_mean, std_of_mean)

def plotFigs():

	sns.set_context({'font.size': '16'})
	
	try:
	
		global random_atmosphere
		global refractive_lidar
		global refractive_radar
		
		num_height_bins = int(round(float(atmos_maxheight.get()) / height_binsize))
		if atmos_newatmos.get():
			if atmos_type.get() == 'cloud':
				mean_of_scale = 4e5
				std_of_scale = 50
			elif atmos_type.get() == 'aerosol':
				mean_of_scale = 4e5
				std_of_scale = 50
			mean_of_std = float(atmos_initstd.get())
			std_of_std = float(atmos_varystd.get())
			mean_of_mean = float(atmos_peakheight.get())
			std_of_mean = float(atmos_varymean.get())
			
			if atmos_type.get() == 'cloud':
				refractive_lidar = 1.33
				refractive_radar = 5 + 2.5j
			elif atmos_type.get() == 'aerosol':
				refractive_lidar = 1.00044776
				refractive_radar = 1.084210372
			
			random_atmosphere = RandomBackscatter(int(atmos_maxheight.get()), num_height_bins, mean_of_scale, std_of_scale, mean_of_std, std_of_std, mean_of_mean, std_of_mean)
	
			if atmos_type.get() == 'cloud' and radar_scattering.get() == 'mie':
				vmin_lidar, vmax_lidar = 1e15, 1e20
				vmin_radar, vmax_radar = 1e13, 1e18
				vmin_ratio, vmax_ratio = 1e-10, 1e-1
			elif atmos_type.get() == 'cloud' and radar_scattering.get() == 'rayleigh':
				vmin_lidar, vmax_lidar = 1e15, 1e20
				vmin_radar, vmax_radar = 1e9, 1e14
				vmin_ratio, vmax_ratio = 1e-9, 1e-6
			elif atmos_type.get() == 'aerosol' and radar_scattering.get() == 'mie':
				vmin_lidar, vmax_lidar = 1e-0, 1e5
				vmin_radar, vmax_radar = 1e-13, 1e-7
				vmin_ratio, vmax_ratio = 1e-16, 1e-11
			elif atmos_type.get() == 'aerosol' and radar_scattering.get() == 'rayleigh':
				vmin_lidar, vmax_lidar = 1e-0, 1e5
				vmin_radar, vmax_radar = 1e-13, 1e-7
				vmin_ratio, vmax_ratio = 1e-16, 1e-11
	
		print('Lidar:')
		cloud_backscatter_lidar = MieBackscatter(num_time_bins, num_height_bins, random_atmosphere, refractive_lidar, lidar_wavelength, atmos_type.get())
		
		plt.figure(1, figsize=(7, 5))
		plt.clf()
		g = sns.heatmap(cloud_backscatter_lidar, cmap='jet', norm=LogNorm(vmin = vmin_lidar, vmax = vmax_lidar), cbar_kws={'label': r'$\beta_\mathrm{lidar}$'})
		plt.gca().invert_yaxis()
		g.set_xlabel('Time of Day')
		g.set_xticks(np.linspace(0,1440,25))
		g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
		g.set_ylabel('Altitude [m]')
		g.set_yticks(np.linspace(0, int(atmos_maxheight.get())/height_binsize, 5))
		g.set_yticklabels(np.linspace(0, int(atmos_maxheight.get()), 5))
		plt.title('Lidar')
		plt.subplots_adjust(left=0.21, bottom=0.21, right=0.91, top=0.91)
		fig1.canvas.draw()
		
		##############
	
		if radar_scattering.get() == 'mie':
			print('Radar:')
			cloud_backscatter_radar = MieBackscatter(num_time_bins, num_height_bins, random_atmosphere, refractive_radar, radar_wavelength, atmos_type.get())
	
			plt.figure(2, figsize=(7, 5))
			plt.clf()
			g = sns.heatmap(cloud_backscatter_radar, cmap='jet', norm=LogNorm(vmin = vmin_radar, vmax = vmax_radar), cbar_kws={'label': r'$\beta_\mathrm{radar}$'})
			plt.gca().invert_yaxis()
			g.set_xlabel('Time of Day')
			g.set_xticks(np.linspace(0,1440,25))
			g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
			g.set_ylabel('Altitude [m]')
			g.set_yticks(np.linspace(0, int(atmos_maxheight.get())/height_binsize, 5))
			g.set_yticklabels(np.linspace(0, int(atmos_maxheight.get()), 5))
			plt.title('Radar')
			plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
			fig2.canvas.draw()
			
		elif radar_scattering.get() == 'rayleigh':
			print('Radar:')
			cloud_backscatter_radar = RadBackscatter(num_time_bins, num_height_bins, random_atmosphere, refractive_radar, radar_wavelength, atmos_type.get())
	
			plt.figure(2, figsize=(7, 5))
			plt.clf()
			g = sns.heatmap(cloud_backscatter_radar, cmap='jet', norm=LogNorm(vmin = vmin_radar, vmax = vmax_radar), cbar_kws={'label': r'$\beta_\mathrm{radar}$'})
			plt.gca().invert_yaxis()
			g.set_xlabel('Time of Day')
			g.set_xticks(np.linspace(0,1440,25))
			g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
			g.set_ylabel('Altitude [m]')
			g.set_yticks(np.linspace(0, int(atmos_maxheight.get())/height_binsize, 5))
			g.set_yticklabels(np.linspace(0, int(atmos_maxheight.get()), 5))
			plt.title('Radar')
			plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
			fig2.canvas.draw()
			
		##############
	
		plt.figure(3, figsize=(7, 5))
		plt.clf()
		g = sns.heatmap(cloud_backscatter_radar/cloud_backscatter_lidar, cmap='jet', norm = LogNorm(vmin = vmin_ratio, vmax = vmax_ratio), cbar_kws={'label': r'$\beta_\mathrm{radar}/\beta_\mathrm{lidar}$'})
		plt.gca().invert_yaxis()
		g.set_xlabel('Time of Day')
		g.set_xticks(np.linspace(0,1440,25))
		g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
		g.set_ylabel('Altitude [m]')
		g.set_yticks(np.linspace(0, int(atmos_maxheight.get())/height_binsize, 5))
		g.set_yticklabels(np.linspace(0, int(atmos_maxheight.get()), 5))
		plt.title('Colour Ratio')
		plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
		fig3.canvas.draw()
		
		print()
		
	except Exception:
		print('An Error occurred. Check your input and try again.')
		pass
	
test_frame = ttk.Frame(mainframe, borderwidth=5, relief='flat', padding=10)
test_frame.grid(column=1, row=6, columnspan=9, sticky=W)

ttk.Button(test_frame,text="Plot",command=plotFigs).grid(column=2, row=7)

plot_widget1 = FigureCanvasTkAgg(fig1, master=test_frame).get_tk_widget().grid(column=1, row=8)
plot_widget2 = FigureCanvasTkAgg(fig2, master=test_frame).get_tk_widget().grid(column=2, row=8)
plot_widget3 = FigureCanvasTkAgg(fig3, master=test_frame).get_tk_widget().grid(column=3, row=8)


######################################################################################################################################################################################################
# Run the GUI

root.mainloop()
