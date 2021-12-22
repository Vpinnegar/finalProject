from updatedGenerateAtmosphere import randomAtmosphere
from updatedLidarModel import mieBackscatter, rayleighBackscatter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import sys

sns.set_style('darkgrid', {'xtick.bottom': True, 'ytick.left': True, 'axes.edgecolor': '0.15'})
sns.set_context({'font.size': '16'})

#########################################

# A NEW ATMOSPHERE IS GENERATED EVERY TIME

#########################################

## OPTIONS TO BE CHANGED BY USER

atmos_type = 'cloud'		 	# 'cloud' or 'aerosol'

atmos_maxheight = 10000 		# max height in metres, default is 10000
atmos_peakheight = 2000 		# cloud peak height in metres, default is 2000
atmos_initstd = 200				# initial cloud standard deviation in metres, default is 200
atmos_varymean = 200		 	# standard deviation of the cloud mean height variation over time, default is 200
atmos_varystd = 50				# standard deviation of the standard deviation variation over time, default is 50

# CHANGING THE BINS/BINSIZES MAKES THE ATMOSPHERE GENERATION NOT WORK AS WELL, BUT THE CODE IS MUCH FASTER
# IT'S GOOD FOR TESTING BUT ISN'T NECESSARILY SIMILAR TO WHAT YOU'D SEE WITH THE DEFAULT VALUES
num_time_bins = 1440			# default is 1440 bins (binsize = 1 minute, always defined as 1 day total)
height_binsize = 5				# default is 5 metres per bin for both radar and lidar

#########################################

## CONSTANTS (most of these shouldn't be changed)

num_height_bins = int(round(atmos_maxheight / height_binsize))
refractive_lidar = 1.33
refractive_radar = 5 + 2.5j
lidar_wavelength = 5.32e-7
radar_wavelength = 8.60e-3
if atmos_type == 'cloud':
	mean_of_scale = 4e5
	std_of_scale = 50
	refractive_lidar = 1.33
	refractive_radar = 5 + 2.5j
elif atmos_type == 'aerosol':
	mean_of_scale = 4e5
	std_of_scale = 50
	refractive_lidar = 1.00044776
	refractive_radar = 1.084210372
	
#########################################

## BEHIND THE SCENES STUFF, JUST CALLING FUNCTIONS AND PLOTTING

print('Atmosphere generation:')
random_atmosphere, particle_radii = randomAtmosphere(atmos_maxheight, num_height_bins, num_time_bins, mean_of_scale,\
									   atmos_initstd, atmos_varystd, atmos_peakheight, atmos_varymean, atmos_type)

sns.set_context({'font.size': '16'})

plt.figure(1, figsize=(8, 6))
g = sns.heatmap(random_atmosphere, cmap='jet', norm = LogNorm(), cbar_kws={'label': r'$P/T \ [\mathrm{Pa}/\mathrm{K}]$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/atmosphere.png'.format(atmos_type), format='png')

plt.figure(2, figsize=(8, 6))
g = sns.heatmap(particle_radii*1e6, cmap='jet', norm = LogNorm(), cbar_kws={'label': r'$r \ \left[\mathrm{\mu} m\right]$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/radii.png'.format(atmos_type), format='png')

print('Lidar:')
cloud_backscatter_lidar, radii_term_lidar = mieBackscatter(num_time_bins, num_height_bins, random_atmosphere, refractive_lidar, lidar_wavelength, atmos_type, particle_radii)
plt.figure(3, figsize=(8, 6))
g = sns.heatmap(cloud_backscatter_lidar, cmap='jet', norm=LogNorm(), cbar_kws={'label': r'$\beta_\mathrm{lidar} \ [\mathrm{m}^{-1}\mathrm{sr}^{-1}]$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.21, bottom=0.21, right=0.91, top=0.91)
plt.show()
# plt.savefig('{}/lidar_backscatter.png'.format(atmos_type), format='png')

print('Radar (Mie):')
cloud_backscatter_radar_mie, radii_term_radar_mie = mieBackscatter(num_time_bins, num_height_bins, random_atmosphere, refractive_radar, radar_wavelength, atmos_type, particle_radii)
plt.figure(4, figsize=(8, 6))
g = sns.heatmap(cloud_backscatter_radar_mie, cmap='jet', norm=LogNorm(), cbar_kws={'label': r'$\beta_\mathrm{radar, Mie} \ [\mathrm{m}^{-1}\mathrm{sr}^{-1}]$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/radar_backscatter_mie.png'.format(atmos_type), format='png')
 	
print('Radar (Rayleigh):')
cloud_backscatter_radar_rayleigh, radii_term_radar_rayleigh = rayleighBackscatter(num_time_bins, num_height_bins, random_atmosphere, refractive_radar, radar_wavelength, atmos_type, particle_radii)
plt.figure(5, figsize=(8, 6))
g = sns.heatmap(cloud_backscatter_radar_rayleigh, cmap='jet', norm=LogNorm(), cbar_kws={'label': r'$\beta_\mathrm{radar, Rayleigh} \ [\mathrm{m}^{-1}\mathrm{sr}^{-1}]$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/radar_backscatter_rayleigh.png'.format(atmos_type), format='png')

colour_ratio_mie = cloud_backscatter_radar_mie/cloud_backscatter_lidar
colour_ratio_rayleigh = cloud_backscatter_radar_rayleigh/cloud_backscatter_lidar

print('Radius Calculation:')
avg_colour_mie = np.zeros(num_time_bins)
avg_colour_rayleigh = np.zeros(num_time_bins)
avg_radius = np.zeros(num_time_bins)
avg_propto_radius = np.zeros(num_time_bins)
avg_eff_radii = np.zeros(num_time_bins)
for sec in range(num_time_bins):
	sys.stdout.write('\rCalculating time bin {:<4d} of {:<4d}'.format(sec+1, num_time_bins))
	sys.stdout.flush()
	sum_colour_mie = 0
	sum_colour_rayleigh = 0
	sum_radius = 0
	sum_propto_radius = 0
	sum_eff_radii = 0
	num = 0
	for al in range(num_height_bins):
		if random_atmosphere[al,sec] >= 200:
			sum_colour_mie += colour_ratio_mie[al,sec]
			sum_colour_rayleigh += colour_ratio_rayleigh[al,sec]
			sum_radius += particle_radii[al,sec]
			sum_propto_radius += colour_ratio_rayleigh[al,sec]**(1/4)
			sum_eff_radii += (radii_term_radar_rayleigh[al,sec]/radii_term_lidar[al,sec])**(1/4)
			num += 1
	if num > 1:
		avg_colour_mie[sec] = sum_colour_mie/num
		avg_colour_rayleigh[sec] = sum_colour_rayleigh/num
		avg_radius[sec] = sum_radius/num
		avg_propto_radius[sec] = sum_propto_radius/num
		avg_eff_radii[sec] = sum_eff_radii/num
	
print()

plt.figure(6, figsize=(8, 6))
g = sns.heatmap(colour_ratio_mie, cmap='jet', norm = LogNorm(), cbar_kws={'label': r'$\beta_\mathrm{radar, Mie}/\beta_\mathrm{lidar}$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/colour_mie.png'.format(atmos_type), format='png')

plt.figure(7, figsize=(8, 6))
g = sns.heatmap(colour_ratio_rayleigh, cmap='jet', norm = LogNorm(), cbar_kws={'label': r'$\beta_\mathrm{radar, Rayleigh}/\beta_\mathrm{lidar}$'})
plt.gca().invert_yaxis()
g.set_xlabel('Time of Day')
g.set_xticks(np.linspace(0,num_time_bins,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'], rotation=45)
g.set_ylabel('Altitude [m]')
g.set_yticks(np.linspace(0, atmos_maxheight/height_binsize, 5))
g.set_yticklabels(np.linspace(0, atmos_maxheight, 5))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/colour_rayleigh.png'.format(atmos_type), format='png')

df = pd.DataFrame()
df['avg_radius'] = avg_radius * 1e6
df['avg_colour_mie'] = avg_colour_mie
df['avg_colour_rayleigh'] = avg_colour_rayleigh
df['avg_propto_radius'] = avg_propto_radius * 1e6
df['avg_eff_radii'] = avg_eff_radii * 1e6

plt.figure(8, figsize=(8, 6))
g = sns.regplot(x='avg_radius', y='avg_colour_mie', color='#4F2683', scatter_kws={'alpha':0.3, 'edgecolor':'k'}, data=df)
g.set_xlabel(r'Average Radius (Per Time Bin) [$\mathrm{\mu}$m]')
g.set_ylabel(r'Colour Ratio (Mie) (Per Time Bin)')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/colour_comp_mie.png'.format(atmos_type), format='png')

plt.figure(9, figsize=(8, 6))
g = sns.regplot(x='avg_radius', y='avg_colour_rayleigh', color='#4F2683', scatter_kws={'alpha':0.3, 'edgecolor':'k'}, data=df)
g.set_xlabel(r'Average Radius (Per Time Bin) [$\mathrm{\mu}$m]')
g.set_ylabel(r'Colour Ratio (Rayleigh) (Per Time Bin)')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/colour_comp_rayleigh.png'.format(atmos_type), format='png')

plt.figure(10, figsize=(8, 6))
g = sns.regplot(x='avg_radius', y='avg_propto_radius', color='#4F2683', scatter_kws={'alpha':0.3, 'edgecolor':'k'}, data=df)
g.set_xlabel(r'$R_\mathrm{avg}$ [$\mathrm{\mu}$m]')
g.set_ylabel(r'$\left(\beta_\mathrm{radar, Rayleigh}/\beta_\mathrm{lidar}\right)^{1/4}$ (Rayleigh)')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/propto_radius.png'.format(atmos_type), format='png')

plt.figure(11, figsize=(8, 6))
g = sns.regplot(x='avg_radius', y='avg_eff_radii', color='#4F2683', scatter_kws={'alpha':0.3, 'edgecolor':'k'}, data=df)
g.set_xlabel(r'$R_\mathrm{avg}$ [$\mathrm{\mu}$m]')
g.set_ylabel(r'$R_\mathrm{eff}$ [$\mathrm{\mu}$m]')
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
plt.show()
# plt.savefig('{}/eff_radii.png'.format(atmos_type), format='png')