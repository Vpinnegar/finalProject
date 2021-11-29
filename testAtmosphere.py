import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from scipy import interpolate

height = np.linspace(0, 10000, num=333)

time = np.linspace(0, 24, 1440)

atmos = np.zeros((25, 333))

std = np.random.normal(200, 50, 1)
mean = np.random.normal(2000, 200, 1)
scale = np.random.normal(1000, 50, 1)

for i in range(len(atmos)):
	std += np.random.normal(0, 50, 1)
	while std < 150:
		std += np.random.normal(0, 50, 1)
	mean += np.random.normal(0, 200, 1)
	while mean < 800:
		mean += np.random.normal(0, 200, 1)
	scale += np.random.normal(0, 50, 1)
	atmos[i] = scale * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(height-mean)**2/(2*std**2))

# for i in range(len(atmos)):
# 	std = np.random.normal(300, 3, 1)
# 	mean = np.random.normal(6000, 200, 1)
# 	scale = np.random.normal(100, 50, 1)
# 	atmos[i] += scale * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(height-mean)**2/(2*std**2))

f = interpolate.interp2d(range(0, 25, 1), height, np.transpose(atmos), kind='linear')

atmos_full = f(time, height)

df = pd.DataFrame(data=atmos_full)

sns.set_style('darkgrid', {'xtick.bottom': True, 'ytick.left': True, 'axes.edgecolor': '0.15'})
sns.set_context({'font.size': '14'})

fig = plt.figure(figsize=(17.12, 9.6))
g = sns.heatmap(df, cmap='jet')
plt.gca().invert_yaxis()
g.set_xticks(np.linspace(0,1440,25))
g.set_xticklabels(['0:00','','','','','','6:00','','','','','','12:00','','','','','','18:00','','','','','','24:00'])
g.set_yticks([0, 83.25, 165.5, 249.75, 333])
g.set_yticklabels([0, 2500, 5000, 7500, 10000])

plt.savefig('/home/callum/Desktop/atmosphere.png')
