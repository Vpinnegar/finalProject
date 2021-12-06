from tkinter import *
from tkinter import ttk

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

######################################################################################################################################################################################################
# Setup

root = Tk()
root.title('gooey')
# root.geometry('1920x1080')

mainframe = ttk.Frame(root, padding=10)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

######################################################################################################################################################################################################
# Inital parameter setup

lidar_power = StringVar(value=1)
lidar_binlength = StringVar(value=2)
lidar_test1 = StringVar(value=3)
lidar_test2 = StringVar(value=4)
lidar_test3 = StringVar(value=5)
lidar_test4 = StringVar(value=6)
lidar_test5 = StringVar(value=7)
lidar_test6 = StringVar(value=8)

radar_power = StringVar(value=9)
radar_binlength = StringVar(value=10)
radar_test1 = StringVar(value=11)
radar_test2 = StringVar(value=12)
radar_test3 = StringVar(value=13)
radar_test4 = StringVar(value=14)
radar_test5 = StringVar(value=15)
radar_test6 = StringVar(value=16)

atmos_test1 = StringVar(value=17)
atmos_test2 = StringVar(value=18)
atmos_test3 = StringVar(value=19)
atmos_test4 = StringVar(value=20)
atmos_test5 = StringVar(value=21)
atmos_test6 = StringVar(value=22)
atmos_test7 = StringVar(value=23)
atmos_test8 = StringVar(value=24)

######################################################################################################################################################################################################
# Creating the frames and the corresponding parameters

# LIDAR
lidar_input_frame = ttk.Frame(mainframe, borderwidth=5, relief='ridge', padding=10)
lidar_power_label = ttk.Label(lidar_input_frame, text='Lidar Power [units]: ').grid(column=1, row=3, sticky=E)
lidar_power_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_power).grid(column=2, row=3, sticky=W, padx=(0,50))
lidar_binlength_label = ttk.Label(lidar_input_frame, text='Bin Size [units]: ').grid(column=1, row=4, sticky=E)
lidar_binlength_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_binlength).grid(column=2, row=4, sticky=W, padx=(0,50))

lidar_test1_label = ttk.Label(lidar_input_frame, text='Test 1 [units]: ').grid(column=3, row=3, sticky=E)
lidar_test1_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_test1).grid(column=4, row=3, sticky=W, padx=(0,50))
lidar_test2_label = ttk.Label(lidar_input_frame, text='Test 2 [units]: ').grid(column=3, row=4, sticky=E)
lidar_test2_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_test2).grid(column=4, row=4, sticky=W, padx=(0,50))

lidar_test3_label = ttk.Label(lidar_input_frame, text='Test 3 [units]: ').grid(column=5, row=3, sticky=E)
lidar_test3_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_test3).grid(column=6, row=3, sticky=W, padx=(0,50))
lidar_test4_label = ttk.Label(lidar_input_frame, text='Test 4 [units]: ').grid(column=5, row=4, sticky=E)
lidar_test4_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_test4).grid(column=6, row=4, sticky=W, padx=(0,50))

lidar_test5_label = ttk.Label(lidar_input_frame, text='Test 5 [units]: ').grid(column=7, row=3, sticky=E)
lidar_test5_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_test5).grid(column=8, row=3, sticky=W, padx=(0,50))
lidar_test6_label = ttk.Label(lidar_input_frame, text='Test 6 [units]: ').grid(column=7, row=4, sticky=E)
lidar_test6_entry = ttk.Entry(lidar_input_frame, width=7, textvariable=lidar_test6).grid(column=8, row=4, sticky=W, padx=(0,50))

# RADAR
radar_input_frame = ttk.Frame(mainframe, borderwidth=5, relief='ridge', padding=10)
radar_power_label = ttk.Label(radar_input_frame, text='Radar Power [units]: ').grid(column=1, row=3, sticky=E)
radar_power_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_power).grid(column=2, row=3, sticky=W, padx=(0,50))
radar_binlength_label = ttk.Label(radar_input_frame, text='Bin Size [units]: ').grid(column=1, row=4, sticky=E)
radar_binlength_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_binlength).grid(column=2, row=4, sticky=W, padx=(0,50))

radar_test1_label = ttk.Label(radar_input_frame, text='Test 1 [units]: ').grid(column=3, row=3, sticky=E)
radar_test1_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_test1).grid(column=4, row=3, sticky=W, padx=(0,50))
radar_test2_label = ttk.Label(radar_input_frame, text='Test 2 [units]: ').grid(column=3, row=4, sticky=E)
radar_test2_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_test2).grid(column=4, row=4, sticky=W, padx=(0,50))

radar_test3_label = ttk.Label(radar_input_frame, text='Test 3 [units]: ').grid(column=5, row=3, sticky=E)
radar_test3_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_test3).grid(column=6, row=3, sticky=W, padx=(0,50))
radar_test4_label = ttk.Label(radar_input_frame, text='Test 4 [units]: ').grid(column=5, row=4, sticky=E)
radar_test4_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_test4).grid(column=6, row=4, sticky=W, padx=(0,50))

radar_test5_label = ttk.Label(radar_input_frame, text='Test 5 [units]: ').grid(column=7, row=3, sticky=E)
radar_test5_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_test5).grid(column=8, row=3, sticky=W, padx=(0,50))
radar_test6_label = ttk.Label(radar_input_frame, text='Test 6 [units]: ').grid(column=7, row=4, sticky=E)
radar_test6_entry = ttk.Entry(radar_input_frame, width=7, textvariable=radar_test6).grid(column=8, row=4, sticky=W, padx=(0,50))

# ATMOSHPERE
atmos_input_frame = ttk.Frame(mainframe, borderwidth=5, relief='ridge', padding=10)
atmos_test1_label = ttk.Label(atmos_input_frame, text='Test 1 [units]: ').grid(column=1, row=3, sticky=E)
atmos_test1_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test1).grid(column=2, row=3, sticky=W, padx=(0,50))
atmos_test2_label = ttk.Label(atmos_input_frame, text='Test 2 [units]: ').grid(column=1, row=4, sticky=E)
atmos_test2_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test2).grid(column=2, row=4, sticky=W, padx=(0,50))

atmos_test3_label = ttk.Label(atmos_input_frame, text='Test 3 [units]: ').grid(column=3, row=3, sticky=E)
atmos_test3_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test3).grid(column=4, row=3, sticky=W, padx=(0,50))
atmos_test4_label = ttk.Label(atmos_input_frame, text='Test 4 [units]: ').grid(column=3, row=4, sticky=E)
atmos_test4_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test4).grid(column=4, row=4, sticky=W, padx=(0,50))

atmos_test5_label = ttk.Label(atmos_input_frame, text='Test 5 [units]: ').grid(column=5, row=3, sticky=E)
atmos_test5_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test5).grid(column=6, row=3, sticky=W, padx=(0,50))
atmos_test6_label = ttk.Label(atmos_input_frame, text='Test 6 [units]: ').grid(column=5, row=4, sticky=E)
atmos_test6_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test6).grid(column=6, row=4, sticky=W, padx=(0,50))

atmos_test7_label = ttk.Label(atmos_input_frame, text='Test 7 [units]: ').grid(column=7, row=3, sticky=E)
atmos_test7_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test7).grid(column=8, row=3, sticky=W, padx=(0,50))
atmos_test8_label = ttk.Label(atmos_input_frame, text='Test 8 [units]: ').grid(column=7, row=4, sticky=E)
atmos_test8_entry = ttk.Entry(atmos_input_frame, width=7, textvariable=atmos_test8).grid(column=8, row=4, sticky=W, padx=(0,50))

######################################################################################################################################################################################################
# Function to switch which parameters are visible
def swapParams(*args):
	if selection.get() == 'lidar':
		lidar_input_frame.grid(column=1, row=3, columnspan=9)
		radar_input_frame.grid_forget()
		atmos_input_frame.grid_forget()

	elif selection.get() == 'radar':
		radar_input_frame.grid(column=1, row=3, columnspan=9)
		lidar_input_frame.grid_forget()
		atmos_input_frame.grid_forget()

	elif selection.get() == 'atmosphere':
		atmos_input_frame.grid(column=1, row=3, columnspan=9)
		lidar_input_frame.grid_forget()
		radar_input_frame.grid_forget()

######################################################################################################################################################################################################
# Buttons to change between each set of parameters
testlabel = ttk.Label(mainframe, text='Select the parameters you wish to change:').grid(column=1, row=1, columnspan=9)
selection = StringVar()
lidar = ttk.Radiobutton(mainframe, text='Lidar Parameters', variable=selection, value='lidar', command=swapParams).grid(column=1, row=2, columnspan=3, sticky=E)
radar = ttk.Radiobutton(mainframe, text='Radar Parameters', variable=selection, value='radar', command=swapParams).grid(column=4, row=2, columnspan=3)
atmosphere = ttk.Radiobutton(mainframe, text='Atmosphere Parameters', variable=selection, value='atmosphere', command=swapParams).grid(column=7, row=2, columnspan=3, sticky=W)

######################################################################################################################################################################################################
# Initial plots displayed

t = np.arange(0.0,3.0,0.01)
s = np.random.normal(0,1,np.size(t))

fig1 = plt.figure(1)
plt.plot(t, s * np.sin(np.pi*t))
plt.title('Test 1')

fig2 = plt.figure(2)
plt.plot(t, s * np.cos(np.pi*t))
plt.title('Test 2')

fig3 = plt.figure(3)
plt.plot(t, s * np.sin(np.pi*t) * np.cos(np.pi*t))
plt.title('Test 3')

######################################################################################################################################################################################################
# Function and button for updating the plots with the new parameters

def plotFigs():

	s = np.random.normal(0,1,np.size(t))

	plt.figure(1)
	plt.clf()
	plt.plot(t, s * np.sin(np.pi*t))
	plt.title('Test 1')
	fig1.canvas.draw()

	plt.figure(2)
	plt.clf()
	plt.plot(t, s * np.cos(np.pi*t))
	plt.title('Test 2')
	fig2.canvas.draw()

	plt.figure(3)
	plt.clf()
	plt.plot(t, s * np.sin(np.pi*t) * np.cos(np.pi*t))
	plt.title('Test 3')
	fig3.canvas.draw()

test_frame = ttk.Frame(mainframe, borderwidth=5, relief='flat', padding=10)
test_frame.grid(column=1, row=6, columnspan=9, sticky=W)

ttk.Button(test_frame,text="Plot",command=plotFigs).grid(column=2, row=5)

plot_widget1 = FigureCanvasTkAgg(fig1, master=test_frame).get_tk_widget().grid(column=1, row=6)
plot_widget2 = FigureCanvasTkAgg(fig2, master=test_frame).get_tk_widget().grid(column=2, row=6)
plot_widget3 = FigureCanvasTkAgg(fig3, master=test_frame).get_tk_widget().grid(column=3, row=6)





######################################################################################################################################################################################################
# Run the GUI

root.mainloop()