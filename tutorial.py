# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:52:39 2025

@author: edillu
"""


#%% Code for reading HDF5 files
import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('./59146237/measure/CH07_59146237_s0000029.h5', 'r')


print(f['measure']['value'].keys())

ecg = f['measure']['value']['_030']['value']['data']['value'][0,:]
time = f['measure']['value']['_030']['value']['time']['value'][0,:]

plt.figure(figsize=(12, 5))
plt.plot(time,ecg)
plt.title("ECG signal")
plt.xlabel('Time (ms)')
plt.ylabel('ECG (mV)')
plt.show()

#%% Code for computing the dZ/dt signal

import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('./59146237/measure/CH07_59146237_s0000029.h5', 'r')


print(f['measure']['value'].keys())

icg = f['measure']['value']['_031']['value']['data']['value'][0,:]
time = f['measure']['value']['_031']['value']['time']['value'][0,:]

plt.figure(figsize=(12, 5))
plt.plot(time,icg)
plt.title("Raw ICG signal")
plt.xlabel('Time (ms)')
plt.ylabel('ICG (Ω)')
plt.show()

dt = np.mean(np.diff(time))
dz = np.gradient(icg, dt)

plt.figure(figsize=(12, 5))
plt.plot(time,dz)
plt.title("dZ/dt signal")
plt.xlabel('Time (ms)')
plt.ylabel('dZ/dt')
plt.show()

#%% Code for plotting ECHO images

import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('./59146237/measure/CH07_59146237_s0000029.h5', 'r')


echo = f['measure']['value']['_091']['value']['data']['value'][0,:,:].transpose()

plt.figure(figsize=(12, 5))
plt.imshow(echo, cmap='viridis', aspect='auto')
plt.title("Echocardiography Image")
plt.xlabel('Time (ms)')
plt.show()