import numpy as np
import pandas as pd
from Tkinter import Tk as tk
from tkFileDialog import askdirectory
from os import listdir, path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

startdir = '/home/mgessner/vm_share/Linearity/22102015/'

tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
dirname = askdirectory(initialdir=startdir)  # show an "Open" dialog box and return the path to the selected file
# print(dirname)
if dirname == ():
    exit()

# filenames = listdir(dirname)
filenames = []
data_per_file = []
# print(filenames)
headernames = ['time', 'pyrometer', 'lamp1', 'lamp2', 'comment']

mlamp1 = np.array(())
mlamp2 = np.array(())
mlampno = np.array(())
mlampboth = np.array(())

sdlamp1 = np.array(())
sdlamp2 = np.array(())
sdlampno = np.array(())
sdlampboth = np.array(())


for file in listdir(dirname):
    if file.endswith(".txt"):
        filenames.append(file)
        data = pd.read_csv(dirname + "/" + file, delimiter='\t', header=1,
                           names=headernames, engine='c', decimal=',')
        data_per_file.append(data)
        # print(data['comment'])
        lamp1 = np.array(data[data['comment'] == 'Lamp_1']['pyrometer'])
        lamp2 = np.array(data[data['comment'] == 'Lamp_2']['pyrometer'])
        lampno = np.array(data[data['comment'] == 'Lamp_no']['pyrometer'])
        lampboth = np.array(data[data['comment'] == 'Lamp_both']['pyrometer'])

        lamp1 = lamp1[10:-10]
        lamp2 = lamp2[10:-10]
        lampno = lampno[10:-10]
        lampboth = lampboth[10:-10]

        # print(lamp1)

        mlamp1 = np.append(mlamp1, np.mean(lamp1))
        mlamp2 = np.append(mlamp2, np.mean(lamp2))
        mlampno = np.append(mlampno, np.mean(lampno))
        mlampboth = np.append(mlampboth, np.mean(lampboth))

        sdlamp1 = np.append(sdlamp1, np.std(lamp1))
        sdlamp2 = np.append(sdlamp2, np.std(lamp2))
        sdlampno = np.append(sdlampno, np.std(lampno))
        sdlampboth = np.append(sdlampboth, np.std(lampboth))

        # print(len(mlamp1))
        # exit()
        # print(path.join(dirname, file))

# print(data_per_file[1])
# print(sdlamp1)
# print(sdlamp2)
# print(sdlampno)
# print(sdlampboth)

from functions import cal_pyrotemp

seq = np.argsort(mlamp1)

mlamp1 = mlamp1[seq]
mlamp2 = mlamp2[seq]
mlampno = mlampno[seq]
mlampboth = mlampboth[seq]
sdlamp1 = sdlamp1[seq]
sdlamp2 = sdlamp2[seq]
sdlampno = sdlampno[seq]
sdlampboth = sdlampboth[seq]

pyrometer = 'PV'

mlampsum = mlamp1 + mlamp2 - 2 * mlampno

Tmlampsum = cal_pyrotemp(mlamp1 + mlamp2, pyrometer)
Tmlampboth = cal_pyrotemp(mlampboth, pyrometer)

maxTmlampsum = cal_pyrotemp(mlamp1 + mlamp2 + sdlamp1 + sdlamp2, pyrometer)
minTmlampsum = cal_pyrotemp(mlamp1 + mlamp2 - sdlamp1 - sdlamp2, pyrometer)

maxTmlampboth = cal_pyrotemp(mlampboth + sdlampboth, pyrometer)
minTmlampboth = cal_pyrotemp(mlampboth - sdlampboth, pyrometer)


# plt.plot(mlampsum, mlampboth, '.')
# plt.show()
# plt.plot(Tmlampsum, Tmlampboth, '.')
plt.errorbar(Tmlampsum, Tmlampboth, marker='.',
             yerr=[Tmlampsum - minTmlampsum, maxTmlampsum - Tmlampsum],
             xerr=[Tmlampboth - minTmlampboth, maxTmlampboth - Tmlampboth])
# plt.show()

x0 = [0, 0, 0]

def lincalib(x, Tsum, Tboth):
    return(Tboth - (x[0] + x[1] * Tsum + x[2] * Tsum**2))

def fun(x, Tsum):
    return(x[0] + x[1] * Tsum + x[2] * Tsum**2)

result = least_squares(lincalib, x0, args=(Tmlampsum, Tmlampboth),
                       method='trf',  # bounds=(-1, 1),
                       verbose=0, jac='3-point',
                       x_scale='jac',  # 10**(20),
                       # f_scale=10**(-8),
                       # max_nfev=2000,
                       xtol=2.22044604926e-16,
                       ftol=2.22044604926e-16,
                       gtol=2.22044604926e-16,
                       loss='cauchy',
                       tr_solver='exact')
print(result['x'])

Temp = np.arange(np.min(Tmlampsum), np.max(Tmlampsum), 0.01)

plt.plot(Tmlampboth, fun(result['x'], Tmlampsum))
plt.show()

