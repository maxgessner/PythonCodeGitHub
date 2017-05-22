import numpy as np
import pandas as pd
from Tkinter import Tk as tk
from Tkinter import Radiobutton, StringVar, W, Button, S
from tkFileDialog import askdirectory
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import least_squares
from functions import cal_pyrotemp
import sys

# set start directory for user to choose from
# startdir = '/home/mgessner/vm_share/Linearity/22102015/'
# startdir = '/home/mgessner/vm_share/Linearity/12052017/'
startdir = '/home/mgessner/vm_share/Linearity/test/'

root0 = tk()
root0.withdraw()  # we don't want a full GUI,
# so keep the root window from appearing

dirname = askdirectory(initialdir=startdir)
# show an "Open" dialog box and return the path to the selected file
if dirname == () or dirname == '':
    sys.exit('no directory given!')
root0.destroy()

# filenames = listdir(dirname)
filenames = []
pyrolist = []
dpf = np.array(())  # dpf = data per file
cpf = np.array(())  # cpf = comment per file
# apf = np.array(())

# initialize arrays for mean values and standard deviation
mlamp1 = np.array(())
mlamp2 = np.array(())
mlampno = np.array(())
mlampboth = np.array(())

sdlamp1 = np.array(())
sdlamp2 = np.array(())
sdlampno = np.array(())
sdlampboth = np.array(())

# load all text files in given directory and append them to apf
for file in listdir(dirname):
    if file.endswith(".txt"):
        filenames.append(file)
        if 'PVM' in file:
            pyrolist.append('PVM')
        elif 'PV' in file:
            pyrolist.append('PV')
        elif 'PUV' in file:
            pyrolist.append('PUV')
        header = pd.read_csv(dirname + '/' + file, delimiter='\t', header=1,
                             engine='c', nrows=0, decimal=',')
        headernames = ['time', 'pyrometer', 'lamp1', 'lamp2', 'comment']
        # print(len(header.columns))
        if len(header.columns) == 3:
            headernames = headernames[:2] + headernames[4:]
            # headernames.pop(2)
        # print(headernames)

        data = pd.read_csv(dirname + '/' + file, delimiter='\t', header=1,
                           names=headernames, engine='c', decimal=',')

        npdata = np.asarray(data['pyrometer'])
        npcomment = np.asarray(data['comment'])
        dpf = np.append(dpf, npdata)
        cpf = np.append(cpf, npcomment)
        apf = np.stack((dpf, cpf), axis=1)
        # apt = np.append(apf, (npdata, npcomment))

# find out where one measurement ends and the next one starts
split = np.where(apf[:-1, 1] != apf[1:, 1])[0]
switch = np.where(apf[1:, 1] == apf[0, 1])[0]

namechange = np.intersect1d(split, switch)

namechange_start = np.append(0, namechange)
namechange_stop = np.append(namechange, len(apf))

apf_split = np.ndarray(())

# cycle over all measurements in apf
for i in range(len(namechange_start)):
    apf_split = apf[namechange_start[i]:namechange_stop[i], 1]
    apf_split_values = apf[namechange_start[i]:namechange_stop[i], 0]

    lamp1 = apf_split_values[apf_split == 'Lamp_1']
    lamp2 = apf_split_values[apf_split == 'Lamp_2']
    lampno = apf_split_values[apf_split == 'Lamp_no']
    lampboth = apf_split_values[apf_split == 'Lamp_both']

    # as the shutters are not lightning fast one has to cut of several
    # values in the beginning of everv change in measurement to get
    # consistend values
    start = 5
    end = 1

    lamp1 = lamp1[start:-end]
    lamp2 = lamp2[start:-end]
    lampno = lampno[start:-end]
    lampboth = lampboth[start:-end]

    mlamp1 = np.append(mlamp1, np.mean(lamp1))
    mlamp2 = np.append(mlamp2, np.mean(lamp2))
    mlampno = np.append(mlampno, np.mean(lampno))
    mlampboth = np.append(mlampboth, np.mean(lampboth))

    sdlamp1 = np.append(sdlamp1, np.std(lamp1))
    sdlamp2 = np.append(sdlamp2, np.std(lamp2))
    sdlampno = np.append(sdlampno, np.std(lampno))
    sdlampboth = np.append(sdlampboth, np.std(lampboth))

# if the name of the measured pyrometer can not be found in the filenames
# the user has to set it by hand via dialog box
if len(set(pyrolist)) != 1:
    # print(filenames)
    # from Tkinter import *

    # global choice
    choice = 0

    root = tk()

    v = StringVar()

    def endGUI():
        global choice
        choice = v.get()
        # print(choice)
        root.quit()
        root.destroy()
        # exit()
        # close_window()
        # root.destroy()

    def quitGUI():
        root.quit()
        root.destroy()
        sys.exit('aborted!')

    RB1 = Radiobutton(root, text='PVM', variable=v, value='PVM').pack(anchor=W)
    RB2 = Radiobutton(root, text='PV', variable=v, value='PV').pack(anchor=W)
    RB3 = Radiobutton(root, text='PUV', variable=v, value='PUV').pack(anchor=W)
    okButton = Button(root, text='OK', command=endGUI).pack(anchor=S)
    cancelButton = Button(root, text="Cancel", command=quitGUI).pack(anchor=W)
    choice = v.get()

    root.mainloop()
    # exit()
    # root.update()
    # print(choice)
elif len(set(pyrolist)) == 1:
    # global choice
    # print(pyrolist[0])
    choice = pyrolist[0]

# exit()
pyrometer = choice
seq = np.argsort(mlamp1)

mlamp1 = mlamp1[seq]
mlamp2 = mlamp2[seq]
mlampno = mlampno[seq]
mlampboth = mlampboth[seq]
sdlamp1 = sdlamp1[seq]
sdlamp2 = sdlamp2[seq]
sdlampno = sdlampno[seq]
sdlampboth = sdlampboth[seq]

# pyrometer = 'PV'

mlampsum = mlamp1 + mlamp2 - 2 * mlampno

Tmlampsum = cal_pyrotemp(mlamp1 + mlamp2, pyrometer)
Tmlampboth = cal_pyrotemp(mlampboth, pyrometer)

maxTmlampsum = cal_pyrotemp(mlamp1 + mlamp2 + sdlamp1 + sdlamp2, pyrometer)
minTmlampsum = cal_pyrotemp(mlamp1 + mlamp2 - sdlamp1 - sdlamp2, pyrometer)

maxTmlampboth = cal_pyrotemp(mlampboth + sdlampboth, pyrometer)
minTmlampboth = cal_pyrotemp(mlampboth - sdlampboth, pyrometer)

fig, ax = plt.subplots()

ax.errorbar(Tmlampsum, Tmlampboth, marker='.', fmt='.',
            yerr=[Tmlampsum - minTmlampsum, maxTmlampsum - Tmlampsum],
            xerr=[Tmlampboth - minTmlampboth, maxTmlampboth - Tmlampboth],
            label='measured points')

x0 = [0, 0, 0]


def lincalib(x, Tsum, Tboth):
    return(Tboth - (x[0] + x[1] * Tsum + x[2] * Tsum**2))


def lincalib_penality(x, Tsum, Tboth):
    x[0] = 1 - (x[1] + x[2])
    return(Tboth - (x[0] + x[1] * Tsum + x[2] * Tsum**2))


def fun(x, Tsum):
    return(x[0] + x[1] * Tsum + x[2] * Tsum**2)

# using least squares technique to determine the best fitting
# from the measured values
result = least_squares(lincalib_penality, x0, args=(Tmlampsum, Tmlampboth),
                       method='lm',  # bounds=(-1, 1),
                       verbose=0,  # jac='3-point',
                       # x_scale='jac',  # 10**(20),
                       # f_scale=10**(-8),
                       # max_nfev=2000,
                       xtol=2.22044604926e-16,
                       ftol=2.22044604926e-16,
                       gtol=2.22044604926e-16,
                       # loss='cauchy',
                       # tr_solver='exact'
                       )

Temp = np.arange(np.min(Tmlampsum), np.max(Tmlampsum), 0.01)

# using LaTEX in labels and annotations via $ $
ax.plot(Temp, fun(result['x'], Temp), label='fitted curve')
plt.legend(loc=4)
plt.xlabel('$\mathregular{T_1 + T_2}$', fontsize=14)
plt.ylabel('$\mathregular{T_{1+2}}$', fontsize=14)
titletext = ('$\mathregular{Linearity \, fit \, with'
             '\, leastsquares \, method:}$ \n'
             '$\mathregular{T(L_1)+T(L_2) = a + b \cdot T(L_1+L_2) +}$' +
             '$\mathregular{c \cdot (T(L_1+L_2))^2}$')
plt.title('Lineatiry calibration for ' + choice)
plt.annotate(titletext, xy=(0.02, 0.88), xycoords='axes fraction')
optimal_values = ('a = ' + str(round(result['x'][0], 4)) +
                  '\nb = ' + str(round(result['x'][1], 4)) +
                  '\nc = ' + str(round(result['x'][2], 4)))
plt.annotate(optimal_values, xy=(0.05, 0.45), xycoords='axes fraction')
plt.grid(True)
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(10))
plt.show()

# change this value if you want to save the plot as pdf
save_figure = False

if save_figure is True:
    fig.savefig(dirname + '/Linearcalib_' + choice + '.pdf', dpi=300)

exit()
