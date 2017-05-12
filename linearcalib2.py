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

# startdir = '/home/mgessner/vm_share/Linearity/22102015/'
startdir = '/home/mgessner/vm_share/Linearity/12052017/'

root0 = tk()
root0.withdraw()  # we don't want a full GUI,
# so keep the root window from appearing

dirname = askdirectory(initialdir=startdir)
# show an "Open" dialog box and return the path to the selected file
# print(dirname)
if dirname == () or dirname == '':
    exit()
root0.destroy()

# filenames = listdir(dirname)
filenames = []
pyrolist = []
data_per_file = []
# print(filenames)
# headernames = ['time', 'pyrometer', 'lamp1', 'lamp2', 'comment']

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
        # print(data.index())
        data_per_file.append(data)
        # print(data['comment'])
        lamp1 = np.array(data[data['comment'] == 'Lamp_1']['pyrometer'])
        lamp2 = np.array(data[data['comment'] == 'Lamp_2']['pyrometer'])
        lampno = np.array(data[data['comment'] == 'Lamp_no']['pyrometer'])
        lampboth = np.array(data[data['comment'] == 'Lamp_both']['pyrometer'])

        start = 3
        end = 1

        lamp1 = lamp1[start:-end]
        lamp2 = lamp2[start:-end]
        lampno = lampno[start:-end]
        lampboth = lampboth[start:-end]

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

# pyrolist.append('PVM')

# print(len(set(pyrometer)))

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

    RB1 = Radiobutton(root, text='PVM', variable=v, value='PVM').pack(anchor=W)
    RB2 = Radiobutton(root, text='PV', variable=v, value='PV').pack(anchor=W)
    RB3 = Radiobutton(root, text='PUV', variable=v, value='PUV').pack(anchor=W)
    okButton = Button(root, text='OK', command=endGUI).pack(anchor=S)
    cancelButton = Button(root, text="Cancel", command=exit).pack(anchor=W)
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

# plt.plot(mlampsum, mlampboth, '.')
# plt.show()
# plt.plot(Tmlampsum, Tmlampboth, '.')
ax.errorbar(Tmlampsum, Tmlampboth, marker='.', fmt='.',
            yerr=[Tmlampsum - minTmlampsum, maxTmlampsum - Tmlampsum],
            xerr=[Tmlampboth - minTmlampboth, maxTmlampboth - Tmlampboth],
            label='measured points')
# plt.show()

x0 = [0, 0, 0]


def lincalib(x, Tsum, Tboth):
    return(Tboth - (x[0] + x[1] * Tsum + x[2] * Tsum**2))


def fun(x, Tsum):
    return(x[0] + x[1] * Tsum + x[2] * Tsum**2)


result = least_squares(lincalib, x0, args=(Tmlampsum, Tmlampboth),
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
# print(result)
# print(result['x'])

Temp = np.arange(np.min(Tmlampsum), np.max(Tmlampsum), 0.01)

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

save_figure = True

if save_figure is True:
    fig.savefig(dirname + '/Linearcalib_' + choice + '.pdf', dpi=300)

exit()
