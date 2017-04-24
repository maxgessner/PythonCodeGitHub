# imports for getting modules
import sys
import os
import platform
import numpy as np
import pandas as pd
# import matplotlib as mpl

# # matplotlib with latex

# mpl.use("pgf")
# pgf_with_custom_preamble = {
#     "font.family": "serif", # use serif/main font for text elements
#     "text.usetex": True,    # use inline math for ticks
#     "pgf.rcfonts": False,   # don't setup fonts from rc parameters
#     "pgf.preamble": [
#          "\\usepackage{units}",         # load additional packages
#          "\\usepackage{metalogo}",
#          "\\usepackage{unicode-math}",  # unicode math setup
#          r"\setmathfont{xits-math.otf}",
#          r"\setmainfont{DejaVu Serif}", # serif font via preamble
#          ]
# }
# mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
# # for use with python 3 and above
# from tkinter import filedialog as fd
# from tkinter import Tk as tk
# for use with python 2.x
# import tkFileDialog as df
from Tkinter import Tk as tk
# import math

# import tkinter
from Tkinter import Label, Entry, END, Button, E, W, mainloop
from Tkinter import Frame, RIGHT, TOP, BOTTOM, Checkbutton
from Tkinter import IntVar, YES, X, GROOVE
# N, S, Y, LEFT
# for Dialog2 to work

# import own functions
# from <filename without .py> import <functionname>

# some default values to be called by user
crosssection = 10
density = 10000
perimeter = 22
length = 10
ambient_temperature = 296.15


# definitions of functions to be called later

from functions import getcalibration

from functions import cal_pyrotemp

from functions import cal_Cp

from functions import getmaterial

from functions import epsht

from functions import defaultplot

from classes import Dialog2

from functions import delta_cp_and_e_ht

from functions import getdata


# end of definitions loading
#
# def c_f(x): return float(x.replace(b',', b'.'))

# def conv(x):
#    return x.replace(b',', b'.')

# is commented for test reasons,
# just to load same file every time, is faster
# # ask user to set path for data
# window = tk()
# window.withdraw()
# file = fd.askopenfilename(filetypes=[('Text files', '*.txt')], \
#   initialdir='I:\THERMISCHE ANALYSE\Messmethoden\PISA\PISA_Labor\highspeed')
# window.destroy()

# for test run every time same file is loaded
if platform.system() == 'Windows':
    file = 'I:\THERMISCHE ANALYSE\Messmethoden\PISA\PISA_Labor\highspeed' \
           '\\160922\HighSpeed_DAQ_22092016_Niob_a_500ms_14V_out.txt'
# changed for testing on Linux Laptop which is not connected to ZAE
elif platform.system() == 'Linux':
    # small test file
    # file = 'HighSpeed_DAQ_22092016_Niob_a_500ms_14V_out.txt'
    # huge 100Mb test file
    # file = 'HighSpeed_DAQ_14062016_16_5V_1500ms_PUV_out_all_2.txt'
    # file = 'HighSpeed_DAQ_22092016_Niob_a_500ms_16V_out_1.txt'
    # file = 'HighSpeed_DAQ_13062016_PUV_16_5V_1250ms_out_6.txt'
    file = 'demodata1_5col.txt'
else:
    sys.exit('It is recommened to use Linux! (or Windows if you have to)')

# check if user aborted file selection
if file == '':
    sys.exit('no file to load data from!')

# read first line of datafile to get headers
try:
    infile = open(file, 'r')
    header_data = infile.readline()
finally:
    infile.close()
header_data = header_data.split()
header_data = [item.lower() for item in header_data]

# get data from file
# important to notice: pandas.read_csv are faster than numpy.genfromtxt
data = pd.read_csv(file, delimiter='\t', header=1,
                   names=header_data, engine='c', decimal=',')
# data = data[data != 0]

# initialize Checkbarlist for dialog later
Checkbarlist = ['save plots?']

[puv, puv_time, pv, time, filt_dat, current, r_spec] = getdata(data)
# plt.plot(r_spec)
# plt.show()

# write data in seperate vatiables for better readability
for i in range(0, len(header_data)):
    if header_data[i] == 'time':
        time = np.array(data['time'], dtype=pd.Series)
    elif header_data[i] == 'current':
        # current = np.array(data['current'], dtype=pd.Series)
        Checkbarlist.append('current')
    elif header_data[i] == 'r_spec':
        # r_spec = np.array(data['r_spec'], dtype=pd.Series)
        Checkbarlist.append('r_spec')
    elif header_data[i] == 'puv':
        # puv = np.array(data['puv'], dtype=pd.Series)
        puv_calib = getcalibration('puv')[0]
        Checkbarlist.append('puv')
    elif header_data[i] == 'pv':
        # pv = np.array(data['pv'], dtype=pd.Series)
        pv_calib = getcalibration('pv')[0]
        Checkbarlist.append('pv')
    elif header_data[i] == 'pvm':
        # pvm = np.array(data['pvm'], dtype=pd.Series)
        pvm_calib = getcalibration('pvm')[0]
        Checkbarlist.append('pvm')
    elif header_data[i] == 'sphere':
        sphere = np.array(data['sphere'], dtype=pd.Series)
    elif header_data[i] == 'digital_switch':
        digital_switch = np.array(data['digital_switch'], dtype=pd.Series)
    elif header_data[i] == 'blub':
        blub = np.array(data['blub'], dtype=pd.Series)
    # else:
        # removal_file[i] = data[header[i]]

# print(puv[0:10])

# ask user to input several data from sample
# [crosssection, density, perimeter, length, ambient_temperature] = dialog2()

# first dialog window with default values
root = tk()
[crosssection, density, perimeter, length, ambient_temperature] = \
    Dialog2(root).show_entry_fields()
# root.bind('<Return>', Dialog2(root).show_entry_fields())
root.mainloop()


# print(density)
# start calculating with values
# rho = 0.5  # specific resistance of measured material
shunt = 0.001  # Ohms

# here be dragons

rawdata = {}

if 'puv' in locals() and 'pv ' in locals():
    rawdata['puv'] = puv
    rawdata['pv'] = pv
    rawdata['time'] = time
    # print('blub2')
    print(puv)
    (puv, puv_time, pv, pv_time, filt_dat) = getdata(rawdata)
    print(puv)


# convert pyrometer values to real temperatures using calibration files
if 'puv' in locals():
    temp_puv = cal_pyrotemp(puv, 'PUV')
    # print('blub')

if 'pv' in locals():
    temp_pv = cal_pyrotemp(pv, 'PV')
    # print('bla')

if 'pvm' in locals():
    temp_pvm = cal_pyrotemp(pvm, 'PVM')


# here be something else


# global cp_calculate_check
cp_calculate_check = 0
e_ht_calculate_check = 0

# Here be Dragons

# =============================================================================
# global check
# check = {}

from classes import Checkbar

# global check


# call dialog window for user to check what to be calculated/shown
root = tk()
Checkbarlist.extend(['C_p', 'e_ht'])
calc = Checkbar(root, Checkbarlist)
calc.pack(side=TOP, fill=X)
calc.config(relief=GROOVE, bd=2)

# Checkbar.getvar(Checkbar)
# print(calc)
# print(check['puv'])
root.mainloop()

check = calc()

# print(check.get('puv'))

# print(calc)
# print(cp_calculate_check, e_ht_calculate_check)
# =============================================================================
'''
class Dialog3():
    def __init__(self,master):
       self.master = master
       master.title("Specify what to be calculated:")
       frame = Frame(master)
       frame.pack()

       def combine_funcs(*funcs):
           def combined_func(*args, **kwargs):
               for f in funcs:
                   f(*args, **kwargs)
           return combined_func

       var1 = IntVar()
       self.ch_cp = Checkbutton(frame, text="Calculate C_p",
                                variable=var1).grid(row=0, sticky=W)
       self.b_cp = Button(frame, text='Calculate C_p',
                          command=self.b_cp_clicked)
       self.b_cp.pack(side=LEFT)


       self.b_quit = Button(frame, text='Quit', command=master.destroy)
       self.b_quit.pack(side=LEFT)

       # self.label_cross = Label(master, text='crosssection:').grid(row=0)
       # Label(master, text='density:').grid(row=1)
       # Label(master, text='perimeter:').grid(row=2)
       # Label(master, text='length:').grid(row=3)
       # Label(master, text='ambient temperature:').grid(row=4)


    def b_cp_clicked(self):
       global cp_calculate_check
       cp_calculate_check = True
       # return cp_calculate_check

root2 = tk()
Dialog3(root2)
root2.mainloop()

print(cp_calculate_check)
'''
# Here be something else
# print(cp_calculate_check, e_ht_calculate_check)
# print(check)


# calculate current and r_spec using shunt resistance
# will be obsolete with new setup
current = current / shunt
r_spec = r_spec / current


# plot current or r_spec vs time if selected in Checkbarlist dialog window
if 'current' in check:
    if check['current'] == 1:
        current_plot = plt.figure(0)
        plt.plot(time, current)

        defaultplot('current vs time', 'time [s]', 'current [A]',
                    time, legend='current')
        np.save('current', (time, current))
    elif check['current'] == 0:
        current = 0
else:
    check['current'] = 0
    print('no current given')

if 'r_spec' in check:
    if check['r_spec'] == 1:
        r_spec_plot = plt.figure(1)
        plt.plot(time, r_spec)

        defaultplot('r_spec vs time', 'time [s]', r'r_spec [$\Omega$/m]',
                    time, legend='r_spec')
        np.save('r_spec', (time, r_spec))
    elif check['r_spec'] == 0:
        r_spec = 0
else:
    check['r_spec'] = 0
    print('no r_spec given')

if 'C_p' in check and 'pv' in check:
#  and \
   # 'current' in check and 'r_spec' in check:
    if check['C_p'] == 1 and check['pv'] == 1 and check['puv'] == 1:
    #  and \
       # check['current'] == 1 and check['r_spec'] == 1:
        # cp = cal_Cp(temp_pv, r_spec, time, current,
        #             crosssection, density, perimeter, ambient_temperature)
        # print(np.shape(puv_time))
        (epsht, delta_cp, f_epsht, f_delta_cp, c_time) = \
            delta_cp_and_e_ht(temp_puv, puv_time, temp_pv, time)
        # hier failt irgendwas, ich denke, dass hier eine window-funktion
        # angebracht waere
        # diese sollte dann auch fuer "normale" Messungen die Daten
        # entsprechend splitten
        c_time = c_time[:-1]

        # print(type(blafu))
        delta_cp_plot = plt.figure(2)
        # print(np.shape(c_time))
        # print(np.shape(delta_cp))
        plt.plot(c_time, delta_cp)
        # plt.text(x,'time')
        defaultplot('heatcapacity vs time', 'time [s]', 'heatcapacity [J/K]',
                    time, legend='heatcapacity')
        np.save('heatcapacity', (time, delta_cp))
    elif check['C_p'] == 0:
        delta_cp = 0
        print('C_p not set')
    elif check['pv'] != 1:
        delta_cp = 0
        print('delta_cp not available due to missing pv')
    elif check['puv'] != 1:
        delta_cp = 0
        print('delta_cp not available due to missing puv')
else:
    check['C_p'] = 0
    print('no C_p given')

if 'e_ht' in check and 'pv' in check:
    if check['pv'] == 1 and check['puv'] == 1 and check['e_ht'] == 1:
        # eht = epsht(time, temp_pv, temp_puv)
        (epsht, delta_cp, f_epsht, f_delta_cp, c_time) = \
            delta_cp_and_e_ht(temp_puv, puv_time, temp_pv, time)
        # print(len(time), len(blub))
        # print(type(blub))
        c_time = c_time[:-1]
        eht_plot = plt.figure(3)
        plt.plot(c_time, epsht)
        defaultplot('hemispherical total emittance vs time', 'time [s]',
                    r'$\varepsilon$ [a.u.]', time, legend=r'$\varepsilon$')
        np.save('e_ht', (time, epsht))
        # plt.show()
    elif check['e_ht'] == 0:
        epsht = 0
        print('e_ht not set')
    elif check['pv'] != 1:
        print('e_th not available due to missing pv')
    elif check['puv'] != 1:
        print('e_th not available due to missing puv')
else:
    check['e_ht'] = 0
    print('no e_ht given')

if 'puv' in check:
    if check['puv'] == 1:
        puv_plot = plt.figure(4)
        plt.plot(puv_time, temp_puv)
        defaultplot('temperature puv vs time', 'time [s]',
                    'temperature [K]', time, legend='puv')
        np.save('puv', (puv_time, temp_puv))
    elif check['puv'] == 0:
        puv_plot = 0
else:
    check['puv'] = 0
    print('no puv given')

if 'pv' in check:
    if check['pv'] == 1:
        pv_plot = plt.figure(5)
        plt.plot(time, temp_pv)
        defaultplot('temperature pv vs time', 'time [s]',
                    'temperature [K]', time, legend='pv')
        np.save('pv', (time, temp_pv))
    elif check['pv'] == 0:
        pv_plot = 0
else:
    check['pv'] = 0
    print('no pv given')


plt.show()
# datadirectory = 'I:\THERMISCHE ANALYSE\Messmethoden \
# \\PISA\PISA_Labor\highspeed'
datadirectory = os.path.dirname(file)
setdpi = 300

# print(datadirectory)

if check['save plots?'] == 1:
    if platform.system() == 'Windows':
        if 'current' in check and check['current'] == 1:
            current_plot.savefig(datadirectory + '\\' + 'current_vs_time.pdf',
                                 dpi=setdpi)
        if 'current' in check and check['r_spec'] == 1:
            r_spec_plot.savefig(datadirectory + '\\' + 'r_spec_vs_time.pdf',
                                dpi=setdpi)
        if 'C_p' in check and check['C_p'] == 1:
            delta_cp_plot.savefig(datadirectory + '\\' + 'cp_vs_time.pdf',
                                  dpi=setdpi)
        if 'e_ht' in check and check['e_ht'] == 1:
            eht_plot.savefig(datadirectory + '\\' + 'eht_vs_time.pdf',
                             dpi=setdpi)
        if 'puv' in check and check['puv'] == 1:
            puv_plot.savefig(datadirectory + '\\' + 'puv_vs_time.pdf',
                             dpi=setdpi)
        if 'pv' in check and check['pv'] == 1:
            pv_plot.savefig(datadirectory + '\\' + 'pv_vs_time.pdf',
                            dpi=setdpi)
    elif platform.system() == 'Linux':
        if 'current' in check and check['current'] == 1:
            current_plot.savefig(datadirectory + 'current_vs_time.pdf',
                                 dpi=setdpi)
        if 'r_spec' in check and check['r_spec'] == 1:
            r_spec_plot.savefig(datadirectory + 'r_spec_vs_time.pdf',
                                dpi=setdpi)
        if 'C_p' in check and check['C_p'] == 1:
            delta_cp_plot.savefig(datadirectory + 'cp_vs_time.pdf',
                                  dpi=setdpi)
        if 'e_ht' in check and check['e_ht'] == 1:
            eht_plot.savefig(datadirectory + 'eht_vs_time.pdf',
                             dpi=setdpi)
        if 'puv' in check and check['puv'] == 1:
            puv_plot.savefig(datadirectory + 'puv_vs_time.pdf',
                             dpi=setdpi)
        if 'pv' in check and check['pv'] == 1:
            pv_plot.savefig(datadirectory + 'pv_vs_time.pdf',
                            dpi=setdpi)
    else:
        sys.exit('It is recommened to use Linux! (or Windows if you have to)')


# print current_rho
# # current_rho.size
# # plt.plot(time,current_rho)
# plt.figure(1)

# print(time)
# print(temp_puv)
# plt.figure(1)
# plt.plot(time, blafu)
# plt.show()
# plt.figure(2)
# plt.plot(time, r_spec)
# # plt.close()
# plt.show()
# # plt.savefig('test2.png')
# #html('<img src='cell://test2.png' />')

# # fd.destroy()
