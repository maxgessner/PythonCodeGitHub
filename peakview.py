import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
# from Tkinter import askopenfilename

# initialdir = 'home/mgessner/'

# ask user to set path for data
# from Tkinter import Tk as tk
# import tkFileDialog as fd
# window = tk()
# window.withdraw()
# file = fd.askopenfilename(filetypes=[('Text files', '*.txt')],
#                           initialdir='/home/mgessner/PythonCode/')
# window.destroy()
# exit()

# for speed reasons
# file = '/home/mgessner/PythonCode/HighSpeed_DAQ_13062016_PUV_16_5V_1250ms_out_6.txt'
file = '/home/mgessner/PythonCode/HighSpeed_DAQ_13062016_PUV_16_5V_1250ms_out_5.txt'

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

# print(data[header_data[0]].unique())
# time = np.array(data[header_data[0]])

# print(data[header_data[0]][1:])
# print(data[header_data[0]][:-1])
if len(data[header_data[0]].unique()) == 1:
    time = DataFrame(range(len(data[header_data[0]])))
    print('hello')
else:
    time = data[header_data[0]]
# exit()

# ax = plt.add_subplot(241)
# corefig, ax = plt.subplots(8, sharex=True)
# corefig.figure('peakview', figsize=(20, 10))
corefig = plt.figure('peakview', figsize=(20, 10))
ax = plt.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.96)

# ax = plt.add_subplot(241)

for i in range(1, len(header_data)):
    ax = corefig.add_subplot(2, 4, i, sharex=ax)
    # ax.subplot(2, 4, i)
    ax.plot(time, data[header_data[i]])
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))
    ax.set_title(header_data[i])
    ax.set_xlabel('time')



# plt.show()

# import matplotlib.pyplot as plt

# start with one
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot([1, 2, 3])
# ax.plot([4, 5, 6])

# # now later you get a new subplot; change the geometry of the existing
# n = len(fig.axes)
# print(n)
# for i in range(n):
#     fig.axes[i].change_geometry(n+1, 1, i+1)

# # add the new
# ax = fig.add_subplot(n+1, 1, n+1)
# ax.plot([4,5,6])

plt.show()

