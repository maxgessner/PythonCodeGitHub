import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
# import pandas as pd
from pandas import DataFrame
from scipy.signal import savgol_filter
from pandas import Categorical as pdC


# define length of demodata to generate
length = 25


def fun(x):
    # result = (-1/(np.sin(x))**4)
    # result = (np.sinc(x-np.pi/2))**10
    # shape -> lower the more sinus like
    # shape -> higher the more square like
    shape = 1
    # result = np.sqrt(shape / (1 + shape * np.sin(x))) * np.sin(x)
    result = shape * np.sin(x)
    # result = 0.99 * signal.square(x)**0.025 + 0.01 * np.sin(x)
    return result


def fun2(x, r):
    result = np.sin(x)**r
    return result


start = 0

sinus = fun2(np.arange(start, np.pi - start, 0.01), 1)
# sinus = fun(np.arange(start, np.pi - start, 0.01))
sinus = np.clip(sinus, 0, 1)

signal = sinus

# print(sinus)
# exit()

rest = np.zeros(len(sinus)) + 300
sinus0 = int(len(sinus) * 2)
# rest2 = rest
# print(rest)
# exit()


for c in range(length):
    # if c <= length/3:
    #     sigma = 1.
    # elif c > length/3:
    #     sigma = 1./(c-length/3)
    d = np.double(c)
    sigma = 2. / (length + c)
    sinus2 = (fun2(np.arange(start, np.pi - start, 0.01), d / length) -
              20 * c * sigma) + 2500
    # signal2 = np.zeros(len(sinus2)) + np.amax(sinus2) - 200
    # print(signal2)
    # sinus2 = (sinus - 1000 * c * sigma) + 2500
    sinus2[:c] = 300
    sinus2[-c:] = 300
    # rest = np.concatenate((rest, (sinus - 1000 * c * sigma) + 2500), axis=0)
    rest = np.concatenate((rest, sinus2), axis=0)
    rest = np.concatenate((rest, np.zeros(sinus0) + 300), axis=0)
    # rest2 = np.concatenate((rest2, signal2), axis=0)
    # rest2 = np.concatenate((rest2, np.zeros(sinus0) + np.amax(signal2)),
    #                         axis=0)
#     print(sinus2)
# plt.plot(sinus2)
# plt.show()
# exit()


# smooth data
for i in range(20):
    rest = savgol_filter(rest, 21, 1)

plt.plot(rest)
# plt.show()
# exit()
# print(type(rest), len(rest))
# print(rest)

nrest = np.arange(len(rest))

# rest2 = np.arange(len(rest))
rest2 = np.ones(len(rest)) * \
    np.amax(rest) - 200 + 200 / \
    (np.arange(len(rest), dtype=np.double()) + 1)**(1. / 10)

rest_noise = rest + 0.001 * np.random.rand(len(rest))
rest2_noise = rest2 + 0.001 * np.random.rand(len(rest2))

current = np.random.rand(len(rest2_noise))
r_spec = np.random.rand(len(rest2_noise))

# for g in range(length):
#     # sigma = 1. / (length + g)
#     h = np.double(g)
#     sigma = 1. / (length + h)
#     plt.plot(((fun2(np.arange(start, np.pi - start, 0.01), h / length)) -
#              1000 * h * sigma) + 2500)
# plt.plot(rest_noise)

# plt.plot(rest2)
# plt.show()
# exit()

# print(np.shape(rest), np.shape(rest2))


# file = 'I:\THERMISCHE ANALYSE\Messmethoden\PISA\PISA_Labor\demodata1.txt'
file = 'demodata1.txt'
file_noise = 'demodata1_noise.txt'
file_5col = 'demodata1_5col.txt'
file_5col_noise = 'demodata1_5col_noise.txt'

output = {'time': nrest, 'puv': rest, 'pv': rest2}
outputdf = DataFrame(data=output)
outputdf = outputdf[['time', 'puv', 'pv']]
# print(output)

output_noise = {'time': nrest, 'puv_noise': rest_noise,
                'pv_noise': rest2_noise}
outputdf_noise = DataFrame(data=output_noise)
outputdf_noise = outputdf_noise[['time', 'puv_noise', 'pv_noise']]
# print(outputdf_noise)
# print(outputdf_noise2)

outputdf.to_csv(file, sep='\t', header=['time', 'puv', 'pv'],
                decimal=',', index=False)

outputdf_noise.to_csv(file_noise, sep='\t', header=['time', 'puv', 'pv'],
                      decimal=',', index=False)

output_5col = {'time': nrest, 'current': current, 'r_spec': r_spec,
               'puv': rest, 'pv': rest2}

output_5coldf = DataFrame(data=output_5col)
output_5coldf = output_5coldf[['time', 'current', 'r_spec', 'puv', 'pv']]

output_5coldf.to_csv(file_5col, sep='\t',
                     header=['time', 'current', 'r_spec', 'puv', 'pv'],
                     decimal=',', index=False)

output_5col_noise = {'time': nrest, 'current': current, 'r_spec': r_spec,
                     'puv_noise': rest_noise, 'pv_noise': rest2_noise}

output_5col_noisedf = DataFrame(data=output_5col_noise)
output_5col_noisedf = output_5col_noisedf[['time', 'current', 'r_spec',
                                           'puv_noise', 'pv_noise']]

# print(output_5col_noisedf)

output_5col_noisedf.to_csv(file_5col_noise, sep='\t',
                           header=['time', 'current', 'r_spec', 'puv', 'pv'],
                           decimal=',', index=False)


# In [21]: df['m'] = pd.Categorical(df['m'], ["March", "April", "Dec"])
