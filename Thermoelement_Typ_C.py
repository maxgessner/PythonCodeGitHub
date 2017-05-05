import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint as pp
from scipy import interpolate
# np.set_printoptions(threshold=np.nan)

file = 'Thermoelement_Typ_C_Reference_2.txt'

data = pd.read_csv(file, delimiter='\t', header=None, engine='c', decimal=',')

# print(data[:, 2:])
# print(data)
# print(data.T)
# exit()
# print(data[0])

value = np.array(())
temp_C = np.array(())

data = np.array(data)
# print(data[0])
# exit()
# print(range(np.shape(data)[0]))
# print(data[188])

# plt.plot(data[:, 2:], '.')
# plt.show()
# exit()

for r in range(1, np.shape(data)[0]):
    for c in range(1, np.shape(data)[1]):
        # print(data[c, r])
        value = np.append(value, data[r, c])
        temp_C = np.append(temp_C, data[0, c] + data[r, 0])
        # print(data[0,c])


temp = temp_C + 273.15 # in Kelvin
print(temp, value)
plt.plot(temp, value, '.')
# plt.show()

f_value = interpolate.interp1d(temp, value, kind='linear')
plt.plot(temp, f_value(temp))
plt.show()
