import sys
import platform
import numpy as np
# import numdifftools as nd
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit as cfit
# plt.xkcd()
# import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# loading modules for fitting
# from scipy.optimize import curve_fit
# from scipy.optimize import leastsq
from scipy.optimize import least_squares
# from scipy.optimize import minimize
# from scipy.optimize import fmin
# from matplotlib import cm
# from scipy import signal
# from window_function import window_function
# from functions import window_function
from functions import getdata
from functions import delta_cp_and_e_ht
from itertools import cycle


# platform dependent, where to search for datafiles
if platform.system() == 'Windows':
    file = 'I:\THERMISCHE ANALYSE\Messmethoden\PISA\PISA_Labor' \
        '\\demodata1.txt'
# changed for testing on Linux Laptop which is not connected to ZAE
elif platform.system() == 'Linux':
    file = 'demodata1_noise.txt'
    # file = 'demodata1.txt'
else:
    sys.exit('It is recommened to use Linux! (or Windows if you have to)')


# check if user aborted file selection
if file == '':
    sys.exit('no file to load data from!')

# datalength = list(range(100))

# get data frprofile nuom file
# important to notice: pandas.read_csv are faster than numpy.genfromtxt
data = pd.read_csv(file, delimiter='\t', header=0, engine='c', decimal=',')

# use the rawdata on function getdata to split profiles and smooth it
[raw_dat, raw_time, raw_pv, raw_pv_time, filt_dat,
 current_dat, r_spec_dat] = getdata(data)


# save raw_d and raw_pv for further computations
np.save('filt_dat', filt_dat)
np.save('raw_dat', raw_dat)
np.save('raw_pv', raw_pv)
np.save('raw_time', raw_time)
np.save('raw_pv_time', raw_pv_time)
# exit()

# from compute_delta_cp_and_e_ht_as_function import delta_cp_and_e_ht
# from functions import delta_cp_and_e_ht
# print(np.shape(raw_time))
# exit()

(epsht, delta_cp, f_epsht, f_delta_cp, c_puv) = \
    delta_cp_and_e_ht(raw_puv=raw_dat,
                      raw_time=raw_time,
                      raw_pv=raw_pv,
                      raw_pv_time=raw_pv_time,
                      plotresult=True)
# exit()

# for 3D-plot of rawdata initiate y and z range
# print(np.shape(filt_dat[0]))
yrange = np.arange(1, np.shape(filt_dat)[0] + 1)
zrange = np.arange(1, np.shape(filt_dat)[1] + 1)

# initiate a 3D-array to be displayed
filt_dat_3d = np.empty([filt_dat.shape[0], filt_dat.shape[1], 3])
# exit()

# fill the 3D-array with line number and profile number and data of course
for c in range(filt_dat.shape[1]):
    for d in range(filt_dat.shape[0]):
        filt_dat_3d[d, c] = (filt_dat[d, c], d + 1, c + 1)

# # load modules for 3D plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.collections import PolyCollection
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

X = filt_dat_3d[:, :, 1].flatten()
Y = filt_dat_3d[:, :, 2].flatten()
Z = filt_dat_3d[:, :, 0].flatten()

fig = plt.figure('data in 3D', figsize=plt.figaspect(.5))
# ax = fig.gca(projection='3d')
plt.subplot(121)
# ax = Axes3D(fig)
ax = fig.add_subplot(1, 2, 1, projection='3d')
plt.title('3D')
# poly = PolyCollection(filt_d)
# ax.add_collection3d(poly, zs=range(1,np.shape(filt_d)[1]+1), zdir='z')

x = np.arange(1, len(X))
y = np.arange(1, len(Y))
# z = filt_d_3d[x,y,0]
# ax.plot(X,Y,Z,color='b')

ax.plot(X, Y, Z,
        '|', drawstyle='default', marker='.', markerfacecolor='black')

# fig = plt.figure('heatmap')
plt.subplot(122)
plt.title('heatmap')
plt.pcolormesh(filt_dat)
plt.show(block=False)
# plt.show()
# exit()

# static values for calculation
# to be set to variable values if they are provided by experiment
# c_p = 0.265  # J/(g*K)
sigma = 5.670367 * 10**(-8)  # W m-2 K-4
p = 20 * 10**(-3)  # m
# delta = 8.57 * 10**6  # g/m**3
# epsilon_ht = 0.02
s = p**2 * np.pi  # m**2
# s = 8.85 * 10**(-6) # m**2
t_a = 300  # K

t_raw = np.array(filt_dat)

# gradient is more powerfull than diff
# it calculates derivatives at the point regarding
# one point left and one point right

# calculate 1st and 2nd gradient along each profile (x direction)
tx = np.gradient(t_raw, axis=0) / np.gradient(raw_time, axis=0)
t2x = (np.gradient(np.gradient(t_raw, axis=0) /
                   np.gradient(raw_time, axis=0), axis=0) /
       np.gradient(raw_time, axis=0))
# calculate 1st derivative from profile to profile (t direction)
tt = np.gradient(t_raw, axis=1) / np.gradient(raw_time, axis=1)

# make them all the same length
t = t_raw
t_time = raw_time
# tx = t_x
# t2x = t_2x
# tt = t_t

# here be Dragons
plt.close('all')
plt.figure('t')
# print(np.shape(t))
# print(np.shape(t_time))
for i in range(np.shape(t)[1]):
    # print(np.shape(t_time[:, i]))
    # print(np.shape(t[:, i]))
    plt.plot(t_time[:, i], t[:, i])
# plt.show()
# exit()
plt.figure('tx')
plt.plot(tx)
plt.figure('t2x')
plt.plot(t2x)
plt.figure('tt')
plt.plot(tt)
plt.show(block=False)
# exit()
plt.close('all')
# here be something else

# set initial values for fitting model to experimental data
# a0 = np.ones(15)
a0 = np.full(15, 0., dtype=np.double)
# a0[4] = 0.
# a0[5] = 0.
# a0[6] = 0.

# initiate xdata and ydata as list of the same length as raw_d
xdata = list(range(len(raw_dat[0])))
ydata = xdata
# print(xdata)


def model(T, a):
    '''
    describes the model for heat capacity as a function of the temperature
    which will be used as left hand side for fitting

    with T is the temperature:
    T[0] = Tm   # temperature profile from scanning method
    T[1] = txm   # first derivative of temperature profile
    T[2] = T2xm  # second derivative of temperature profile

    and a is the fitting parameter

    fitting will be to the 15. order of a
    '''

    # return \
    #   a[0]  * T[0]**(1 - 1)  * (        T[2] + (1 - 1)  * T[1]**2 ) \
    # + a[1]  * T[0]**(2 - 2)  * ( T[0] * T[2] + (2 - 1)  * T[1]**2 ) \
    # + a[2]  * T[0]**(3 - 2)  * ( T[0] * T[2] + (3 - 1)  * T[1]**2 ) \
    # + a[3]  * T[0]**(4 - 2)  * ( T[0] * T[2] + (4 - 1)  * T[1]**2 ) \
    # + a[4]  * T[0]**(5 - 2)  * ( T[0] * T[2] + (5 - 1)  * T[1]**2 ) \
    # + a[5]  * T[0]**(6 - 2)  * ( T[0] * T[2] + (6 - 1)  * T[1]**2 ) \
    # + a[6]  * T[0]**(7 - 2)  * ( T[0] * T[2] + (7 - 1)  * T[1]**2 ) \
    # + a[7]  * T[0]**(8 - 2)  * ( T[0] * T[2] + (8 - 1)  * T[1]**2 ) \
    # + a[8]  * T[0]**(9 - 2)  * ( T[0] * T[2] + (9 - 1)  * T[1]**2 ) \
    # + a[9]  * T[0]**(10 - 2) * ( T[0] * T[2] + (10 - 1) * T[1]**2 ) \
    # + a[10] * T[0]**(11 - 2) * ( T[0] * T[2] + (11 - 1) * T[1]**2 ) \
    # + a[11] * T[0]**(12 - 2) * ( T[0] * T[2] + (12 - 1) * T[1]**2 ) \
    # + a[12] * T[0]**(13 - 2) * ( T[0] * T[2] + (13 - 1) * T[1]**2 ) \
    # + a[13] * T[0]**(14 - 2) * ( T[0] * T[2] + (14 - 1) * T[1]**2 ) \
    # + a[14] * T[0]**(15 - 2) * ( T[0] * T[2] + (15 - 1) * T[1]**2 )

    return \
      a[0]  *              (        T[2]                ) \
    + a[1]  *              ( T[0] * T[2] + 1  * T[1]**2 ) \
    + a[2]  * T[0]**(1)  * ( T[0] * T[2] + 2  * T[1]**2 ) \
    + a[3]  * T[0]**(2)  * ( T[0] * T[2] + 3  * T[1]**2 ) \
    + a[4]  * T[0]**(3)  * ( T[0] * T[2] + 4  * T[1]**2 ) \
    + a[5]  * T[0]**(4)  * ( T[0] * T[2] + 5  * T[1]**2 ) \
    + a[6]  * T[0]**(5)  * ( T[0] * T[2] + 6  * T[1]**2 ) \
    + a[7]  * T[0]**(6)  * ( T[0] * T[2] + 7  * T[1]**2 ) \
    + a[8]  * T[0]**(7)  * ( T[0] * T[2] + 8  * T[1]**2 ) \
    + a[9]  * T[0]**(8)  * ( T[0] * T[2] + 9  * T[1]**2 ) \
    + a[10] * T[0]**(9)  * ( T[0] * T[2] + 10 * T[1]**2 ) \
    + a[11] * T[0]**(10) * ( T[0] * T[2] + 11 * T[1]**2 ) \
    + a[12] * T[0]**(11) * ( T[0] * T[2] + 12 * T[1]**2 ) \
    + a[13] * T[0]**(12) * ( T[0] * T[2] + 13 * T[1]**2 ) \
    + a[14] * T[0]**(13) * (T[0] * T[2] + 14 * T[1]**2)


def fun(a, rhs, T):
    '''
    combining the model "model" with the right hand side of the formula
    '''
    return model(T, a) - rhs


def fun2(T, rhs, a):
    '''
    combining the model "model" with the right hand side of the formula
    changed order of arguments for possible other fitting method(?)
    '''
    return model(T, a) - rhs


def heat_cond(a, t):
    '''
    calculate the thermal conductivity as a polynom of the temperature
    for this calculation the best values from the fitting shall be used
    '''
    # tc = a[0] * t**(1. - 1) \
    #     + a[1] * t**(2. - 1) \
    #     + a[2] * t**(3. - 1) \
    #     + a[3] * t**(4. - 1) \
    #     + a[4] * t**(5. - 1) \
    #     + a[5] * t**(6. - 1) \
    #     + a[6] * t**(7. - 1) \
    #     + a[7] * t**(8. - 1) \
    #     + a[8] * t**(9. - 1) \
    #     + a[9] * t**(10. - 1) \
    #     + a[10] * t**(11. - 1) \
    #     + a[11] * t**(12. - 1) \
    #     + a[12] * t**(13. - 1) \
    #     + a[13] * t**(14. - 1) \
    #     + a[14] * t**(15. - 1)
    tc=a[0] \
        + a[1] * np.power(t, 1.) \
        + a[2] * np.power(t, 2.) \
        + a[3] * np.power(t, 3.) \
        + a[4] * np.power(t, 4.) \
        + a[5] * np.power(t, 5.) \
        + a[6] * np.power(t, 6.) \
        + a[7] * np.power(t, 7.) \
        + a[8] * np.power(t, 8.) \
        + a[9] * np.power(t, 9.) \
        + a[10] * np.power(t, 10.) \
        + a[11] * np.power(t, 11.) \
        + a[12] * np.power(t, 12.) \
        + a[13] * np.power(t, 13.) \
        + a[14] * np.power(t, 14.)
    return tc


# initiate values
c=0
rest=[]
rhs=[]
T=[]
blub=[]

# print(ttm)

# movement of point caued by thermal expansion
# w is the mass per unit length of the specimen
# w = delta * s


for c in range(0, len(tx.T)):
    '''
    for every profile
    '''
    # xx = np.array(tx)
    # yy = np.array(tx)
    tm2=t[:, c]
    txm2=tx[:, c]
    t2xm2=t2x[:, c]
    ttm=tt[:, c]

    # append values for right hand side to rhs
    # rhs.append(
    #     delta * c_p * ttm + (epsilon_ht * sigma * p * (tm2**4 - t_a**4)) / s)
    rhs.append(
        f_delta_cp(tm2) * ttm + (f_epsht(tm2) * sigma * p *
                                 (tm2**4 - t_a**4)) / s)
    blub.append(ttm)

    # print simple progress bar to see how the fitting proceeds
    # print('[' + np.shape(rhs)[0] * '==' + (len(tx.T) -
    #       np.shape(rhs)[0]) * '  ' + ']' + '  ' +
    #       str(round(float(np.shape(rhs)[0]) / len(tx.T) * 100, 1)) + '%')

    # print('\n' * 100)
    sys.stdout.write('[' + np.shape(rhs)[0] * '##' + (len(tx.T) -
                     np.shape(rhs)[0]) * '..' + ']' + '  ' +
                     '%.1f%%   \r' % (round(float(np.shape(rhs)[0]) /
                                      len(tx.T) * 100, 1)))
    sys.stdout.flush()
    # import sublime
    # sublime.status_message(str(round(float(np.shape(rhs)[0]) /
                                      # len(tx.T) * 100)))

    # os.system('clear')

    # append derivatives of temperature (x direction)
    # to one big temperature variable
    T.append([tm2, txm2, t2xm2])

    # run the least square fit
    # args are the inputs for function 'fun'
    # res, cov_x, infodict, mesg, ier = \
    #    leastsq(fun, a0, args=(rhs[c], T[c]), full_output=1)
    # res = minimize(fun, a0, args=(rhs[c], T[c]))
    res = least_squares(fun, a0, args=(rhs[c], T[c]),
                        method='trf',  # bounds=(-1, 1),
                        verbose=1, jac='3-point',
                        x_scale='jac',  # 10**(20),
                        # f_scale=10**(-8),
                        # max_nfev=2000,
                        xtol=0, #2.22044604926e-16,
                        ftol=2.22044604926e-16,
                        gtol=2.22044604926e-16,
                        loss='cauchy',
                        tr_solver='exact'
                        )
    # res = curve_fit(fun, T[c], rhs[c])
    # res = cfit(model, T[c], rhs[c])

    # append the results to variable
    rest.append(res['x'].tolist())

    # the trick to get a working leastsq is to set
    # the arguments in "args" in the correct order
    # same order as in definition of function "fun"

# up to this point everything ist straight forward
# now calculate the thermalconductivity from the fit values

# somehow resetting the calculated values in the given function
# returns values around 10**13 instead of 0

# print(rhs[0])
# figure3 = plt.figure()

# # ax = Axes3D(fig)

# # ax.plot(rhs[])

# # plt.plot(range(len(rhs[0])),rhs[0])
# # plt.plot(range(len(rhs[1])),rhs[1])
# plt.plot(range(len(rhs[:])),rhs[:])
# plt.show()

# exit()

# for g in range(0,len(tx.T)):
#     print(rest[g])
#     print(fun(rest[g],rhs[g],t[:,g]))
#     # print(np.shape(tm2))
#     # print(np.shape(txm2))
#     # print(np.shape(ttm))

# exit()


# print(list(range(0,len(tx.T))))
# plt.plot(t[1,:])
# plt.show()
rest2=[a for a in rest]
# print(type(rest2[1]))

# import pprint
# pp = pprint.PrettyPrinter(width=320)
# pp.pprint(np.array(rest2[:][5]))
# pp.pprint(np.array(rest2[:][6]))

# print(rest)
# exit()
# print(type(rest))
# print(len(res))
# print(len(tm2))
# yy = fitfunction(t,tx,t2x,popt[0],popt[1])

# # xx0 = np.array(list(range(0,len(tx.T))))
# # xx0 = np.array(list(t_raw[len(t_raw.T)/2,:]))
# # print(len(t_raw)/2+25)
# xx0 = np.mean(t_raw[(len(t_raw)/2)-25:(len(t_raw)/2)+25], axis=0)
# # print(xx0)
# # exit()
# # xx0 = np.array(list(range(len(rest)+2)))
# print(np.shape(xx0),np.shape(rest2))
# xx1 = np.diff(xx0,1)
# xx2 = np.diff(xx0,2)
# xx0 = xx0[:-1]
# xx1 = xx1[:]
# xx = zip(xx0,xx1,xx2)
# # print(xx[:3])
# # print('\n')
# # print(xx[-3:])
# yy = list(map(model,xx,rest2))

# print('\n')
# print(yy[:3])
# print('\n')
# print(yy[-3:])

# print('\n')
# print(t[len(t.T)/2,:3])
# print('\n')
# print(t[len(t.T)/2,-3:])
# print(np.shape(rest2))
# print(t[len(t.T)])

# print(np.shape(rest2))

# temp_range = np.arange(np.int(np.min(filt_dat)), np.int(np.max(filt_dat)), 1)
temp_range=c_puv
# print(temp_range)

# np.savetxt('temp_range.txt', temp_range)

tc=[]
# exit()

for c in range(1, np.shape(rest2)[0]):
    tc.append(heat_cond(rest[c], temp_range))
# print(tc)
# print(fun(rest[0], ))
# exit()
# print(len(xx),len(yy))
# print(len(xx0), len(yy))
# plt.plot(yy)
# plt.show()
# print(yy,t)
# print(np.shape(yy),np.shape(t))
# print(t[len(t)/2])
# print(t[len(t.T)/2])
# print('\n')
# print(rest)
# print('\n')
# print(t[len(t.T)/2])

# exit()

# plt2 = plt.figure()
# ax2 = Axes3D(plt2)
# ax2.plot(tc,t[len(t.T)/2,:],range(1,len(t.T)+1))
# ax = fig1.gca()

# print(np.shape(tc))
# fig2 = plt.figure()
# # plt.plot(filt_d[0],tc[0])
# for g in range(np.shape(tc)[1]-2):
#     plt.plot(filt_d[g],tc[g])
# # plt.plot(t[len(t.T)/2,:],yy)
# # plt.plot(t[len(t.T)/2,:],yy)
# # plt.xlabel('temperature [K]')
# # plt.ylabel('heat conductivity')
# # print(len(t.T))
# # plt.plot(t)
# print('ready')
# plt.show()


# fig8 = plt.figure()
# print(type(rhs))
# print(np.shape(rhs))
# print(rhs[0][np.shape(rhs)[1] / 2])
# rhs_mean = []
# for e in range(np.shape(rhs)[0]):
#     rhs_mean.append(rhs[e][np.shape(rhs)[1] / 2])

# plt.plot(np.arange(1, np.shape(rhs_mean)[0] + 1), rhs_mean)
# plt.show(block=False)
# exit()


# print(type(rhs[1]))
# print(type(np.shape(rhs)[1]))
# rhsnumber = np.arange(1,np.shape(rhs)[1]+1)
# print(np.shape(rhsnumber))
# print(np.shape(rest), np.shape(rhs), np.shape(T[1]))

# plot the whole data, the fitted and the calculated values

# from itertools import cycle
lines=["-", "--", "-.", ":"]
linecycler=cycle(lines)

# exit()
plt.figure('results', figsize=(15, 10))
# print(np.shape(T)[2])
# fig2 = plt.figure('filt_d')
ax1=plt.subplot(221)
plt.title('filt_d')
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

for f in range(np.shape(rhs)[0]):
    plt.plot(np.arange(1, np.shape(filt_dat)[0] + 1), filt_dat,
             next(linecycler))
# plt.plot(np.arange(1, np.shape(filt_d)[0] + 1), filt_d[:,0])
# plt.plot(np.arange(1, np.shape(filt_d)[0] + 1), filt_d[:,1])
# plt.show()


# fig3 = plt.figure('tt')
ax2=plt.subplot(224, sharex=ax1)
plt.title('rhs')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

for h in range(np.shape(rhs)[0]):
    plt.plot(np.arange(1, np.shape(rhs)[1] + 1), blub[h],
             next(linecycler))
    # plt.plot(np.arange(1, np.shape(rhs)[1] + 1), tt[:,h])
    # print(rhs[h],np.shape(rhs)[1])
# plt.plot(np.arange(1, np.shape(rhs)[1] + 1), tt[:,0])
# plt.show()
# exit()

# fig4 = plt.figure('rest')
ax3=plt.subplot(223, sharex=ax1)
plt.title('lhs')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

for j in range(np.shape(rhs)[0]):
    plt.plot(np.arange(1, np.shape(rhs)[1] + 1), model(T[j], rest[j]),
             next(linecycler))
# plt.show(block=False)

# fig5 = plt.figure('Ende')
# fig5, ax5 = plt.subplots()
ax4=plt.subplot(222, sharex=ax1)
plt.title('Ende')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

for k in range(np.shape(rhs)[0]):
    # plt.plot(np.arange(1, np.shape(rhs)[1] + 1), model(T[k], rest[k]))
    # plt.plot(np.arange(1, np.shape(rhs)[1] + 1), rhs[k])
    plt.plot(np.arange(1, np.shape(rhs)[1] + 1),
             np.add(model(T[k], rest[k]), - rhs[k]), next(linecycler),
             label=k)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.plot(np.arange(1, np.shape(rhs)[1] + 1), fun(rest[1], rhs[1], T[1]))
# plt.ticklabel_format(style='sci', axis='y')

plt.show(block=False)

plotall=True

if plotall is True:

    plt.figure('thermal conductivity')
    # plt.title('heat conductivity')
    # print(np.shape(temp_range), np.shape(heat_cond(rest[0], temp_range)))
    for k in range(np.shape(rest)[0]):
        plt.plot(temp_range, heat_cond(rest[k], temp_range))
        # blafu = heat_cond(rest[k], temp_range)
    # plt.show()

    # plt.figure('volumetric heat capacity')

    # # for k in range(np.shape(rest)[0]):
    # plt.plot(temp_range, f_delta_cp(temp_range))

    # plt.figure('total hemispherical emissivity')

    # plt.plot(temp_range, f_epsht(temp_range))

plt.show()

sys.stdout.flush()
sys.stdout.write('\n')

# fig2 = plt.figure('3D thermal conductivity') #, figsize=plt.figaspect(.5))
# ax2 = fig2.gca(projection='3d')
# # ax2 = plt.figure('3D thermal conductivity', projection='3d')

# X2 = range(1, np.shape(rest)[0] + 1)
# Y2 = temp_range
# Z2 = heat_cond(X2, Y2)

# ax2.plot(X2, Y2, Z2,
#         '|', drawstyle='default', marker='.', markerfacecolor='black')

# plt.show()

# exit()

# np.savetxt('temp_range.txt', temp_range)
# np.savetxt('heat_cond.txt', blafu)
# # Ab hier Spielwiese!!!
# from matplotlib.widgets import Slider, Button, RadioButtons

# sa0 = plt.axes([0.25, 0.1, 0.65, 0.03])

# a_0 = Slider(sa0, 'a[0]', -1, 1, valinit=rest[0,0])

# def update(val):
#     a[0] = a_0.val
#     fig.canvas.draw_idle()

# a_0.on_changed(update)

# plt.show()
