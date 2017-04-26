def getcalibration(pyroname):
    import platform
    import sys
    import pandas as pd
    import numpy as np

    # value = 0
    # err = 0
    pyroname = pyroname.upper()
    directory = ''
    filename = ''

    # specify directory and filename to get the calibration data from

    if platform.system() == 'Windows':
        directory = 'i:\THERMISCHE ANALYSE\Messmethoden\PISA\ '\
                    'PISA_Labor\Kalibrationsmessungen'
        # changed directory for testing on Linux Laptop
        # which is not connected to ZAE
        filename = directory + '\_Kalibration_Pyrometer.txt'
    elif platform.system() == 'Linux':
        directory = ''
        filename = directory + '_Kalibration_Pyrometer.txt'
    else:
        sys.exit('It is recommened to use Linux! (or Windows if you have to)')
    # print(filename)
    # read the data using pandas
    data = pd.read_csv(filename, delimiter='\t', decimal='.', engine='c')

    value = np.array(data['value/mV'][data['pyrometer'] == pyroname])
    error = np.array(data['error/mV'][data['pyrometer'] == pyroname])
    # pyrometers = data['pyrometer'].index[pyroname.upper()]
    # print(pyrometers[pyrometers == pyroname])

    value /= 1000   # change from mV to V
    error /= 1000

    # print(type(pyrometers))

    if pyroname in data['pyrometer']:
        print(data['value/mV'])

    return (value, error)
# passt


def cal_pyrotemp(rawdata, pyrometer):
    import numpy as np

    C_2 = 0.014388  # m*K      constant
    T_90_Cu = 1357.77  # K     melting plateau of pure copper
    pyrometer = pyrometer.upper()

    rawdata = abs(rawdata)

    # each pyrometer has its own effective wavelenth
    # which is needed for the calculation later on
    if pyrometer == 'PVM':
        Lambda = 925.2  # nm
    elif pyrometer == 'PV':
        Lambda = 903.1  # nm
    elif pyrometer == 'PUV':
        Lambda = 904.4  # nm
    else:
        Lambda = 900.0  # nm

    # get value at the melding plateau for choosen
    # pyromter from calibration file
    # #calibration data is in mV measured data in V so "/ 1000"
    (U_T90_Cu, sdU_T90_Cu) = getcalibration(pyrometer)
    # print(U_T90_Cu)
    # U_T90_Cu = U_T90_Cu / 1000
    # sdU_T90_Cu = sdU_T90_Cu / 1000

    Lambda = Lambda * 1e-9  # m
    # convert wavelenth from nm to m

    q = rawdata / U_T90_Cu   # scalar division to get simplified variable q

    # calculate real temperature from comparison with calibrated values
    # and convert it to numpy array type: float64
    T_90 = (C_2 / Lambda) / np.log(np.array(abs(((np.exp(C_2 /
                                            (Lambda * T_90_Cu)) + q - 1) / q)),
                                            dtype='float64'))

    return T_90
# passt


def cal_Cp(temp_pv, rho, time, current, s, delta, p, t_a):
    #             T = T_PV;
    #           rho = R_SPEC;
    #             t = time;
    #    epsilon_ht = 1 for Blackbody;
    #       current = CURRENT;

    #    int i = 1;
    #    double sigma;

    # initialize values for temperature, time and heatcapacity
    # T_a = T_a + 23; #ambient temperature at 23 C -> 296.15 K

    # as the pv looks into the blackbody hole in the specimen
    import numpy as np
    epsilon_ht = 1

    sigma = 5.670367e-8  # W m^-2 K^-4

    # [S,delta,p,T_a]=dialog2();

    # calculate discrete derivative of temperature
    dtemp_pv = np.diff(temp_pv)
    dtemp_pv[dtemp_pv == 0] = 0.0000000001
    dtemp_pv = np.append(dtemp_pv, dtemp_pv[-1])
    # dtemp(abs(dT)>=100) = 0;

    dtime = np.diff(time)  # and of time
    dtime[dtime == 0] = 0.0000000001
    dtime = np.append(dtime, dtime[-1])

    dtimedtemp = np.divide(dtime, dtemp_pv)
    # calculate the inverse of the first derivative of temperature over time
    # print(len(dtime))

    # calculate the heatcapacity
    Cp = dtimedtemp / delta * (((rho * (current**2)) /
                                (s**2)) - (epsilon_ht * sigma * p *
                                           ((temp_pv**4) - (t_a**4)) / s))

    return Cp
# obsolete as delta_cp_and_e_ht is now in order


def getmaterial():
    # get materialdata from file
    import sys
    # import os
    from Tkinter import Tk as tk
    import tkFileDialog as fd
    from Tkinter import Message
    from Tkinter import mainloop

    # name = ''

    def Msg(name):
        master = tk()
        master.title('Error message')
        msg = Message(master, text='unable to load ' + name + ' from file')
        # msg.config()
        msg.pack()
        # close_button = Button(master, text='OK', command=master.destroy)
        mainloop()

    window = tk()
    # window.title()
    window.withdraw()
    filename = fd.askopenfilename(filetypes=[('Text files', '*.txt')],
                                  initialdir='/home/mgessner/'
                                             'PythonCode/Materialien',
                                  title='Get materialdata from file:')
    window.destroy()

    if filename == ():
        exit()

    file = open(filename, 'r')
    # materialname = file.read(1)
    data = file.readlines()
    file.close()

    if data == []:
        sys.exit('File to load was empty, please choose another file!')

    data = list(map(str.strip, data))

    # print(len(data))

    materialname = data[0]

    if data[3].find('Length') != -1:
        length = float(data[3].split('\t')[0].replace('D', 'E'))
    else:
        length = float('nan')
        Msg('Length')
    if data[4].find('Crosssection') != -1:
        crosssection = float(data[4].split('\t')[0].replace('D', 'E'))
    else:
        crosssection = float('nan')
        Msg('Crosssection')
    if data[5].find('Density') != -1:
        density = float(data[5].split('\t')[0].replace('D', 'E'))
    else:
        density = float('nan')
        Msg('Density')
    if data[6].find('Perimeter') != -1:
        perimeter = float(data[6].split('\t')[0].replace('D', 'E'))
    else:
        perimeter = float('nan')
        Msg('Perimeter')
    if data[7].find('ambient Temperature') != -1:
        ambient_temperature = float(data[7].split('\t')[0].replace('D', 'E'))
    else:
        ambient_temperature = float('nan')
        Msg('ambient Temperature')

    return [materialname, length, crosssection, density, perimeter,
            ambient_temperature]
# passt


def epsht(time, t_pv, t_puv):
    # calculate total hemispherical emissivity
    # by comparing pv and puv data
    # eps_ht_pv = 1
    import numpy as np
    sigma = 5.670367 * 10**-8  # W m^-2 K**-4
    sigma = sigma * 10**-3  # W mm^-2 K**-4
    [name, length, crosssection, density, perimeter, t_a] = getmaterial()
    # t_pv_t = np.diff(t_pv)
    t_pv = np.array(t_pv[:-1])
    # t_puv_t = np.diff(t_puv)
    t_puv = np.array(t_puv[:-1])

    '''
    assumptinos:

    current = 0

    we are only focussing on one point
    in the middle of the specimen -> dT/dx = 0

    we are only focussing on one point in time -> dT/dt = 0

    speciman has fixed perimeter, crossection

    the only thing that changes is \varepsilon_ht
    '''
    epsht = (t_pv**4 - t_a**4) / (t_puv**4 - t_a**4)

    return epsht
# obsolete as delta_cp_and_e_ht is now in order


def defaultplot(title, xlabel, ylabel, xvalues, legend=''):
    import matplotlib.pyplot as plt
    # int figurenumber
    # if figurenumber != 0:
    #     plt.figure(figurenumber)
    # else:
    if 'legend' in locals():
        if legend != '':
            plt.legend([legend])
    # plt.figure()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(min(xvalues), max(xvalues))
# passt


def delta_cp_and_e_ht(raw_puv=[], raw_time=[], raw_pv=[], raw_pv_time=[],
                      plotresult=False):
    import numpy as np
    import matplotlib.pyplot as plt
    # import sys
    # import platform
    from scipy import interpolate
    # from functions import getcalibration
    # from functions import cal_pyrotemp

    if raw_puv == []:
        raw_puv = np.load('raw_dat.npy')
    if raw_time == []:
        raw_time = np.load('raw_time.npy')
    if raw_pv == []:
        raw_pv = np.load('raw_pv.npy')
    if raw_pv_time == []:
        raw_pv_time = np.load('raw_pv_time.npy')

    sigma = 5.670367 * 10**(-8)  # W m-2 K-4
    # p = 20 * 10**(-3)  # m
    # s = p**2 * np.pi  # m**2

    puv_means = []
    cpt = []

    c_puv = []
    c_pv = []
    c_time = []
    c_pv_time = []

    # print(np.shape(raw_time))

    for i in range(np.shape(raw_time)[1]):
        # print(np.shape(raw_time[:,i]))
        # puv_means = np.append(puv_means, (np.shape(raw_time[:, i])[0] / 2))
        puv_means = np.shape(raw_time[:, i])[0] / 2
        cpt = np.append(cpt, raw_time[int(puv_means), i])
        c_pv_time = np.append(c_pv_time, raw_pv_time[cpt[i] == raw_pv_time])
        c_pv = np.append(c_pv, raw_pv[cpt[i] == raw_pv_time])
        c_puv = np.append(c_puv, raw_puv[cpt[i] == raw_time])
        c_time = np.append(c_time, raw_time[cpt[i] == raw_time])

    # if real values from pyrometers are achieved
    # they have to be converted to kelvin!!!
    # c_pv_90 = cal_pyrotemp(c_pv, 'pv')
    # c_puv_90 = cal_pyrotemp(c_puv, 'puv')
    # for demovalues no convertion is required
    c_puv_90 = c_puv
    c_pv_90 = c_pv
    # print(c_puv_90)
    # print(cpt)
    # print(c_pv_90)
    # print(c_time)

    # plt.plot(c_time, c_puv_90)
    # plt.plot(c_time, c_pv_90)
    # plt.show()

    # t_a = 300  # K

    def cal_delta_cp(puv_90, pv_90, time, t_a=300, p=20 * 10 ^ (-3)):
        if t_a not in locals():
            t_a = 300

        if p not in locals():
            p = 20 * 10**(-3)  # m

        s = p**2 * np.pi  # m**2

        dt_puv_90 = np.diff(puv_90) / np.diff(time)
        puv_90_s = puv_90[:len(np.diff(puv_90))]
        dt_pv_90 = np.diff(pv_90) / np.diff(time)
        pv_90_s = pv_90[:len(np.diff(pv_90))]

        epsht = (pv_90_s**4 - t_a**4) / (puv_90_s**4 - t_a**4) * \
            dt_puv_90 / dt_pv_90
        # print(dt_puv_90)
        # print(dt_pv_90)

        f_epsht = interpolate.interp1d(puv_90_s, epsht, kind='linear',
                                       fill_value='extrapolate')

        delta_cp = (-1) * (dt_pv_90)**(-1) * 1 * sigma * p * \
                   (pv_90_s**4 - t_a**4) / s

        f_delta_cp = interpolate.interp1d(puv_90_s, delta_cp, kind='linear',
                                          fill_value='extrapolate')

        # calculate epsht from delta_cp values!!!

        # delta_cp = (-1) * (dt_puv_90)**(-1) * f_epsht(puv_90_s) * sigma * p * \
        #            (puv_90_s**4 - t_a**4) / s

        # f_delta_cp = interpolate.interp1d(puv_90_s, delta_cp, kind='linear',
        #                                   fill_value='extrapolate')

        return(epsht, delta_cp, f_epsht, f_delta_cp)

    (epsht, delta_cp, f_epsht, f_delta_cp) = cal_delta_cp(c_puv_90, c_pv_90,
                                                          c_time,
                                                          t_a=300, p=0.02)

    # print(delta_cp)
    # print(dir(interpolate.interp1d))
    # print(f_epsht.item())
    # print(c_puv_90)
    # print(f_epsht(1800), f_epsht(1800.0001), f_epsht(1801))


    # plt.plot(c_puv_90, fun_epsht(c_puv_90), 'r.')
    # plt.show()
    # exit()

    # # print(min(c_puv_90))

    # temp_epsht = np.arange(np.ceil(min(c_puv_90)), np.floor(max(c_puv_90)), 0.1)
    # y_epsht = fun_epsht(temp_epsht)
    # print(fun_epsht(2000))

    # # print(type(epsht_all))
    # plt.figure('epsht_all')
    if plotresult is True:
        newrange = np.arange(min(c_puv), max(c_puv))
        plt.figure('total hemispherical emissivity')
        plt.plot(newrange, f_epsht(newrange), label='epsht')
        plt.plot(c_puv_90[:-1], epsht, 'r.', label='data')
        plt.legend()

        plt.figure('volumetric heat capacity')
        plt.plot(newrange, f_delta_cp(newrange), label='delta_cp')
        plt.plot(c_puv_90[:-1], delta_cp, 'r.', label='data')
        plt.legend()
        # plt.show()
        plt.show(block=False)

    return(epsht, delta_cp, f_epsht, f_delta_cp, c_time)

# (a, b, c, d) = delta_cp_and_e_ht()
# print(a,b,c(1400),c(2300),d)
# passt


def choose_cutoff(data, time):
    import matplotlib.pyplot as plt
    import numpy as np
    # import Tkinter as tk
    import sys
    from Tkinter import Spinbox, mainloop, Tk, Button, LEFT
    # from Tkinter import *

    global choose
    choose = True
    # i = 0
    global cut
    cut = 10
    # global l_line
    # l_line = plt.axvline(cut)

    def combine_funcs(*funcs):
        def combined_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)
        return combined_func

    def destroywindows(**kwargs):
        for m in kwargs:
            m.destroy()
        for f in kwargs:
            f.close()
        # master.destroy()

    def setvalues(var, value):
        var = value
        return var

    def choosefalse():
        global choose
        choose = False
        # print(choose)

    def spintocut():
        global cut
        cut = int(cutspin.get())
        # print(cut)

    # def callback():
        # print("something")
        # global l_line
        # l_line(cut)
        # l_line = plt.axvline(cut)
        # plt.draw()
        # plt.ion()
        # plt.pause(0.05)

    # plt.figure('data')
    # plt.plot(data)
    # plt.show(block=False)
    # # plt.draw()

    while choose is True:
        plt.figure('data')
        plt.plot(data)
        l_line = plt.axvline(cut)
        plt.axvline(np.shape(data)[0] - cut)
        # plt.draw()
        plt.show(block=False)
        # plt.ion()
        # plt.pause(0.05)

        # print(choose)
        # print(cut)

        master = Tk()

        # print(type(data[0]))

        cutspin = Spinbox(master, from_=0, to=np.shape(data)[0],
                          textvariable=cut) #, command=callback)
        cutspin.delete(0)
        cutspin.insert(0, cut)

        cutspin.pack()
        # print(type(cutspin))
        # print(cutspin.get())
        applyButton = Button(master, text='apply',
                             command=combine_funcs(spintocut,
                                                   plt.close,
                                                   master.destroy))
        applyButton.pack(side=LEFT)
        goonButton = Button(master, text='go on',
                            command=combine_funcs(master.destroy,
                                                  plt.close,
                                                  choosefalse))
        goonButton.pack(side=LEFT)
        quitButton = Button(master, text='quit',
                            command=sys.exit)
        quitButton.pack(side=LEFT)

        master.mainloop()
        # Tk.update()
        # plt.update()
        # plt.show(block=False)

        # l_left.remove()
        # l_right.remove()
        # plt.draw()

    cutoff = cut

    return cutoff

# import numpy as np
# filt_dat = np.load("filt_dat.npy")
# print(type(filt_dat[0,0]))
# print(np.shape(filt_dat))
# time = np.load("raw_time.npy")
# blub = choose_cutoff(filt_dat, time)
# print(blub)


def getdata(rawdata):
    import numpy as np
    from functions import window_function
    from scipy.signal import savgol_filter
    from functions import choose_cutoff

    # ln_all = np.arange(len(rawdata))
    puv_all = np.array(rawdata['puv'])
    pv_all = np.array(rawdata['pv'])
    time_all = np.array(rawdata['time'])
    if 'current' in rawdata:
        current_all = np.array(rawdata['current'])
    else:
        current_all = []
    if 'r_spec' in rawdata:
        r_spec_all = np.array(rawdata['r_spec'])
    else:
        r_spec_all = []

    (nsplit, desc, high, low) = window_function(puv_all, 310, 360,
                                                plotdata=False,
                                                printlist=False)
    nsplit = np.insert(nsplit, 0, 0)
    nsplit = np.append(nsplit, len(puv_all))

    maxnsplit = np.int(max(np.diff(high)))

    # initiate lists to fill with splitted profiles
    raw_dat = list(range(len(high[:]) - 1))
    raw_time = list(range(len(high[:]) - 1))
    filt_dat = list(range(len(high[:]) - 1))

    # define the cutoff from each side
    # cutoff = 50
    # print(np.shape(puv_all))

    # define ambient temperature
    T_a = 300  # K

    for i in range(0, len(high) - 1):
        #
        raw_dat[i] = np.array([puv_all[high[i][0]:high[i][1]]])
        raw_time[i] = np.array([time_all[high[i][0]:high[i][1]]])

        # flatten the splitted data
        raw_dat[i] = raw_dat[i].flatten()
        raw_time[i] = raw_time[i].flatten()

        # make it the same length, all of them
        lendiff = maxnsplit - len(raw_dat[i])
        raw_dat[i] = np.insert(raw_dat[i], 0, np.ones(lendiff / 2) * T_a)
        raw_dat[i] = np.append(raw_dat[i], np.ones(lendiff / 2) * T_a)
        raw_dat[i] = np.resize(raw_dat[i], (maxnsplit))

        raw_time[i] = np.lib.pad(raw_time[i], (lendiff / 2, lendiff / 2),
                                 'reflect', reflect_type='odd')
        raw_time[i] = np.resize(raw_time[i], (maxnsplit))
        # print(np.shape(raw_time))
        # print(np.shape(raw_dat))
        # cut of several values in the beginning and in the end

        # smooth the data by using filter on them
        filt_dat[i] = savgol_filter(raw_dat[i], 15, 3, axis=0, mode='mirror')
        # link: https://en.wikipedia.org/wiki/Savitzky-Golay_filter
        # uncomment next line to get rid of the filter
        # filt_d[i] = raw_d[i]

    raw_dat = np.asarray(raw_dat).T
    raw_time = np.asarray(raw_time).T
    raw_pv = pv_all
    raw_pv_time = time_all
    filt_dat = np.asarray(filt_dat).T

    # print(np.shape(filt_dat))
    # # print(filt_dat[0][:])
    # # print(filt_dat[:][0])
    # print(type(filt_dat[0]))
    # plt.plot(filt_dat)
    # # plt.plot(filt_dat[:][1])
    # plt.show()
    # exit()
    cutoff = choose_cutoff(filt_dat, raw_time)

    raw_dat = raw_dat[cutoff:-cutoff]
    raw_time = raw_time[cutoff:-cutoff]
    filt_dat = filt_dat[cutoff:-cutoff]

    # for i in range(0, len(high) - 1):
    #     raw_dat[i] = raw_dat[i][cutoff:-cutoff]
    #     raw_time[i] = raw_time[i][cutoff:-cutoff]
    #     filt_dat[i] = filt_dat[i][cutoff:-cutoff]

    # raw_dat = np.asarray(raw_dat).T
    # raw_time = np.asarray(raw_time).T
    # raw_pv = pv_all
    # raw_pv_time = time_all
    # filt_dat = np.asarray(filt_dat).T
    # print(np.shape(raw_time))
    # print(np.shape(filt_dat))

    return(raw_dat, raw_time, raw_pv, raw_pv_time, filt_dat,
           current_all, r_spec_all)
# passt


def window_function(rawdata, lowerbound, upperbound,
                    printlist=False, plotdata=False):
    # printlist = False
    # plotdata = False
    # assumption: data starts low with peaks to higher values
    import numpy as np
    import matplotlib.pyplot as plt
    # import scipy

    # lowerindex = np.where(a[i:] > lowerbound)[0]
    # output = np.array(())
    firstupper = 0
    firstlower = 0
    lastupper = 0
    lastlower = 0
    # global middle
    middle = np.array(())
    middle = np.append(middle, 0)
    high = []
    low = []

    # condlist_l = [rawdata >= lowerbound]
    # condlist_u = [rawdata <= upperbound]

    # output_l = np.select(condlist_l, [rawdata])
    # output_u = np.select(condlist_u, [rawdata])

    # output = output_l + output_u - rawdata

    # # print(output)

    # lineoutput = np.where(rawdata == output)
    # print(lineoutput)

    slope = 'up'
    # global start
    start = 0
    step = 3
    stop = False
    description = {}
    # exit()
    i = 0
    j = 0

    if rawdata[0] <= lowerbound:
        slope = 'up'
        description['0 start at'] = 0
    elif rawdata[0] >= upperbound:
        slope = 'down'
        description['0 start'] = 0
    elif rawdata[0] > lowerbound and rawdata[0] < upperbound:
        middle = np.append(middle, 0)
        if np.where(rawdata >= upperbound)[0][0] > \
           np.where(rawdata <= lowerbound)[0][0]:
            slope = 'up'
        elif np.where(rawdata >= upperbound)[0][0] < \
             np.where(rawdata <= lowerbound)[0][0]:
            slope = 'down'

    # while i <= len(rawdata)

    while stop is False:
        if slope == 'up':
            # print(np.where(rawdata[start:] >= upperbound))

            try:
                firstupper = np.where(rawdata[start:] >=
                                      upperbound)[0][0] + start
            except IndexError:
                stop = True
                break
            # print(firstupper)

            try:
                lastlower = np.where(rawdata[start:firstupper] <=
                                     lowerbound)[0][-1] + start
            except IndexError:
                stop = True
                break
            # print(lastlower)

            middle = np.append(middle, (firstupper + lastlower) / 2)
            # print(type(middle[0]))
            start = firstupper + step
            slope = 'down'
            i += 1
            description[str(i) + '. ' + slope + ' to'] = np.int64(middle[-1])
            # low = np.append(low, [(middle[-2], middle[-1])])
            low.append((np.int(middle[-2]), np.int(middle[-1])))
            # print((firstupper + lastlower) / 2)
            # print(middle)
            # exit()

        elif slope == 'down':
            # print(start)
            try:
                firstlower = np.where(rawdata[start:] <=
                                      lowerbound)[0][0] + start
            except IndexError:
                stop = True
                break
            # print(firstlower)

            try:
                lastupper = np.where(rawdata[start:firstlower] >=
                                     upperbound)[0][-1] + start
            except IndexError:
                stop = True
                break
            # print(lastupper)
            # print(start, firstlower)

            middle = np.append(middle, (firstlower + lastupper) / 2)
            start = firstlower + step
            slope = 'up'
            j += 1
            description[str(j) + '.  ' + slope + ' to '] = np.int64(middle[-1])
            # high = np.append(high, (middle[-2], middle[-1]))
            high.append((np.int(middle[-2]), np.int(middle[-1])))
            # print(middle)
            # exit()

    # middle = (firstupper + lastlower) / 2
    # print(middle)
    # choicelist = [rawdata, rawdata]

    # output = np.select(condlist, choicelist)

    # print(output)

    middle = middle.astype(np.int64)
    # print(type(middle[0]))

    # print('high', high[1])
    # print('low', low)

    # printlist = False

    if printlist is True:
        for x in sorted(description.items()):
            print(str(x) + '\n')

    # plotdata = False

    if plotdata is True:
        plt.plot(rawdata)
        # plt.plot((0, len(rawdata)), (lower, lower))
        plt.axhline(y=lowerbound, color='g')
        plt.axhline(y=upperbound, color='r')
        for i in range(np.shape(middle)[0]):
            plt.axvline(x=middle[i], color='b')
        plt.show()

    return(middle, description, high, low)
# passt


def bb_cavity(e_w, l, d):
    import numpy as np
    e_z = np.longdouble()
    e_z = 1. - (1. - e_w) / e_w * 1. / (1. + (2. * l / d)**2)

    return e_z


def special_bb_cavity(e_w, l, d):
    import numpy as np
    e_z = np.longdouble()
    # l = np.longdouble()
    # d = np.longdouble()
    # e_w = np.longdouble()
    # e_z = (e_w + 1. - (1. - e_w) / e_w * 1. / (1. + (2. * l / d))) / 2.
    e_z = (1. - (1. - e_w) / e_w * 1. /
           (1. + (2. * l / d)**2)) * (1. - (d / l)) \
        + e_w * (d / l)

    return e_z


def special_bb_cavity_d_l(e_w, d_l):
    import numpy as np
    e_z = np.array(())
    # l = np.longdouble()
    # d = np.longdouble()
    # e_w = np.longdouble()
    # e_z = (e_w + 1. - (1. - e_w) / e_w * 1. / (1. + (2. * l / d))) / 2.
    e_z = (1. - (1. - e_w) / e_w * 1. /
           (1. + (2. * (d_l)**(-1))**2)) * (1. - d_l) \
        + e_w * d_l

    return e_z
# passt
