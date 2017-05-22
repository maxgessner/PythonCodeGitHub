import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from functions import bb_cavity
from functions import special_bb_cavity
from functions import special_bb_cavity_d_l
# from pylab import *
# import pylab

e_w = np.arange(0.5, 0.999, 0.1)
length = np.arange(1., 20., 0.001)
dist = np.arange(0.3, 1., 0.001)

dist_length = np.arange(min(dist) / max(length), max(dist) / min(length), 0.1)

# ls_length = np.linspace(length[0], length[-1], length[-1] - length[0] + 1)
# ls_dist = np.linspace(dist[0], dist[-1], dist[-1] - dist[0] + 1)

# print(bb_cavity(0.9, 1, d))
# print(bb_cavity(0.9, 1, 1))
# print(length)

# exit()

result = np.empty([len(length), len(dist), 3])  # , dtype=np.longdouble())
resultlist = np.array(())
# print(np.shape(result))
# print(result[1, 1])
# print(len(length))
# print(np.shape(length)[0])

plot3D = False

if plot3D is True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)

    for i in range(len(length)):
        for j in range(len(dist)):
            result[i, j] = (length[i], dist[j],
                            # bb_cavity(0.9, length[i], dist[j]))
                            special_bb_cavity(0.9, length[i], dist[j]))
            # print(result[i, j])
            # print()
            # result.append(value)
            # resultlist = resultlist.append(result[i, j])
            # ax.plot(l[i], d[i], result[i, j])
    X = result[:, :, 0].flatten()
    Y = result[:, :, 1].flatten()
    Z = result[:, :, 2].flatten()

    # X, Y = np.meshgrid(X, Y)

    colors = cm.hsv((Z - min(Z)) / (max(Z) - min(Z)))
    # print(max(Z))
    # print(min(Z))

    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(Z)

    ax.scatter(X, Y, Z,
               '|',
               # drawstyle='default',
               marker='.',
               c=colors,
               # markerfacecolor='black'
               )
    # ax.plot(X, Y, Z,
    #         '|', drawstyle='default',
    #         marker='.',
    #         markerfacecolor=colors)
    # print(result)
    # print(np.shape(result))

    # fig = plt.figure()
    # # fig(projection='3d')

    # plt.plot(l, d, result[l, d])
    # print(result)
    # plt.show(block=False)
    plt.show()

# print(bb_cavity(0.5, 6., 0.4))
# print(special_bb_cavity(0.5, 6., 0.4))
# print(special_bb_cavity(0.5, 12., 0.4))
# print(special_bb_cavity(0.5, 6., 0.2))

plt.figure()
ax2 = plt.axes()
# e_z_a = np.empty(len(e_w) + 1)
e_z_a = np.empty([len(e_w), len(dist_length)])
# print(type(e_z_a))
# print(e_z_a)

# print(np.array(special_bb_cavity_d_l(e_w[0], dist_length)))
# print(np.array(special_bb_cavity_d_l(e_w[1], dist_length)))
# print(type(np.array(special_bb_cavity_d_l(e_w[2], dist_length))))

for k in range(len(e_w)):
    # e_z_a = np.append(e_z_a, special_bb_cavity_d_l(e_w[k], dist_length))
    # print(k)
    e_z_a[k] = np.array(special_bb_cavity_d_l(e_w[k], dist_length))
    # print(e_z_a[k])
    plt.plot(dist_length, e_z_a[k],
             label=r"$\varepsilon_{ht} = " + str(e_w[k]) + r'$')


# print(e_z_a)
# plt.plot(e_z_a[:])
ax2.arrow(0.6, 0.6, 0.1, 0.35,
          head_width=0.02, head_length=0.03, fc='k', ec='k')
ax2.text(0.45, 0.6, r'$\varepsilon_{ht} = 0.5$', fontsize=16)
ax2.text(0.55, 0.95, r'$\varepsilon_{ht} = 0.9$', fontsize=16)
ax2.set_xlabel('ratio hole diameter to cavity length d/L')
ax2.set_ylabel(r'effective emissivity $\varepsilon_{ht}^{new}$')
ax2.set_title('new black body geometry')
plt.legend(loc=3, fontsize=16)
plt.show()
