import numpy as np
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'figure.subplot.hspace': 0.6}
plt.rcParams.update(params)

plt.subplot(1, 2, 1)
plt.hist(roman_as_num, bins='auto')
#plt.colorbar()
plt.xlabel('Chord')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(chord_seq, bins='auto')
#plt.colorbar()
plt.xlabel('Chord')
plt.ylabel('Count')
plt.savefig('plots/chord_hist.png')
plt.show()

# Convert to radians.
chord_rad = 2*np.pi*super_sequence / 7

chord_x = np.cos(chord_rad).reshape(-1, 1)
chord_y = np.sin(chord_rad).reshape(-1, 1)
chord_vec = np.hstack((chord_x, chord_y))
h, xedges, yedges, image = plt.hist2d(chord_vec[:, 0], chord_vec[:, 1], bins=(30, 30))
plt.colorbar()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax = plt.gca()
ax.axis('tight')
ax.set_aspect('equal', adjustable='box')