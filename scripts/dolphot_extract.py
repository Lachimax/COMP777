import numpy as np
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits
from COMP777.functions import *

data_dir = "..//data//"

# The following two lines provide a means of converting between image pixel coordinates and sky coordinates
# (ie Right Ascension and Declination)
hdulist = fits.open(data_dir + "ibhy12n3q_flt.fits")
w = wcs.WCS(hdulist[2].header)

data = np.genfromtxt(data_dir + "wfc3_attempt_1")

print(len(data))
# Include bright stars only; this excludes various artifacts, galaxies, and some background stars
data = data[data[:, 10] == 1]
print(len(data))
# Trim out objects with sharpness not in the range -0.5 < sharpness < 0.5
data = data[data[:, 6] < 0.5]
print(len(data))
data = data[data[:, 6] > -0.5]
print(len(data))

# Cut any stars outside of the CMD region of interest (Main Sequence Turnoff and subgiant branch).
data = data[data[:, 28] - data[:, 41] < 3]
print(len(data))
data = data[data[:, 28] - data[:, 41] > -1]
print(len(data))
data = data[data[:, 28] < 23]
print(len(data))

# F336W magnitude
U = data[:, 15]
# F475W magnitude
B = data[:, 28]
# F814 magnitude
I = data[:, 41]

# B-I color
B_I = B - I

x = data[:, 2]
y = data[:, 3]

# Convert x and y (pixel coordinates) to world coordinates.
pixel_coords = np.array([x, y]).transpose()
print(pixel_coords)
world_coords = w.wcs_pix2world(pixel_coords, 1)
print(world_coords)
ra = world_coords[:, 0]
dec = world_coords[:, 1]

plt.scatter(B_I, B)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.show()

plt.scatter(x, y)
plt.show()

print(ra)
print(dec)

# The centre of the cluster is at RA = 01h 07m 56.22s, Dec = -71deg 46' 04.40''
centre_ra = 1 * 15. + 7 * (15. / 60.) + 56.22 * (15. / 3600.)
centre_dec = -71. - 46. * (1. / 60.) - 04.4 * (1. / 3600.)

plt.scatter(dec, ra)
plt.scatter(centre_ra, centre_dec)
plt.show()

dist = np.sqrt((ra - centre_ra) ** 2 + (dec - centre_dec) ** 2)

# Select stars only within 50 arcsec of centre.
in_cluster = np.array(dist < 50. / 3600.)
print(in_cluster)

plt.scatter(ra[in_cluster == True], dec[in_cluster == True], c='red')
plt.scatter(ra[in_cluster == False], dec[in_cluster == False], c='blue')
plt.autoscale(enable=False)
plt.show()
