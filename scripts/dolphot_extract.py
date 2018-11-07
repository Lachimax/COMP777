# Code by Lachlan Marnoch 2018

import numpy as np
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

data_dir = "..//data//"

# The following two lines provide a means of converting between image pixel coordinates and sky coordinates
# (ie Right Ascension and Declination)
hdulist = fits.open(data_dir + "ibhy12050_drz.fits")
w = wcs.WCS(hdulist[2].header)

data = np.genfromtxt(data_dir + "wfc3_attempt_2")

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
# data = data[data[:, 28] - data[:, 41] < 3]
# print(len(data))
# data = data[data[:, 28] - data[:, 41] > -1]
# print(len(data))
# data = data[data[:, 28] < 23]
# print(len(data))

# F475W magnitude
B = data[:, 15]
# F814 magnitude
I = data[:, 28]

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

# HR Diagram

# plt.scatter(B_I, B)
# plt.xlim(0, 2)
# plt.ylim(23, 18)
# plt.xlabel('B - I Colour Index')
# plt.ylabel('B magnitude')
# plt.show()

# In the fits files, Right Ascension is treated as the horizontal coordinate, and Declination as the vertical. We will
# continue to do so here for consistency.

# Sky map (pixel coordinates)
plt.scatter(x, y)
plt.show()

# The centre of the cluster is at RA = 01h 07m 56.22s, Dec = -71deg 46' 04.40'', according to Li et al
# Convert these to degrees (because the sky coordinate system is clunky as hell)
c = SkyCoord(ra='01h07m56.22s', dec='-71d46min04.40s')
centre_ra = c.ra.deg
centre_dec = c.dec.deg
print('RA:', centre_ra)
print('DEC:', centre_dec)

#Limits of the frame, to correct the orientation.
top = -71.785
bottom = -71.737
left = 16.904
right = 17.051

# Sky coordinates map
plt.scatter(ra, dec)
plt.scatter(centre_ra, centre_dec)
plt.xlim(left, right)
plt.ylim(bottom, top)
plt.autoscale(enable=False)
plt.show()

# Calculate angular distance of each star from centre of cluster
dist = np.sqrt((ra - centre_ra) ** 2 + (dec - centre_dec) ** 2)
plt.hist(dist)
plt.show()

# Select stars only within 50 arcsec of centre
in_cluster = np.array(dist < 50. / 3600.)
print(in_cluster)

plt.scatter(ra[in_cluster], dec[in_cluster], c='red')
plt.scatter(ra[in_cluster], dec[in_cluster], c='blue')
# plt.scatter(centre_ra, centre_dec, c='green')
plt.xlim(left, right)
plt.ylim(bottom, top)
plt.autoscale(enable=False)
plt.show()

# HR Diagram limited to cluster stars

plt.scatter(B_I[in_cluster], B[in_cluster])
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.show()

print(sum(in_cluster == True))
