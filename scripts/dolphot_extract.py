# Code by Lachlan Marnoch 2018

import numpy as np
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

# Pixel Scale, from FITS header
pixel_scale = 0.03962000086903572  # arcsec

# Limits of the frame in equatorial coordinates, to correct orientation
top = -71.785
bottom = -71.737
left = 16.904
right = 17.051

# Importing data
print('Importing Data')
data_dir = "..//data//"

# The following two lines provide a means of converting between image pixel coordinates and sky coordinates
# (ie Right Ascension and Declination)
hdulist = fits.open(data_dir + "ibhy12050_drz.fits")
w = wcs.WCS(hdulist[2].header)

data = np.genfromtxt(data_dir + "wfc3_attempt_2")

print('Number of Stars:', len(data))
# Include bright stars only; this excludes various artifacts, galaxies, and some background stars
print('Excluding bad stars')
data = data[data[:, 10] == 1]
print('Number of Stars:', len(data))

# Trim out objects with sharpness not in the range -0.5 < sharpness < 0.5
print('Excluding stars with |sharpness| > 0.5')
data = data[data[:, 6] < 0.5]
data = data[data[:, 6] > -0.5]
print('Number of Stars:', len(data))

# Cut any stars outside of the CMD region of interest (Main Sequence Turnoff and subgiant branch).
print('Excluding stars outside CMD region of interest')
data = data[data[:, 15] - data[:, 28] < 3]
data = data[data[:, 15] - data[:, 28] > -1]
data = data[data[:, 28] < 23]
print('Number of Stars:', len(data))

print()
print('Calculating')
# F475W magnitude
B = data[:, 15]
# F814 magnitude
I = data[:, 28]

# B-I color
B_I = B - I

x_pix = data[:, 2]
y_pix = data[:, 3]
x = pixel_scale * x_pix
y = pixel_scale * y_pix

# Convert x_pix and y_pix (pixel coordinates) to world coordinates.
pixel_coords = np.array([x_pix, y_pix]).transpose()
world_coords = w.all_pix2world(pixel_coords, 1)
ra = world_coords[:, 0]
dec = world_coords[:, 1]

# In the fits files, Right Ascension is treated as the horizontal coordinate, and Declination as the vertical. We will
# continue to do so here for consistency.

# The centre of the cluster is at RA = 01h 07m 56.22s, Dec = -71deg 46' 04.40'', according to Li et al
# Convert these to degrees (because the sky coordinate system is clunky as hell)
c = SkyCoord(ra='01h07m56.22s', dec='-71d46min04.40s')
centre_ra = c.ra.deg
centre_dec = c.dec.deg
print()
print('Centre position: ')
print('RA:', centre_ra)
print('DEC:', centre_dec)

# Find the centre of the cluster in pixel coordinates.
centre = w.all_world2pix([[centre_ra, centre_dec]], 1)
centre_x_pix = centre[0, 0]
centre_y_pix = centre[0, 1]
centre_x = centre_x_pix * pixel_scale
centre_y = centre_y_pix * pixel_scale
print('x (pixels):', centre_x_pix)
print('y (pixels):', centre_y_pix)

# Calculate angular distance of each star from centre of cluster.
print('Finding stars within 50 arcsec of cluster centre')
pix_dist = np.sqrt((x_pix - centre_x_pix) ** 2 + (y_pix - centre_y_pix) ** 2)
equ_dist = np.sqrt((ra - centre_ra) ** 2 + (dec - centre_dec) ** 2)
dist = pix_dist * pixel_scale

# Find the stars that are within 50 arcsec of the cluster centre (in accordance with Li et al)
in_cluster = dist < 50
in_cluster_equ = np.array(equ_dist < 50. / 3600.)
print('Stars in cluster:', sum(in_cluster == True))


# PLOTS

print('Plotting')

# Sky maps

# Pixel Coordinates, showing cluster centre
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.scatter(x_pix, y_pix, c='black', marker=',', s=1)
ax4.scatter(x_pix[in_cluster], y_pix[in_cluster], c='blue', marker=',', s=1)
ax4.scatter(centre_x_pix, centre_y_pix, c='green')
ax4.axis('equal')
ax4.set_title('')
ax4.set_title('Star Pixel Coordinates')
ax4.set_xlabel('x (pixels)')
ax4.set_ylabel('y (pixels)')
plt.show()

# Equatorial Coordinates, showing cluster centre
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(ra, dec, c='black', marker=',', s=1)
ax2.scatter(centre_ra, centre_dec, c='green')
# ax2.set_xlim(left, right)
# ax2.set_ylim(bottom, top)
# ax2.axis('equal')
ax2.set_title('Star Equatorial Coordinates')
ax2.set_xlabel('Right Ascension ($^\circ$)')
ax2.set_ylabel('Declination ($^\circ$)')
plt.show(fig2)

# Histogram of angular distance from cluster centre
plt.title('Angular Distance from Cluster Centre')
plt.hist(dist)
plt.show()

# Plot of both cluster determination methods (pixel and WCS coordinates), in equatorial coordinates

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.axis('equal')
ax3.scatter(ra, dec, c='blue', marker=',', s=1)
ax3.scatter(ra[in_cluster], dec[in_cluster], c='blue', marker=',', s=1)
ax3.scatter(ra[in_cluster_equ], dec[in_cluster_equ], c='red', marker=',', s=1)
ax3.scatter(centre_ra, centre_dec, c='green')
ax3.set_xlim(left, right)
ax3.set_ylim(bottom, top)
ax3.set_title('Equatorial Coordinates of Stars')
ax3.set_xlabel('Right Ascension ($^\circ$)')
ax3.set_ylabel('Declination ($^\circ$)')
plt.show(fig3)

# The same again, but in image (pixel*pixel scale) coordinates

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.axis('equal')
ax5.scatter(x, y, c='black', marker=',', s=1)
ax5.scatter(x[in_cluster], y[in_cluster], c='blue', marker=',', s=1)
ax5.scatter(x[in_cluster_equ], y[in_cluster_equ], c='red', marker=',', s=1)
ax5.scatter(centre_x, centre_y, c='green')
ax5.set_title('')
# ax5.set_xlim(left, right)
# ax5.set_ylim(bottom, top)
ax5.set_title('Image Coordinates of Stars')
ax5.set_xlabel('x (arcsec)')
ax5.set_ylabel('y (arcsec)')
plt.show(fig5)

# Hertzsprung-Russell / Colour-Magnitude Diagrams

# Raw HR Diagram

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(B_I, B, c='black', marker=',', s=1)
ax1.set_xlim(0, 2)
ax1.set_ylim(23, 18)
ax1.set_xlabel('B - I Colour Index')
ax1.set_ylabel('B magnitude')
ax1.set_title('Unprocessed Colour-Magnitude Diagram for stars in image')
plt.show()

# HR Diagram limited to cluster stars

plt.scatter(B_I[in_cluster], B[in_cluster], c='black', marker=',', s=1)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.title('Colour-Magnitude Diagram for stars < 50 arcsec from cluster centre')
plt.show()

# HR Diagram with stars labelled not in cluster

plt.scatter(B_I[in_cluster == False], B[in_cluster == False], c='black', marker=',', s=1)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.title('Colour-Magnitude Diagram for stars > 50 arcsec from cluster centre')
plt.show()

# HR Diagram for

# plt.scatter(B_I[in_cluster_equ == False], B[in_cluster_equ == False])
# plt.xlim(0, 2)
# plt.ylim(23, 18)
# plt.xlabel('B - I Colour Index')
# plt.ylabel('B magnitude')
# plt.show()
