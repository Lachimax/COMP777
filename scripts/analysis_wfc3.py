# Code by Lachlan Marnoch, 2018

import numpy as np
import matplotlib.pyplot as plt
# import math as m
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
import random as r


def sub_stars(condition):
    subtract = np.zeros(len(B), dtype=bool)

    for i in range(4):
        for j in range(20):
            print('Subtracting stars in cell', i, ',', j)
            # This gives the number of stars to remove1 from the corresponding cell in the cluster CMD
            n = sum((B_I_cell == i) & (B_cell == j) & condition)
            # This gives the indices of stars in the corresponding cell in the cluster CMD
            ind_cell_cluster = np.nonzero((B_I_cell == i) & (B_cell == j) & in_cluster)[0]
            # If the number of stars to be removed from a cell is greater than the number of stars in the cell, just
            # remove1 them all
            if n >= len(ind_cell_cluster):
                for k in ind_cell_cluster:
                    subtract[k] = True
            else:
                # This while loop randomly selects stars in the respective cluster CMD cell and removes them until the
                # same number has been removed as is in the background CMD, or else all have been.
                k = 0
                while k < n:
                    # Pick a random index from the indices of stars in the same cell in the cluster CMD
                    rr = r.randint(0, len(ind_cell_cluster) - 1)
                    ind = ind_cell_cluster[rr]
                    # If that star has not been subtracted, do so. If it has, pick a new random index.
                    if not subtract[ind]:
                        subtract[ind] = True
                        k += 1
    print('Subtracted', sum(subtract), 'stars')
    print('Stars after decontamination:', sum(in_cluster & (subtract == False)))

    return subtract, condition


def draw_cells(ax):
    for z in B_I_grid:
        ax.plot([z, z], [B_min, B_max], c='blue')
    for z in B_grid:
        ax.plot([B_I_max, B_I_min], [z, z], c='blue')


# Pixel Scale, from FITS header
pixel_scale = 0.03962000086903572  # arcsec

# Limits of the frame in equatorial coordinates, to correct orientation
top = -71.785
bottom = -71.737
left = 16.904
right = 17.051

# Define CMD region of interest
B_I_max = 3
B_I_min = -1
B_max = 23
B_min = 17

# Importing data
print('Importing Data, thank you for your patience')
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
#data = data[data[:, 15] - data[:, 28] < B_I_max]
#data = data[data[:, 15] - data[:, 28] > B_I_min]
#data = data[data[:, 15] < B_max]
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

# In the FITS files, Right Ascension is treated as the horizontal coordinate, and Declination as the vertical. We will
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
print()
print('Finding stars within 50 arcsec of cluster centre')
pix_dist = np.sqrt((x_pix - centre_x_pix) ** 2 + (y_pix - centre_y_pix) ** 2)
equ_dist = np.sqrt((ra - centre_ra) ** 2 + (dec - centre_dec) ** 2)
dist = pix_dist * pixel_scale

# Find the stars that are within 50 arcsec of the cluster centre (in accordance with Li et al)
in_cluster = dist < 50
in_cluster_equ = np.array(equ_dist < 50. / 3600.)
print('Stars in cluster:', sum(in_cluster == True))

# Decontamination
# Method from Hu et al
print('Decontaminating')

# Divide CMD field1 into cells 0.5*0.25 mag^2
B_I_grid = np.arange(B_I_min, B_I_max, step=0.5)
B_grid = np.arange(B_min, B_max, step=0.25)

B_I_cell = np.floor(B_I / 0.5)
B_cell = np.floor((B - 18.) / 0.25)

remove1, field1 = sub_stars(condition=y_pix <= 800)
remove2, field2 = sub_stars(condition=x_pix <= 800)
remove3, field3 = sub_stars(condition=(80 <= dist) & (dist <= 100))

# PLOTS

print('Plotting')

# Sky maps

# Pixel Coordinates, showing cluster centre
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(x_pix, y_pix, c='black', marker=',', s=1)
ax1.scatter(x_pix[in_cluster], y_pix[in_cluster], c='blue', marker=',', s=1)
ax1.scatter(centre_x_pix, centre_y_pix, c='green')
ax1.axis('equal')
ax1.set_title('')
ax1.set_title('1. Star Pixel Coordinates')
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
plt.show()

# Equatorial Coordinates, showing cluster centre
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(ra, dec, c='black', marker=',', s=1)
ax2.scatter(centre_ra, centre_dec, c='green')
# ax2.set_xlim(left, right)
# ax2.set_ylim(bottom, top)
# ax2.axis('equal')
ax2.set_title('2. Star Equatorial Coordinates')
ax2.set_xlabel('Right Ascension (deg)')
ax2.set_ylabel('Declination (deg)')
plt.show(fig2)

# Histogram of angular distance from cluster centre
plt.title('3. Angular Distance from Cluster Centre')
plt.hist(dist)
plt.show()

# Plot of both cluster determination methods (pixel and WCS coordinates), in equatorial coordinates

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.axis('equal')
ax4.scatter(ra, dec, c='black', marker=',', s=1)
ax4.scatter(ra[in_cluster], dec[in_cluster], c='blue', marker=',', s=1)
ax4.scatter(ra[in_cluster_equ], dec[in_cluster_equ], c='red', marker=',', s=1)
ax4.scatter(centre_ra, centre_dec, c='green')
ax4.set_xlim(left, right)
ax4.set_ylim(bottom, top)
ax4.set_title('4. Equatorial Coordinates of Stars')
ax4.set_xlabel('Right Ascension (deg)')
ax4.set_ylabel('Declination (deg)')
plt.show(fig4)

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
ax5.set_title('5. Image Coordinates of Stars')
ax5.set_xlabel('x (arcsec)')
ax5.set_ylabel('y (arcsec)')
plt.show(fig5)

# Hertzsprung-Russell / Colour-Magnitude Diagrams

# Raw HR Diagram

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('1. Unprocessed Colour-Magnitude Diagram for stars in image')
ax1.scatter(B_I, B, c='black', marker=',', s=1)
ax1.set_xlim(0, 2)
ax1.set_ylim(23, 18)
ax1.set_xlabel('B - I Colour Index')
ax1.set_ylabel('B magnitude')
plt.show()

# HR Diagram limited to cluster stars

plt.title('2. Colour-Magnitude Diagram for stars < 50 arcsec from cluster centre')
plt.scatter(B_I[in_cluster], B[in_cluster], c='black', marker=',', s=1)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.show()

# HR Diagram with stars labelled not in cluster

plt.title('3. Colour-Magnitude Diagram for stars > 50 arcsec from cluster centre')
plt.scatter(B_I[in_cluster == False], B[in_cluster == False], c='black', marker=',', s=1)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.show()

# Demonstration of Decontamination Technique

fig7, ax7 = plt.subplots(2, 2)
draw_cells(ax7[0, 0])
ax7[0, 0].scatter(B_I[y_pix <= 800], B[y_pix <= 800], c='black', marker=',', s=1)
ax7[0, 0].set_title('Background CMD, with y <= 800')
ax7[0, 0].set_xlabel('B - I Colour Index')
ax7[0, 0].set_ylabel('B magnitude')
ax7[0, 0].set_xlim(0, 2)
ax7[0, 0].set_ylim(23, 18)

ax7[0, 1].set_title('Cluster CMD')
draw_cells(ax7[0, 1])
ax7[0, 1].scatter(B_I[in_cluster], B[in_cluster], c='black', marker=',', s=1)
ax7[0, 1].set_xlabel('B - I Colour Index')
ax7[0, 1].set_ylabel('B magnitude')
ax7[0, 1].set_xlim(0, 2)
ax7[0, 1].set_ylim(23, 18)

ax7[1, 0].set_title('Cluster CMD, subtracted stars in red')
draw_cells(ax7[1, 0])
ax7[1, 0].scatter(B_I[in_cluster & (remove1 == False)], B[in_cluster & (remove1 == False)], c='black', marker=',', s=1)
ax7[1, 0].scatter(B_I[remove1], B[remove1], c='red', marker=',', s=1)
ax7[1, 0].set_xlabel('B - I Colour Index')
ax7[1, 0].set_ylabel('B magnitude')
ax7[1, 0].set_xlim(0, 2)
ax7[1, 0].set_ylim(23, 18)

ax7[1, 1].set_title('Cluster CMD after subtraction')
draw_cells(ax7[1, 1])
ax7[1, 1].scatter(B_I[in_cluster & (remove1 == False)], B[in_cluster & (remove1 == False)], c='black', marker=',', s=1)
ax7[1, 1].set_xlabel('B - I Colour Index')
ax7[1, 1].set_ylabel('B magnitude')
ax7[1, 1].set_xlim(0, 2)
ax7[1, 1].set_ylim(23, 18)

plt.show()

# Recreate Figure 1 in Li et al

fig8, ax8 = plt.subplots(3, 3)

ax8[0, 0].scatter(B_I[in_cluster & (remove1 == False)], B[in_cluster & (remove1 == False)], c='black', marker=',', s=1)
ax8[0, 0].set_xlabel('B - I Colour Index')
ax8[0, 0].set_ylabel('B magnitude')
ax8[0, 0].set_xlim(0, 2)
ax8[0, 0].set_ylim(23, 18)

ax8[1, 0].scatter(B_I[field1], B[field1], c='black', marker=',', s=1)
ax8[1, 0].set_xlabel('B - I Colour Index')
ax8[1, 0].set_ylabel('B magnitude')
ax8[1, 0].set_xlim(0, 2)
ax8[1, 0].set_ylim(23, 18)

ax8[2, 0].axis('equal')
ax8[2, 0].scatter(x, y, c='black', marker=',', s=1)
ax8[2, 0].scatter(x[in_cluster], y[in_cluster], c='blue', marker=',', s=1)
ax8[2, 0].scatter(x[field1], y[field1], c='red', marker=',', s=1)
ax8[2, 0].set_xlabel('x (arcsec)')
ax8[2, 0].set_ylabel('y (arcsec)')

ax8[0, 1].scatter(B_I[in_cluster & (remove2 == False)], B[in_cluster & (remove2 == False)], c='black', marker=',', s=1)
ax8[0, 1].set_xlabel('B - I Colour Index')
ax8[0, 1].set_ylabel('B magnitude')
ax8[0, 1].set_xlim(0, 2)
ax8[0, 1].set_ylim(23, 18)

ax8[1, 1].scatter(B_I[field2], B[field2], c='black', marker=',', s=1)
ax8[1, 1].set_xlabel('B - I Colour Index')
ax8[1, 1].set_ylabel('B magnitude')
ax8[1, 1].set_xlim(0, 2)
ax8[1, 1].set_ylim(23, 18)

ax8[2, 1].axis('equal')
ax8[2, 1].scatter(x, y, c='black', marker=',', s=1)
ax8[2, 1].scatter(x[in_cluster], y[in_cluster], c='blue', marker=',', s=1)
ax8[2, 1].scatter(x[field2], y[field2], c='red', marker=',', s=1)
ax8[2, 1].set_xlabel('x (arcsec)')
ax8[2, 1].set_ylabel('y (arcsec)')

ax8[0, 2].scatter(B_I[in_cluster & (remove3 == False)], B[in_cluster & (remove3 == False)], c='black', marker=',', s=1)
ax8[0, 2].set_xlabel('B - I Colour Index')
ax8[0, 2].set_ylabel('B magnitude')
ax8[0, 2].set_xlim(0, 2)
ax8[0, 2].set_ylim(23, 18)

ax8[1, 2].scatter(B_I[field3], B[field3], c='black', marker=',', s=1)
ax8[1, 2].set_xlabel('B - I Colour Index')
ax8[1, 2].set_ylabel('B magnitude')
ax8[1, 2].set_xlim(0, 2)
ax8[1, 2].set_ylim(23, 18)

ax8[2, 2].axis('equal')
ax8[2, 2].scatter(x, y, c='black', marker=',', s=1)
ax8[2, 2].scatter(x[in_cluster], y[in_cluster], c='blue', marker=',', s=1)
ax8[2, 2].scatter(x[field3], y[field3], c='red', marker=',', s=1)
ax8[2, 2].set_xlabel('x (arcsec)')
ax8[2, 2].set_ylabel('y (arcsec)')

plt.show()
