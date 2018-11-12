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
    # Set seed for replicability
    r.seed(729626)
    for i in range(4):
        for j in range(20):
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


def draw_sgb_box(ax):
    ax.plot([left_sgb, left_sgb], [top_sgb, bot_sgb], c='red')
    ax.plot([left_sgb, right_sgb], [bot_sgb, bot_sgb], c='red')
    ax.plot([right_sgb, right_sgb], [top_sgb, bot_sgb], c='red')
    ax.plot([left_sgb, right_sgb], [top_sgb, top_sgb], c='red')


def draw_isochrones():
    plt.plot(B_I_iso_min, B_iso_min, c='purple', label='1 Gyrs')
    plt.plot(B_I_iso_max, B_iso_max, c='violet', label='3.09 Gyrs')
    plt.plot(B_I_138, B_138, c='blue', label='1.38 Gyrs')
    plt.plot(B_I_218, B_218, c='red', label='2.18 Gyrs')
    plt.plot(B_I_best, B_best, c='green', label='1.58 Gyrs')


def draw_all_isochrones():
    for iso in iso_list:
        plt.plot(iso[:, 29] - iso[:, 34] + B_I_offset, iso[:, 29] + DM)

def get_nearest(xx, arr):
    return np.abs(xx - arr).argmin()


def mse(yy, y_dash):
    return sum((yy - y_dash) ** 2) / len(yy)


# Distance modulus (correction for absolute madnitudes, which are the values we have in the isochrone data, to apparent
# magnitudes, the values in the DOLPHOT data) to the SMC
DM = 19.35

B_I_offset = 0.14
# B_I_offset = 0

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

# Define CMD region of subgiant branch

top_sgb = 21.3
bot_sgb = 20.25
left_sgb = 0.8
right_sgb = 1.25

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
data = data[data[:, 15] - data[:, 28] < B_I_max]
data = data[data[:, 15] - data[:, 28] > B_I_min]
data = data[data[:, 15] < B_max]
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

# Find stars in the SGB region
in_sgb = (B < top_sgb) & (B > bot_sgb) & (B_I < right_sgb) & (B_I > left_sgb)
print('SGB: ', sum(in_sgb))

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
remove4, field4 = sub_stars(condition=dist >= 80)

# Import isochrone data

isos = np.genfromtxt(data_dir + "isochrones3.dat")
# Get the list of ages used in the isochrones
ages = np.unique(isos[:, 1])
# Seperate the big bad isochrone file into
iso_list = []
for i, a in enumerate(ages):
    iso = isos[isos[:, 1] == a]
    iso_list.append(iso)

select_sgb = in_sgb & in_cluster & (remove3 == False)

# Find the points on the isochrone with the nearest x-values to our SGB stars
mses = np.zeros(len(ages))
for j, iso in enumerate(iso_list):
    y_dash = np.zeros(sum(select_sgb))
    for i, xx in enumerate(B_I[select_sgb]):
        nearest = get_nearest(xx, iso[:, 29] - iso[:, 34] + B_I_offset)
        y_dash[i] = iso[nearest, 29] + DM
    mses[j] = (mse(B[select_sgb], y_dash))

print('Selected stars in SGB:', sum(select_sgb))

# Extract some useful individual isochrones

# Our youngest isochrone
iso_min = iso_list[0]
B_iso_min = iso_min[:, 29] + DM
I_iso_min = iso_min[:, 34] + DM
B_I_iso_min = B_iso_min - I_iso_min + B_I_offset

# 1.38 Gyrs
iso_138 = iso_list[np.abs(ages - 1.38e9).argmin()]
B_138 = iso_138[:, 29] + DM
I_138 = iso_138[:, 34] + DM
B_I_138 = B_138 - I_138 + B_I_offset

# 2.18 Gyrs
iso_218 = iso_list[np.abs(ages - 2.18e9).argmin()]
B_218 = iso_218[:, 29] + DM
I_218 = iso_218[:, 34] + DM
B_I_218 = B_218 - I_218 + B_I_offset

# Our best-fitting isochrone:
iso_best = iso_list[mses.argmin()]
B_best = iso_best[:, 29] + DM
I_best = iso_best[:, 34] + DM
B_I_best = B_best - I_best + B_I_offset

# Our oldest isochrone:
iso_max = iso_list[-1]
B_iso_max = iso_max[:, 29] + DM
I_iso_max = iso_max[:, 34] + DM
B_I_iso_max = B_iso_max - I_iso_max + B_I_offset

# PLOTS

print('Plotting')

# Sky maps

# Pixel Coordinates, showing cluster centre
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Star Pixel Coordinates')
ax1.scatter(x_pix, y_pix, c='black', marker=',', s=1)
ax1.scatter(x_pix[in_cluster], y_pix[in_cluster], c='blue', marker=',', s=1)
ax1.scatter(centre_x_pix, centre_y_pix, c='green')
ax1.axis('equal')
ax1.set_title('')
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
plt.show()

# Equatorial Coordinates, showing cluster centre
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title('Star Equatorial Coordinates')
ax2.scatter(ra, dec, c='black', marker=',', s=1)
ax2.scatter(centre_ra, centre_dec, c='green')
# ax2.set_xlim(left, right)
# ax2.set_ylim(bottom, top)
# ax2.axis('equal')
ax2.set_xlabel('Right Ascension (deg)')
ax2.set_ylabel('Declination (deg)')
plt.show(fig2)

# Histogram of angular distance from cluster centre
plt.title('Angular Distance from Cluster Centre')
plt.xlabel('Angle x (arcseconds)')
plt.ylabel('Angle y (arcsecs)')
plt.hist(dist)
plt.show()

# Plot of both cluster determination methods (pixel and WCS coordinates), in equatorial coordinates

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.axis('equal')
ax4.set_title('Equatorial Coordinates of Stars')
ax4.scatter(ra, dec, c='black', marker=',', s=1)
ax4.scatter(ra[in_cluster], dec[in_cluster], c='blue', marker=',', s=1)
ax4.scatter(ra[in_cluster_equ], dec[in_cluster_equ], c='red', marker=',', s=1)
ax4.scatter(centre_ra, centre_dec, c='green')
ax4.set_xlim(left, right)
ax4.set_ylim(bottom, top)
ax4.set_xlabel('Right Ascension (deg)')
ax4.set_ylabel('Declination (deg)')
plt.show(fig4)

# The same again, but in image (pixel*pixel scale) coordinates

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.axis('equal')
ax5.set_title('Image Coordinates of Stars')
ax5.scatter(x, y, c='black', marker=',', s=1)
ax5.scatter(x[in_cluster], y[in_cluster], c='blue', marker=',', s=1)
ax5.scatter(x[in_cluster_equ], y[in_cluster_equ], c='red', marker=',', s=1)
ax5.scatter(centre_x, centre_y, c='green')
ax5.set_title('')
# ax5.set_xlim(left, right)
# ax5.set_ylim(bottom, top)
ax5.set_xlabel('x (arcsec)')
ax5.set_ylabel('y (arcsec)')
plt.show(fig5)

# Hertzsprung-Russell / Colour-Magnitude Diagrams

# Raw HR Diagram

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Unprocessed Colour-Magnitude Diagram for stars in image')
ax1.scatter(B_I, B, c='black', marker=',', s=1)
ax1.set_xlim(0, 2)
ax1.set_ylim(23, 18)
ax1.set_xlabel('B - I Colour Index')
ax1.set_ylabel('B magnitude')
plt.show()

# HR Diagram limited to cluster stars

plt.title('Colour-Magnitude Diagram for stars < 50 arcsec from cluster centre')
plt.scatter(B_I[in_cluster], B[in_cluster], c='black', marker=',', s=1)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.show()

# HR Diagram with stars labelled not in cluster

plt.title('Colour-Magnitude Diagram for stars > 50 arcsec from cluster centre')
plt.scatter(B_I[in_cluster == False], B[in_cluster == False], c='black', marker=',', s=1)
plt.xlim(0, 2)
plt.ylim(23, 18)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.show()

# Demonstration of Decontamination Technique

fig7, ax7 = plt.subplots(2, 2)
ax7[0, 0].set_title('Background CMD, with y <= 800')
draw_cells(ax7[0, 0])
ax7[0, 0].scatter(B_I[field4], B[field4], c='black', marker=',', s=1)
ax7[0, 0].set_xlabel('B - I Colour Index')
ax7[0, 0].set_ylabel('B magnitude')
ax7[0, 0].set_xlim(0, 2)
ax7[0, 0].set_ylim(23, 18)
draw_sgb_box(ax7[0, 0])

ax7[0, 1].set_title('Cluster CMD')
draw_cells(ax7[0, 1])
ax7[0, 1].scatter(B_I[in_cluster], B[in_cluster], c='black', marker=',', s=1)
ax7[0, 1].set_xlabel('B - I Colour Index')
ax7[0, 1].set_ylabel('B magnitude')
ax7[0, 1].set_xlim(0, 2)
ax7[0, 1].set_ylim(23, 18)
draw_sgb_box(ax7[0, 1])

ax7[1, 0].set_title('Cluster CMD, subtracted stars in red')
draw_cells(ax7[1, 0])
ax7[1, 0].scatter(B_I[in_cluster & (remove4 == False)], B[in_cluster & (remove4 == False)], c='black', marker=',', s=1)
ax7[1, 0].scatter(B_I[remove4], B[remove4], c='red', marker=',', s=1)
ax7[1, 0].set_xlabel('B - I Colour Index')
ax7[1, 0].set_ylabel('B magnitude')
ax7[1, 0].set_xlim(0, 2)
ax7[1, 0].set_ylim(23, 18)
draw_sgb_box(ax7[1, 0])

ax7[1, 1].set_title('Cluster CMD after subtraction')
draw_cells(ax7[1, 1])
ax7[1, 1].scatter(B_I[in_cluster & (remove4 == False)], B[in_cluster & (remove4 == False)], c='black', marker=',', s=1)
ax7[1, 1].set_xlabel('B - I Colour Index')
ax7[1, 1].set_ylabel('B magnitude')
ax7[1, 1].set_xlim(0, 2)
ax7[1, 1].set_ylim(23, 18)
draw_sgb_box(ax7[1, 1])

plt.show()

# Recreate Figure 1 in Li et al

fig8, ax8 = plt.subplots(3, 4)

ax8[0, 0].scatter(B_I[in_cluster & (remove1 == False)], B[in_cluster & (remove1 == False)], c='black', marker=',', s=1)
ax8[0, 0].set_xlabel('B - I Colour Index')
ax8[0, 0].set_ylabel('B magnitude')
ax8[0, 0].set_xlim(0, 2)
ax8[0, 0].set_ylim(23, 18)
draw_sgb_box(ax8[0, 0])

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
draw_sgb_box(ax8[0, 1])

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
draw_sgb_box(ax8[0, 2])

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

ax8[0, 3].scatter(B_I[in_cluster & (remove4 == False)], B[in_cluster & (remove4 == False)], c='black', marker=',', s=1)
ax8[0, 3].set_xlabel('B - I Colour Index')
ax8[0, 3].set_ylabel('B magnitude')
ax8[0, 3].set_xlim(0, 2)
ax8[0, 3].set_ylim(23, 18)
draw_sgb_box(ax8[0, 3])

ax8[1, 3].scatter(B_I[field3], B[field3], c='black', marker=',', s=1)
ax8[1, 3].set_xlabel('B - I Colour Index')
ax8[1, 3].set_ylabel('B magnitude')
ax8[1, 3].set_xlim(0, 2)
ax8[1, 3].set_ylim(23, 18)

ax8[2, 3].axis('equal')
ax8[2, 3].scatter(x, y, c='black', marker=',', s=1)
ax8[2, 3].scatter(x[in_cluster], y[in_cluster], c='blue', marker=',', s=1)
ax8[2, 3].scatter(x[field4], y[field4], c='red', marker=',', s=1)
ax8[2, 3].set_xlabel('x (arcsec)')
ax8[2, 3].set_ylabel('y (arcsec)')

plt.show()

# Show isochrones

plt.title('Colour-Magnitude Diagram with isochrones')
plt.scatter(B_I[in_cluster & (remove3 == False)], B[in_cluster & (remove3 == False)], c='black', marker=',', s=1)
plt.scatter(B_I[select_sgb], B[select_sgb], c='violet',
            marker=',', label='SGB stars')
draw_isochrones()
draw_sgb_box(plt)
plt.xlim(0.1, 1.5)
plt.ylim(22.5, 19.5)
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.legend()
plt.show()

plt.title('Colour-Magnitude Diagram with isochrones')
plt.xlabel('B - I Colour Index')
plt.ylabel('B magnitude')
plt.scatter(B_I[in_cluster & (remove3 == False)], B[in_cluster & (remove3 == False)], c='black', marker=',', s=1)
draw_all_isochrones()
draw_sgb_box(plt)
plt.scatter(B_I[select_sgb], B[select_sgb], c='violet',
            marker=',', label='SGB stars')
plt.xlim(0.1, 1.5)
plt.ylim(22.5, 19.5)
plt.legend()
plt.show()

# Plot MSEs of isochrones

plt.title("Mean Squared Error of Isochrones to the Subgiant Branch")
plt.plot(ages, mses)
plt.xlabel('Age')
plt.ylabel('Mean Squared Error')
plt.show()

print('Optimum age:', ages[mses.argmin()])
