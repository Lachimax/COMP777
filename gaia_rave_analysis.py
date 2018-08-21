# Code by Lachlan Marnoch 2018; some code adapted from pywebofworlds module, also by Lachlan Marnoch

# This code is designed to work with the dataset "Stars from Gaia DR2 and RAVE DR5", by José H. Solórzano
# The dataset is available here: https://www.kaggle.com/solorzano/rave-dr5-gaia-dr2-consolidated


import pandas as pd
import matplotlib.pyplot as plt
import math


class Star:
    def __init__(self):
        # Gaia DR2 source ID
        self.source_id = None
        # Right ascension
        self.ra = None
        # Declination
        self.dec = None
        # Proper motion in right ascension [milliarcseconds / year]
        self.pmra = None
        # Proper motion in declination [milliarcseconds / year]
        self.pmdec = None
        # Galactic longitude [degrees]
        self.l = None
        # Galactic latitude [degrees]
        self.b = None
        # Parallax [milliarcseconds]
        self.parallax = None
        # Parallax error [milliarcseconds]
        self.parallax_error = None
        # G magnitude
        self.phot_g_mean_mag = None
        # BP magnitude
        self.phot_bp_mean_mag = None
        # RP magnitude
        self.phot_rp_mean_mag = None
        # Radial velocity from RAVE
        self.hrv = None
        # Metallicity
        self.metallicity = None
        # Spectrophotometric distance [parsecs?]
        # TODO: Verify units
        self.distance = None
        # Spectrophotometric parallax from RAVE DR5
        self.r_parallax = None
        # J magnitude from 2MASS
        self.jmag_2mass = None
        # H magnitude from 2MASS
        self.hmag_2mass = None
        # K magnitude from 2MASS
        self.kmag_2mass = None
        # Mg abundance
        self.mg = None
        # Si abundance
        self.si = None
        # Fe abundance
        self.fe = None
        # RAVE quality flag
        self.r_quality = None
        # RAVE right ascension
        self.r_ra = None
        # RAVE declination
        self.r_dec = None
        # W1 magnitude from ALLWISE
        self.w1mag_allwise = None
        # W2 magnitude from ALLWISE
        self.w2mag_allwise = None
        # W3 magnitude from ALLWISE
        self.w3mag_allwise = None
        # W4 magnitude from ALLWISE
        self.w4mag_allwise = None
        # B magnitude from APASS DR9
        self.bmag_apass = None
        # V magnitude from APASS DR9
        self.vmag_apass = None
        # RP magnitude from APASS DR9
        self.rpmag_apass = None
        # I magnitude from DENIS
        self.imag_denis = None
        # J magnitude from DENIS
        self.jmag_denis = None
        # K magnitude from DENIS
        self.kmag_denis = None
        # Absolute magnitue
        self.abs_mag = None

    def absolute_magnitude(self):
        self.abs_mag = self.vmag_apass - 5 * math.log10(self.distance / 10)


class StarSet:
    def __init__(self, path: "str" = None):

        self.catalogue = None
        self.star_list = []

        if path is not None:
            self.read_csv(path)

    def read_csv(self, path: "str" = "data\gaia-dr2-rave-35.csv"):
        # Being verbose, because I like to know which stage a program is up to
        print("Importing CSV file")

        # Read catalogue as pandas dataframe, because that seems to be the best existing module for reading CSV files
        cat = pd.read_csv(path)
        # Convert dataframe to a numpy matrix, because that's what I know how to navigate
        self.catalogue = cat.as_matrix()
        print("CSV file imported")
        for row in self.catalogue:
            star = Star()
            # Gaia DR2 source ID
            star.source_id = int(row[0])
            print("Extracting " + str(star.source_id))
            # Right ascension
            star.ra = row[1]
            # Declination
            star.dec = row[2]
            # Proper motion in right ascension [milliarcseconds / year]
            star.pmra = row[3]
            # Proper motion in declination [milliarcseconds / year]
            star.pmdec = row[4]
            # Galactic longitude [degrees]
            star.l = row[5]
            # Galactic latitude [degrees]
            star.b = row[6]
            # Parallax [milliarcseconds]
            star.parallax = row[7]
            # Parallax error [milliarcseconds]
            star.parallax_error = row[8]
            # G magnitude
            star.phot_g_mean_mag = row[9]
            # BP magnitude
            star.phot_bp_mean_mag = row[10]
            # RP magnitude
            star.phot_rp_mean_mag = row[11]
            # Radial velocity from RAVE
            star.hrv = row[12]
            # Metallicity
            star.metallicity = row[13]
            # Spectrophotometric distance [parsecs?]
            star.distance = row[14]
            # Spectrophotometric parallax from RAVE DR5
            star.r_parallax = row[15]
            # J magnitude from 2MASS
            star.jmag_2mass = row[16]
            # H magnitude from 2MASS
            star.hmag_2mass = row[17]
            # K magnitude from 2MASS
            star.kmag_2mass = row[18]
            # Mg abundance
            star.mg = row[19]
            # Si abundance
            star.si = row[20]
            # Fe abundance
            star.fe = row[21]
            # RAVE quality flag
            star.r_quality = row[22]
            # RAVE right ascension
            star.r_ra = row[23]
            # RAVE declination
            star.r_dec = row[24]
            # W1 magnitude from ALLWISE
            star.w1mag_allwise = row[25]
            # W2 magnitude from ALLWISE
            star.w2mag_allwise = row[26]
            # W3 magnitude from ALLWISE
            star.w3mag_allwise = row[27]
            # W4 magnitude from ALLWISE
            star.w4mag_allwise = row[28]
            # B magnitude from APASS DR9
            star.bmag_apass = row[29]
            # V magnitude from APASS DR9
            star.vmag_apass = row[30]
            # RP magnitude from APASS DR9
            star.rpmag_apass = row[31]
            # I magnitude from DENIS
            star.imag_denis = row[32]
            # J magnitude from DENIS
            star.jmag_denis = row[33]
            # K magnitude from DENIS
            star.kmag_denis = row[34]

            star.absolute_magnitude()

            self.star_list.append(star)

        print("Read complete")

    # TODO: HR Diagram method

    def sky_map(self, n=None):
        sky_map = plt.figure()
        ax = sky_map.add_subplot(111)
        if n is not None:
            trunc_sl = self.star_list[:n]
        else:
            trunc_sl = self.star_list
        for i, s in enumerate(trunc_sl):
            print("Plotting ", i, s.source_id)
            ax.scatter(x=s.ra, y=s.dec)
        plt.show()
