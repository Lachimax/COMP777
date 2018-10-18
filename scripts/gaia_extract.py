import gaia_rave_analysis as g

gaia = g.StarSet(
    "..\\data\\gaia-dr2-rave-35.csv")

gaia.sky_map(1000)
