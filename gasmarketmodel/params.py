'''
Parameters module
'''

import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

# Locations
PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent.absolute()
PROJECT_NAME = PROJECT_DIR / "gasmarketmodel"
DATA_FOLDER = PROJECT_NAME / "data"
SCENARIO_FOLDER = DATA_FOLDER / "scenarios"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
OUTPUT_TEMPLATE_FILE = OUTPUT_FOLDER / "template.xlsx"

# Country names
country_names = {
    "Deutschland" : "Germany",
    "Niederlande" : "Netherlands",
    "Frankreich" : "France",
    "Schweiz" : "Switzerland",
    "Luxemburg" : "Luxembourg",
    "Belgien" : "Belgium",
    "Spanien" : "Spain",
    "Portugal" : "Portugal",
    "Italien" : "Italy",
    "Österreich" : "Austria",
    "Schweden" : "Sweden",
    "Finnland" : "Finland",
    "Dänemark" : "Denmark",
    "Polen" : "Poland",
    "Tschechien" : "Czechia",
    "Slowakei" : "Slovakia",
    "Litauen" : "Lithuania",
    "Lettland" : "Latvia",
    "Estland" : "Estonia",
    "Ungarn" : "Hungary",
    "Slowenien" : "Slovenia",
    "Kroatien" : "Croatia",
    "Rumänien" : "Romania",
    "Bulgarien" : "Bulgaria",
    "Griechenland" : "Greece",
    "Irland" : "Ireland",
    "UK" : "United Kingdom",
    "Malta" : "Malta",
    "Zypern" : "Cyprus",
    "Ukraine" : "Ukraine",
    "Algerien" : "Algeria",
    "Marokko" : "Morocco",
    "Tunesien" : "Tunisia",
    "Libyen" : "Libya",
    "Ägypten" : "Egypt",
    "Russland" : "Russia",
    "Weißrussland" : "Belarus",
    "Moldawien" : "Moldova",
    "Albanien" : "Albania",
    "Serbien" : "Serbia",
    "Montenegro" : "Montenegro",
    "Kosovo" : "Kosovo",
    "Türkei" : "Turkey",
    "Nordmazedonien" : "North Macedonia"    
}

# Seasons
seasons_dict = {
    #"Winter" : ["Okt-21", "Nov-21", "Dez-21", "Jan-22", "Feb-22", "Mrz-22"],
    #"Sommer" : ["Apr-22", "Mai-22", "Jun-22", "Jul-22", "Aug-22", "Sep-22"],
    "Test" : ["Aug-21", "Sep-21"]
}
seasons_df = pd.melt(pd.DataFrame.from_dict(seasons_dict), var_name = "season", value_name = "cycle").set_index("season")

# Europe frame
europe_x = [-12, 42]
europe_y = [30, 71]
europe_frame = gpd.GeoDataFrame(
    geometry = gpd.GeoSeries(
        Polygon([
            [europe_x[0], europe_y[0]],
            [europe_x[0], europe_y[1]],
            [europe_x[1], europe_y[1]],
            [europe_x[1], europe_y[0]]
        ]),
        index = ["Europe"]
    )
)

# Colors and formatting
color_dict = {
    1 : (192/255, 192/255, 255/255),
    2 : (51/255, 51/255, 153/255),
    3 : (133/255, 133/255, 194/255),
    4 : (0/255, 0/255, 255/255),
    5 : (127/255, 127/255, 127/255),
    6 : (255/255, 204/255, 0/255),
    7 : (255/255, 153/255, 0/255),
    8 : (237/255, 96/255, 9/255),
    9 : (255/255, 0/255, 0/255),
    10 : (0/255, 153/255, 0/255),
    11 : (60/255, 60/255, 160/255),
    12 : (112/255, 48/255, 160/255)
}
linewidth_main = 1
linewidth_sub = 0.5
edgecolor_main = "black"
edgecolor_sub = "grey"
