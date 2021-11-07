'''
Parameters module
'''

import os
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import matplotlib.colors

# Locations
PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent.absolute()
PROJECT_NAME = PROJECT_DIR / "gasmarketmodel"
DATA_FOLDER = PROJECT_NAME / "data"
SCENARIO_FOLDER = DATA_FOLDER / "scenarios"
PARAMS_FOLDER = DATA_FOLDER / "params"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
TEMP_FOLDER = DATA_FOLDER / "temp"
OUTPUT_TEMPLATE_FILE = PARAMS_FOLDER / "template.xlsx"
COUNTRY_SHAPE_FILE = PARAMS_FOLDER / "country_shapes.zip"

# Country names
country_translations = {
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
    "Serbien" : "Republic of Serbia",
    "Montenegro" : "Montenegro",
    "Kosovo" : "Kosovo",
    "Türkei" : "Turkey",
    "Nordmazedonien" : "North Macedonia",
    "Norwegen" : "Norway",
    "Bosnien" : "Bosnia and Herzegovina"
}

# Piped importer countries
piped_importer_countries = [
    "Norwegen",
    "Algerien",
    "Libyen",
    "Russland"
]

# Output metrics
# Label : [label_metric, colormap_metric, label_round_digits]
output_metrics = {
    "Alles" : {
        "label_metric" : "Demand",
        "label_digits" : 1,
        "colormap_metric" : "Demand",
        "show_colorscale" : True,
        "piped_imports" : 0,
        "piped_imports_digits" : 0,
        "piped_exports" : 0,
        "piped_exports_digits" : 0,
        "lng_imports" : -1,
        "lng_imports_digits" : 0,
        "aggregate_transit" : True,
        "transits" : -1,
        "transits_digits" : 0
    },
    "Preis" : {
        "colormap_metric" : "Price",
        "show_colorscale" : True
    }
}

# Label/sheet name : total_row_switch, row offset, index columns
output_dict = {
    "Demand" : [True, 0, 0],
    "Production" : [True, 0, 0],
    "Price" : [False, 0, 0],
    "Storage Volumes": [True, 0, 0],
    "Storage Levels": [True, 1, 0],
    "LNG" : [True, 8, [0, 1, 2]],
    "Piped Imports" : [True, 12, [0, 1, 2]],
    "Piped Exports" : [True, 12, [0, 1, 2]],
    "Connections" : [False, 12, [0, 1, 2]],
    "Supply Mix" : [False, 12, [0, 1]]
}

# Seasons
seasons_dict = {
    "Winter" : ["Okt-21", "Nov-21", "Dez-21", "Jan-22", "Feb-22", "Mrz-22"],
    "Sommer" : ["Apr-22", "Mai-22", "Jun-22", "Jul-22", "Aug-22", "Sep-22"]
}
seasons_df = pd.melt(pd.DataFrame.from_dict(seasons_dict), var_name = "season", value_name = "cycle").set_index("season")

# Individual countries
zoom_list = ["Deutschland", "Italien", "Schweiz"]
zoom_padding = 1

# Europe frame
europe_x = [-12, 42]
europe_y = [30, 74.5]
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
).set_crs(epsg = 4326)

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
resolution_level = 5
formatting_dict = {
    "linewidth_main" : 1 * resolution_level,
    "linewidth_sub" : 0.5 * resolution_level,
    "linewidth_importer" : 1 * resolution_level,
    "edgecolor_main" : "black",
    "edgecolor_sub" : "grey",
    "edgecolor_importer" : "grey",
    "facecolor_main" : color_dict[1],
    "facecolor_sub" : (0.95, 0.95, 0.95),
    "facecolor_importer" : color_dict[6],
    "label_fontsize" : 12 * resolution_level,
    "connection_label_fontsize" : 10 * resolution_level,
    "lng_label_fontsize" : 10 * resolution_level,
    "cmap_main" : matplotlib.colors.LinearSegmentedColormap.from_list("", [color_dict[2],color_dict[1],color_dict[3]]),
    "connection_arrow_col" : color_dict[7],
    "connection_arrow_alpha" : 1,
    "lng_connection_arrow_col" : color_dict[12],
    "lng_connection_arrow_alpha" : 1,
    "piped_import_label_col" : color_dict[6],
    "piped_import_base_linewidth" : 10 * resolution_level
}

