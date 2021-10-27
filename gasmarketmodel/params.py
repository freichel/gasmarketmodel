'''
Parameters module
'''

import os
from pathlib import Path

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
    "Frankreich" : "France",
    "Niederlande" : "Netherlands",
    "Schweiz" : "Switzerland",
    "Luxemburg" : "Luxembourg"
}