'''
Module to calculate differences between scenarios
'''

import pandas as pd
import os
from params import OUTPUT_FOLDER, output_dict, OUTPUT_TEMPLATE_FILE
import shutil
import openpyxl

# Find all scenarios
scenarios = [scenario for scenario in os.listdir(OUTPUT_FOLDER / "scenarios") if os.path.isdir(os.path.join(OUTPUT_FOLDER / "scenarios", scenario)) and os.path.isfile(OUTPUT_FOLDER / "scenarios" /  scenario / "output.xlsx")]

# Initialise empty dictionary
scenario_dict = {}

# Iterate over all scenarios
for scenario_index, scenario in enumerate(scenarios):
    # Read in data for scenario
    scenario_dict[scenario] = {}
    
    # Read in data from each sheet
    for output_metric, output_params in output_dict.items():
        scenario_dict[scenario][output_metric] = pd.read_excel(
            io = OUTPUT_FOLDER / "scenarios" /  scenario / "output.xlsx",
            sheet_name = output_metric,
            skiprows = 4 + output_params[1],
            index_col = output_params[2]
        )

# Subtract scenarios
#TODO - make list
base_scenario = "Master"
alt_scenario = "Test"

# Initialise empty difference df dict
diff_df_dict = {}
for output_metric in output_dict.keys():
    diff_df_dict[output_metric] = scenario_dict[alt_scenario][output_metric].subtract(scenario_dict[base_scenario][output_metric])

# Output file name
try:
    os.mkdir(OUTPUT_FOLDER / "scenarios" / f"Delta {alt_scenario}-{base_scenario}")
except FileExistsError:
    pass
output_file = OUTPUT_FOLDER / "scenarios" / f"Delta {alt_scenario}-{base_scenario}" / "output.xlsx"
# Copy template
shutil.copyfile(OUTPUT_TEMPLATE_FILE, output_file)
# Excel Writer
wb = openpyxl.load_workbook(output_file)
for output_key, output_data in diff_df_dict.items():
    ws = wb[output_key]
    index_cols = output_data.index.tolist()
    value_cols = output_data.values.tolist()
    index_row = output_data.columns.tolist()
    for row_index, row in enumerate(value_cols):
        # Turn single-column index into list to enable for loop below
        if not isinstance(index_cols[row_index], tuple):
            index_cols[row_index] = [index_cols[row_index]]
        for index_col_index, index_col in enumerate(index_cols[row_index]):
            ws.cell(row = row_index + 6 + output_dict[output_key][1], column = index_col_index + 1, value = index_col)
            # On a multi-index, delete additional indices
            if row_index == len(value_cols) - 1 and output_dict[output_key][0] and index_col_index != 0:
                ws.cell(row = row_index + 6 + output_dict[output_key][1], column = index_col_index + 1, value = "")
            for col_index, col in enumerate(row):
                ws.cell(row = row_index + 6 + output_dict[output_key][1], column = col_index + 1 + len(index_cols[row_index]), value = col)
                ws.cell(row = 5 + output_dict[output_key][1], column = col_index + 1 + len(index_cols[row_index]), value = index_row[col_index])
                # Make total row bold
                if row_index == len(value_cols) - 1 and output_dict[output_key][0]:
                    ws.cell(row = row_index + 6 + output_dict[output_key][1], column = col_index + 1 + len(index_cols[row_index])).font = openpyxl.styles.Font(bold = True)
                
# Scenario name
ws = wb["Scenario"]
ws.cell(1, 2, value = f"Delta {alt_scenario}-{base_scenario}")

# Save
wb.save(output_file)     