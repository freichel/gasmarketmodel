'''
Main module to create scenario outputs
'''

from params import SCENARIO_FOLDER, OUTPUT_FOLDER, OUTPUT_TEMPLATE_FILE
import excelimport as ei
import os
import pandas as pd
import pulp
import numpy as np
import shutil
import openpyxl

# Find all scenario files
print("Finding scenarios...")
scenario_file_list = []
for file in os.listdir(SCENARIO_FOLDER):
    if file.endswith(".xlsm"):
        scenario_file_list.append(SCENARIO_FOLDER / file)
print("...scenarios loaded.")
print("")

'''
Scenario-unspecific imports - use first scenario file as reference
'''
print("Importing general data...")
# Cycles 
cycles_days_dict = ei.import_excel_generic_df(SCENARIO_FOLDER / scenario_file_list[0], "Cycles", 4, 0).fillna(0).iloc[0].to_dict()
# Unit conversions
unit_conversions_df = ei.import_excel_generic_df(SCENARIO_FOLDER / scenario_file_list[0], "Unit Conversions", 6, 0)
# Forex conversions
forex_conversions_df = ei.import_excel_generic_df(SCENARIO_FOLDER / scenario_file_list[0], "Forex Conversions", 4, 0)
print("...general data imported")
print("")

print("Scenario modelling")
# Loop over scenario files
for scenario_file in scenario_file_list:
    '''
    Scenario specific imports from Excel scenario file
    '''
    # Scenario name
    scenario = ei.import_excel_cell(scenario_file, "Scenario", 4, 2)
    print(f" Modelling scenario {scenario}...")
    
    print("  Importing data...")
    # Region imports
    # Region defs
    regions_df = ei.import_excel_generic_df(scenario_file, "Regions > Index", 4, 1)
    # Region demand data
    regions_demand_df = ei.import_excel_generic_df(scenario_file, "Regions > Demand", 4, 1)
    # Region production data
    regions_production_df = ei.import_excel_generic_df(scenario_file, "Regions > Production", 4, 1)
    
    # Piped importers
    # Piped importers defs
    piped_importers_df = ei.import_excel_generic_df(scenario_file, "Piped Importers > Index", 4, 1)
    # Piped importers production
    piped_importers_production_df = ei.import_excel_generic_df(scenario_file, "Piped Importers > Production", 4, 1)
    # Piped importers connections
    piped_importers_connections_df = ei.import_excel_generic_df(scenario_file, "Piped Importers > Connections", 4, 8)
    # Piped importers connections data
    piped_importers_connections_data_df = ei.import_excel_connections_data_df(scenario_file, "Piped Importers > Data", 4, [0,1], piped_importers_connections_df)
    
    # LNG importers
    # LNG importers defs
    lng_importers_df = ei.import_excel_generic_df(scenario_file, "LNG Importers > Index", 4, 1)
    # LNG importers connections
    lng_importers_connections_df = ei.import_excel_generic_df(scenario_file, "LNG Importers > Connections", 4, 8)
    # LNG importers connections data
    lng_importers_connections_data_df = ei.import_excel_connections_data_df(scenario_file, "LNG Importers > Data", 4, [0,1])
    
    # Region connections
    # Region connections defs
    connections_df = ei.import_excel_generic_df(scenario_file, "Connections > Index", 4, 8)
    # Region connections data
    connections_data_df = ei.import_excel_connections_data_df(scenario_file, "Connections > Data", 4, [0,1])
    
    #TODO
    # Storage
    # Storage defs
    storage_df = ei.import_excel_generic_df(scenario_file, "Storage > Index", 4, 1)
    # Storage connections
    storage_connections_df = ei.import_excel_generic_df(scenario_file, "Storage > Connections", 4, 8)
    # Storage connections data
    storage_connections_data_df = ei.import_excel_connections_data_df(scenario_file, "Storage > Data", 4, [0,1])
    
    # LNG
    # LNG supply curve
    lng_supply_curve_df = ei.import_excel_generic_df(scenario_file, "LNG > Price Curve", 4, 0)
    # Other LNG demand
    lng_other_demand_dict = ei.import_excel_generic_df(scenario_file, "LNG > Other Demand", 4, 0).fillna(0).iloc[0].to_dict()
    print("  ...data imported")
    
    '''
    Define output dataframes
    '''
    output_demand_df = regions_demand_df.drop(columns = ["Source/Comments", "DemandID"]).set_index("Region")
    output_production_df = regions_production_df.drop(columns = ["Source/Comments", "ProductionID"]).set_index("Region")
    output_piped_importers_df = pd.DataFrame()
    output_lng_importers_df = pd.DataFrame()
    output_connections_df = pd.DataFrame()
    output_prices_df = pd.DataFrame()
    output_supply_mix_df = pd.DataFrame(
        0,
        pd.MultiIndex.from_product(
        [
            output_demand_df.index.values.tolist(),
            ["Production", "Piped Imports", "LNG Imports", "Imports"]
        ],
        names = ["Region", "Type"]
        ),
        []
    )
    
    
    '''
    Analysis for each cycle
    '''
    # Counter variable
    cycle_index = 0
    print("  Starting analysis...")
    for cycle, cycle_days in cycles_days_dict.items():
        print(f"   Cycle {cycle} ({cycle_index + 1} of {len(cycles_days_dict)})...")
        '''
        Model preparation
        '''
        # Set up model variables
        # Piped importers
        piped_importers_connections_dict = {}
        for piped_importers_connection_index, piped_importers_connection in piped_importers_connections_df.iterrows():
            piped_importers_connections_dict[piped_importers_connection_index] = pulp.LpVariable(
                piped_importers_connection["Name"],
                lowBound = 0,
                upBound = (
                    piped_importers_connections_data_df.loc[
                        piped_importers_connection["Name"],
                        "Capacity - Max"]
                    [cycle]
                    - piped_importers_connections_data_df.loc[
                        piped_importers_connection["Name"],
                        "Capacity - Min"]
                    [cycle]
                ) * cycle_days
            )
        # LNG importers
        lng_importers_connections_dict = {}
        for lng_importers_connection_index, lng_importers_connection in lng_importers_connections_df.iterrows():
            lng_importers_connections_dict[lng_importers_connection_index] = pulp.LpVariable(
                lng_importers_connection["Name"],
                lowBound = 0,
                upBound = (
                    lng_importers_connections_data_df.loc[
                        lng_importers_connection["Name"],
                        "Capacity - Max"]
                    [cycle]
                    - lng_importers_connections_data_df.loc[
                        lng_importers_connection["Name"],
                        "Capacity - Min"]
                    [cycle]
                ) * cycle_days
            )
        # Region connections
        connections_dict = {}
        for connection_index, connection in connections_df.iterrows():
            connections_dict[connection_index] = pulp.LpVariable(
                connection["Name"],
                lowBound = 0,
                upBound = (
                    connections_data_df.loc[
                        connection["Name"],
                        "Capacity - Max"]
                    [cycle]
                    - connections_data_df.loc[
                        connection["Name"],
                        "Capacity - Min"]
                    [cycle]
                ) * cycle_days
            )
        #TODO
        # Storage
        storage_connections_dict = {}
        for storage_connection_index, storage_connection in storage_connections_df.iterrows():
            storage_connections_dict[storage_connection_index] = pulp.LpVariable(
                storage_connection["Name"],
                lowBound = 0,
                upBound = (
                    storage_connections_data_df.loc[
                        storage_connection["Name"],
                        "Capacity - Max"]
                    [cycle]
                    - storage_connections_data_df.loc[
                        storage_connection["Name"],
                        "Capacity - Min"]
                    [cycle]
                ) * cycle_days
            )
        
        # Regional overviews
        # Dictionary ountry: [factor, connection, fixed tariff, variable tariff, lng source switch, name, source]
        regions_supply_dict = {}
        # Dictionary country: demand
        regions_demand_dict = {}
        for region_index, region in regions_df.iterrows():
            # Demand
            region_demand = regions_demand_df.loc[region_index][cycle]
            regions_demand_dict[region["Region"]] = region_demand * cycle_days
            
            # Production
            region_production = regions_production_df.loc[region_index][cycle]
            regions_supply_dict[region["Region"]] = [[1, region_production * cycle_days, 0, 0, 0, f"Production {region['Region']}", ""]]
            
            # Piped imports
            piped_importer_list = piped_importers_connections_df[piped_importers_connections_df["Region Index"] == region_index]["Name"].to_dict()
            for piped_importer_id, piped_importer in piped_importer_list.items():
                # Min flow
                regions_supply_dict[region["Region"]].append([1, piped_importers_connections_data_df.loc[piped_importer, "Capacity - Min"][cycle] * cycle_days, 0, 0, 0, f"{piped_importer} - Min", ""])
                # Variable flow
                regions_supply_dict[region["Region"]].append(
                    [
                        1,
                        piped_importers_connections_dict[piped_importer_id],
                        piped_importers_connections_data_df.loc[piped_importer, "Cost to Market - fixed"][cycle],
                        piped_importers_connections_data_df.loc[piped_importer, "Cost to Market - variable"][cycle] * unit_conversions_df.loc["mcm"]["MWh"],
                        0,
                        piped_importer,
                        "Pipe"
                    ]
                )
            
            # LNG imports
            lng_importer_list = lng_importers_connections_df[lng_importers_connections_df["Region Index"] == region_index]["Name"].to_dict()
            for lng_importer_id, lng_importer in lng_importer_list.items():
                # Min flow
                regions_supply_dict[region["Region"]].append([1, lng_importers_connections_data_df.loc[lng_importer, "Capacity - Min"][cycle] * cycle_days, 0, 0, 0, f"{lng_importer} - Min", ""])
                # Variable flow
                regions_supply_dict[region["Region"]].append(
                    [
                        1,
                        lng_importers_connections_dict[lng_importer_id],
                        lng_importers_connections_data_df.loc[lng_importer, "Cost to Market - fixed"][cycle],
                        lng_importers_connections_data_df.loc[lng_importer, "Cost to Market - variable"][cycle] * unit_conversions_df.loc["mcm"]["MWh"],
                        1,
                        lng_importer,
                        "LNG"
                    ]
                )
                
            # Region connections (imports)
            import_list = connections_df[connections_df["Destination Index"] == region_index]["Name"].to_dict()
            for importer_id, importer in import_list.items():
                # Min flow
                regions_supply_dict[region["Region"]].append([1, connections_data_df.loc[importer, "Capacity - Min"][cycle] * cycle_days, 0, 0, 0, f"{importer} - Min", ""])
                # Variable flow
                regions_supply_dict[region["Region"]].append(
                    [
                        1,
                        connections_dict[importer_id],
                        connections_data_df.loc[importer, "Tariff - fixed"][cycle],
                        connections_data_df.loc[importer, "Tariff - variable"][cycle] * unit_conversions_df.loc["mcm"]["MWh"],
                        0,
                        importer,
                        connections_df[connections_df["Name"] == importer]["Origin"].item()
                    ]
                )
            
            # Region connections (exports)
            export_list = connections_df[connections_df["Origin Index"] == region_index]["Name"].to_dict()
            for exporter_id, exporter in export_list.items():
                # Min flow
                regions_supply_dict[region["Region"]].append([-1, connections_data_df.loc[exporter, "Capacity - Min"][cycle] * cycle_days, 0, 0, 0, f"{exporter} - Min", ""])
                # Variable flow
                regions_supply_dict[region["Region"]].append([-1, connections_dict[exporter_id], 0, -connections_data_df.loc[exporter, "Tariff - variable"][cycle] * unit_conversions_df.loc["mcm"]["MWh"], 0, exporter, connections_df[connections_df["Name"] == exporter]["Destination"].item()])
            
            '''
            #TODO needs rework
            # Storage
            storage_list = storage_connections_df[storage_connections_df["Region Index"] == region_index]["Name"].to_dict()
            for storage_id, storage in storage_list.items():
                # Min Flow
                print(storage_connections_data_df.loc[storage, "Capacity - Min"][cycle])
                # Variable Flow
                print(storage_connections_dict[storage_id])
            '''
        
        '''
        Model
        '''
        # Initialise model
        cost_total = pulp.LpProblem('Cost', pulp.LpMinimize)
        
        # Volume balancing
        # Supply/demand matching
        for region_key, region_data in regions_supply_dict.items():
            # Aggregate all supply sources (minus exports) for region
            region_supply = pulp.lpSum([row[0] * row[1] for row in region_data])
            # Add constraints so that supply will exactly match demand
            cost_total += (region_supply <= regions_demand_dict[region_key])
            cost_total += (region_supply >= regions_demand_dict[region_key])
        # Piped imports
        for piped_importer_index, piped_importer in piped_importers_df.iterrows():
            # Aggregate all piped import connections
            importer_supply = pulp.lpSum(
                [piped_importers_connections_dict[piped_importers_connection] for piped_importers_connection in piped_importers_connections_df[piped_importers_connections_df["Importer Index"] == piped_importer_index].index.tolist()]
            )
            # Add constraints so that the sum of flows will not exceed production
            cost_total += (
                importer_supply
                <= (
                    piped_importers_production_df.loc[piped_importer_index][cycle]
                    - piped_importers_connections_data_df[piped_importers_connections_data_df["Importer"] == piped_importer["Importer"]].xs(
                        "Capacity - Min",
                        level = 1,
                        drop_level = False
                    )
                    [cycle].sum()) * cycle_days
            )
        # LNG
        # LNG volume
        lng_total = (
            # Demand outside markets
            lng_other_demand_dict[cycle] * unit_conversions_df.loc["Mt LNG"]["mcm"]
            # Demand
            + regions_demand_df[cycle].sum()
            # Production
            - regions_production_df[cycle].sum()
            # Piped Imports - min of available production and max capacity
            - min(
                piped_importers_production_df[cycle].sum(),
                piped_importers_connections_data_df.xs("Capacity - Max", level = 1, drop_level = False)[cycle].sum()
            )
        ) * unit_conversions_df.loc["mcm"]["Mt LNG"]
        # LNG price curve
        lng_pricelist = lng_supply_curve_df[cycle].to_frame().reset_index().values
        # LNG price at given point
        lng_price = lng_pricelist[np.argmin(np.abs(lng_pricelist[0:, 0] - lng_total)), 1] / forex_conversions_df.loc["USD"][cycle] / unit_conversions_df.loc["MMBTU"]["mcm"]
        #TODO
        # Storage
        
        # Cost calculations
        #TODO add fixed costs
        regions_cost_dict = {}
        for region_key, region_data in regions_supply_dict.items():
            # Multiply attracted supplies (1, not -1) by their respective tariffs
            # 0 is factor
            # 1 is volume
            # 3 is variable tariff
            # 4 is lng switch
            region_cost = pulp.lpSum([row[0] * row[1] * (max(row[3], 0) + row[4] * lng_price) for row in region_data if row[0] == 1])
            # Append to dictionary
            regions_cost_dict[region_key] = region_cost
        # Add up total cost and add as objective    
        cost_total += pulp.lpSum([region_data for region_key, region_data in regions_cost_dict.items()])
        
        # Solve
        res = cost_total.solve(pulp.PULP_CBC_CMD(msg=0))
        assert res == pulp.LpStatusOptimal
        print("    Solution for cycle found!")
        
        '''
        Calculation outputs
        '''
        # Store value for each model variable in dict
        model_vars_dict = {}
        for model_var in cost_total.variables():
            model_vars_dict[model_var.name] = model_var.value()
        
        # Prices
        # Price sources
        regions_sources_dict = {}
        regions_prices_dict = {}
        for region_key, region_data in regions_supply_dict.items():
            regions_sources_dict[region_key] = []
            # Check if any flows are imports. If so, we can discard exports (see below)
            positive_flag = False
            for region_vars in region_data:
                # Only variable flows
                if isinstance(region_vars[1], pulp.pulp.LpVariable):
                    # Only non-zero flows
                    if model_vars_dict[region_vars[1].name] != 0 and region_vars[3] != 0 and region_vars[6] != "Pipe":
                        # Set flag if this is a positive flow
                        if region_vars[3] >= 0:
                            positive_flag = True
                        # Source, cost per MWh imported (only variable)
                        regions_sources_dict[region_key].append(
                            [
                                region_vars[6],
                                region_vars[3] / unit_conversions_df.loc["mcm"]["MWh"]
                            ]
                        )
            # We actually only need to keep exports (negative tariffs) if no positive ones exist (see above)
            if positive_flag == True:
                temp_list = regions_sources_dict[region_key]
                regions_sources_dict[region_key] = [temp_row for temp_row in temp_list if temp_row[1] >= 0]
            regions_prices_dict[region_key] = None
        # Find prices
        # The algorithm always finds the next possible market price (i.e. one with no ambiguity)
        # The region is then removed from the dict
        # Continuous iteration until all solutions are found
        # Start with LNG
        regions_prices_dict["LNG"] = lng_price / unit_conversions_df.loc["mcm"]["MWh"]
        # Keep looping until all values have been found
        while None in regions_prices_dict.values():
            # Take a temporary copy to check that there has been progress in solving this
            temp_dict = regions_sources_dict.copy()
            # Iterate over all regions
            for region_key, region_prices in regions_sources_dict.items():
                for price_index, region_price in enumerate(region_prices):
                    # Check if market price has already been found
                    if isinstance(region_price, list):
                        if regions_prices_dict[region_price[0]] != None:
                            regions_sources_dict[region_key][price_index] = regions_prices_dict[region_price[0]] + region_price[1]
                # Now check if all values have been found, and if so save it. Then delete entry from dict and restart
                if all(isinstance(region_price, (int, float)) for region_price in region_prices):
                    regions_prices_dict[region_key] = max(region_prices)
                    del regions_sources_dict[region_key]
                    break
            # If no progress has been made
            if regions_sources_dict == temp_dict:
                # Iterate over remaining regions
                for region_key, region_prices in regions_sources_dict.items():
                    # Find first instance of an unresolved price
                    for region_price_index, region_price in enumerate(region_prices):
                        if isinstance(region_price, list):
                            # Delete it
                            del regions_sources_dict[region_key][region_price_index]
                            break
                        else:
                            continue
                    break
        
        '''
        Dataframe outputs
        '''
        # Piped imports
        for piped_importer_id, piped_importer in piped_importers_connections_dict.items():
            piped_importer_name = piped_importers_connections_df.loc[piped_importer_id]["Name"]
            # Only in first cycle
            if cycle_index == 0:
                output_piped_importers_df.at[
                    piped_importer_name,
                    "Importer"
                ] = piped_importers_connections_df.loc[piped_importer_id]["Importer"]
                output_piped_importers_df.at[
                    piped_importer_name,
                    "Region"
                ] = piped_importers_connections_df.loc[piped_importer_id]["Region"]
            output_piped_importers_df.at[
                piped_importer_name,
                cycle
            ] = piped_importer.value() / cycle_days + piped_importers_connections_data_df.loc[piped_importer_name, "Capacity - Min"][cycle]
        
        # LNG imports
        for lng_importer_id, lng_importer in lng_importers_connections_dict.items():
            lng_importer_name = lng_importers_connections_df.loc[lng_importer_id]["Name"]
            # Only in first cycle
            if cycle_index == 0:
                output_lng_importers_df.at[
                    lng_importer_name,
                    "Terminal"
                ] = lng_importers_connections_df.loc[lng_importer_id]["Terminal"]
                output_lng_importers_df.at[
                    lng_importer_name,
                    "Region"
                ] = lng_importers_connections_df.loc[lng_importer_id]["Region"]
            output_lng_importers_df.at[
                lng_importer_name,
                cycle
            ] = lng_importer.value() / cycle_days + lng_importers_connections_data_df.loc[lng_importer_name, "Capacity - Min"][cycle]
        
        # Region connections
        for connection_id, connection in connections_dict.items():
            connection_name = connections_df.loc[connection_id]["Name"]
            # Only in first cycle
            if cycle_index == 0:
                output_connections_df.at[
                    connection_name,
                    "Origin"
                ] = connections_df.loc[connection_id]["Origin"]
                output_connections_df.at[
                    connection_name,
                    "Destination"
                ] = connections_df.loc[connection_id]["Destination"]
            output_connections_df.at[
                connection_name,
                cycle
            ] = connection.value() / cycle_days + connections_data_df.loc[connection_name, "Capacity - Min"][cycle]
            
        #TODO
        # Storage
        
        # Prices
        # Only in first cycle
        if cycle_index == 0:
            output_prices_df = output_prices_df.from_dict(regions_prices_dict, orient = "index", columns = [cycle])
        else:
            output_prices_df[cycle] = pd.Series(regions_prices_dict)
        
        # Regional supply mix
        output_supply_mix_dict = {}
        output_supply_mix_df[cycle] = 0

        for region_key, region_data in regions_demand_dict.items():
            demand = output_demand_df.loc[region_key][cycle]
            exports = -output_connections_df[output_connections_df["Origin"] == region_key][cycle].sum()
            
            #TODO - may be needed
            '''
            output_supply_mix_dict["Production"] = output_production_df.loc[region_key][cycle]
            output_supply_mix_dict["Piped Imports"] = output_piped_importers_df[output_piped_importers_df["Region"] == region_key][cycle].sum()
            output_supply_mix_dict["LNG Imports"] = output_lng_importers_df[output_lng_importers_df["Region"] == region_key][cycle].sum()
            output_supply_mix_dict["Imports"] = output_connections_df[output_connections_df["Destination"] == region_key][cycle].sum()
            
            supply = sum(output_supply_mix_dict.values())
            
            for mix_key, mix_value in output_supply_mix_dict.items():
                output_supply_mix_df.loc[(region_key, mix_key), cycle] = mix_value / supply
            '''
            
            output_supply_mix_df.loc[(region_key, "Demand"), cycle] = demand
            output_supply_mix_df.loc[(region_key, "Exports"), cycle] = exports
            output_supply_mix_df.loc[(region_key, "Production"), cycle] = output_production_df.loc[region_key][cycle]
            output_supply_mix_df.loc[(region_key, "Piped Imports"), cycle] = output_piped_importers_df[output_piped_importers_df["Region"] == region_key][cycle].sum()
            output_supply_mix_df.loc[(region_key, "LNG Imports"), cycle] = output_lng_importers_df[output_lng_importers_df["Region"] == region_key][cycle].sum()
            output_supply_mix_df.loc[(region_key, "Imports"), cycle] = output_connections_df[output_connections_df["Destination"] == region_key][cycle].sum()
        
        
        print(f"   ...cycle {cycle} ({cycle_index + 1} of {len(cycles_days_dict)}) completed")
        
        
        cycle_index += 1
        #TODO temporary
        if cycle_index == 1:
            break
        
    print("  Writing data...")
    '''
    File output
    '''
    # Dictionary with all output dataframes.
    # label: [dataframe, total_row_switch, row offset]
    output_dict = {
        "Demand": [output_demand_df.sort_index(), True, 0],
        "Production": [output_production_df.sort_index(), True, 0],
        "Price": [output_prices_df.sort_index(), False, 0],
        "LNG": [output_lng_importers_df.sort_values(["Terminal", "Region"]), True, 8],
        "Piped Imports": [output_piped_importers_df.sort_values(["Importer", "Region"]), True, 12],
        "Connections": [output_connections_df.sort_values(["Origin", "Destination"]), False, 12],
        "Supply Mix": [output_supply_mix_df.sort_index().reset_index().set_index("Region"), False, 10]
    }
    # Output file name
    output_file = OUTPUT_FOLDER / (scenario + ".xlsx")
    # Copy template
    shutil.copyfile(OUTPUT_TEMPLATE_FILE, output_file)
    # Excel Writer
    wb = openpyxl.load_workbook(output_file)
    for output_key, output_data in output_dict.items():
        ws = wb[output_key]
        # Add total row if required
        if output_data[1]:
            index_col = output_data[0].append(output_data[0].sum(numeric_only = True).rename("Total")).index.tolist()
            value_cols = output_data[0].append(output_data[0].sum(numeric_only = True).rename("Total")).values.tolist()
        else:
            index_col = output_data[0].index.tolist()
            value_cols = output_data[0].values.tolist()
        index_row = output_data[0].columns.tolist()
        for row_index, row in enumerate(value_cols):
            ws.cell(row = row_index + 6 + output_data[2], column = 1, value = index_col[row_index])
            for col_index, col in enumerate(row):
                ws.cell(row = row_index + 6 + output_data[2], column = col_index + 2, value = col)
                ws.cell(row = 5 + output_data[2], column = col_index + 2, value = index_row[col_index])
    # Scenario name
    ws = wb["Scenario"]
    ws.cell(1, 2, value = scenario)

    # Save
    wb.save(output_file)     
    print("  ...data written") 
    print(" ...scenario modelled")
print("Finished") 
