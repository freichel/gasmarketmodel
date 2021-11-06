'''
Module to evaluate scenario results
'''

import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from params import OUTPUT_FOLDER, TEMP_FOLDER, europe_frame, COUNTRY_SHAPE_FILE, country_translations, piped_importer_countries, output_metrics, formatting_dict, output_dict, resolution_level, zoom_list, zoom_padding
from PIL import Image
import pickle
from shapely.geometry import Polygon

'''
Read in parameters
'''
params_dict = pd.read_excel(io = OUTPUT_FOLDER / "params.xlsx", sheet_name = None, index_col = 0)

'''
Mapping preparations
'''
# Read in world data
world_gdf = gpd.read_file(COUNTRY_SHAPE_FILE)[["ADMIN", "geometry"]]
# Restrict to countries within defined Europe rectangle
europe_gdf = world_gdf.overlay(europe_frame, how = "intersection")
# Find Crimea and attribute it to Ukraine
russia_gdf = europe_gdf[europe_gdf["ADMIN"] == "Russia"].explode(index_parts = False)
russia_gdf["area"] = russia_gdf.geometry.area
russia_gdf.sort_values("area", ascending = False, inplace = True)
crimea_gdf = gpd.GeoDataFrame(geometry = gpd.GeoSeries(russia_gdf.iloc[1]["geometry"]))
crimea_gdf["ADMIN"] = "Ukraine"
europe_gdf = europe_gdf.explode(index_parts = False).overlay(crimea_gdf, how = "symmetric_difference")[["ADMIN_1", "geometry"]]
europe_gdf = europe_gdf.append(crimea_gdf)[["ADMIN_1", "geometry"]].fillna("Ukraine").dissolve(by = "ADMIN_1").reset_index().rename(columns = {"ADMIN_1" : "Country"})
# Translation mapping
europe_gdf = europe_gdf.merge(
    pd.DataFrame.from_dict(
        country_translations,
        orient = "index",
        columns = ["c_name"]
    ).reset_index(),
    left_on = "Country",
    right_on = "c_name",
)[["index", "geometry"]].set_index("index")

# Countries in regions
countries_gdf = europe_gdf.merge(
    params_dict["Mapping"],
    left_index = True,
    right_on = ["Region"]
)
# Regions
regions_gdf = countries_gdf.dissolve(by = "Master").drop(columns = ["Region"])
regions_gdf["labelpoint"] = regions_gdf["geometry"].apply(lambda x: x.representative_point().coords[:][0])
# Re-index Countries
countries_gdf = countries_gdf.set_index("Region")
countries_gdf["labelpoint"] = countries_gdf["geometry"].apply(lambda x: x.representative_point().coords[:][0])
# Piped importers
piped_importers_gdf = europe_gdf.merge(
    pd.DataFrame(piped_importer_countries).set_index(0),
    left_index = True,
    right_index = True
)
piped_importers_gdf["labelpoint"] = piped_importers_gdf["geometry"].apply(lambda x: x.representative_point().coords[:][0])
# Piped importer connections
piped_importers_connections_df = params_dict["C Piped Importers Conn"].reset_index().merge(
    params_dict["C Piped Importers"],
    on = "Importer"
).set_index(["Name", "Importer", "Region"])

# Base map
# Set up plot
fig, ax = plt.subplots(figsize = (14 * resolution_level, 18 * resolution_level))

# Plot Europe
europe_gdf.plot(
    ax = ax,
    edgecolor = formatting_dict["edgecolor_sub"],
    linewidth = formatting_dict["linewidth_sub"],
    facecolor = formatting_dict["facecolor_sub"]
    )

# Plot Piped importers
piped_importers_gdf.plot(
    ax = ax,
    edgecolor = formatting_dict["edgecolor_importer"],
    linewidth = formatting_dict["linewidth_importer"],
    facecolor = formatting_dict["facecolor_importer"]
    )

# Save base plot
pickle.dump(fig, open(TEMP_FOLDER / "base.pkl", "wb"))

# Zoom maps
for zoom_country in zoom_list:
    # Find outer bounds of country shape
    zoom_frame_orig = regions_gdf.loc[zoom_country]["geometry"].bounds
    # Define frame with padding
    zoom_frame = gpd.GeoDataFrame(
        geometry = gpd.GeoSeries(
            Polygon([
                [zoom_frame_orig[0] - zoom_padding, zoom_frame_orig[1] - zoom_padding],
                [zoom_frame_orig[0] - zoom_padding, zoom_frame_orig[3] + zoom_padding],
                [zoom_frame_orig[2] + zoom_padding, zoom_frame_orig[3] + zoom_padding],
                [zoom_frame_orig[2] + zoom_padding, zoom_frame_orig[1] - zoom_padding]
            ]),
            index = ["Europe"]
        )
    )
    # Aspect ratio
    zoom_aspect_ratio = (zoom_frame_orig[3] - zoom_frame_orig[1] + 2 * zoom_padding) / (zoom_frame_orig[2] - zoom_frame_orig[0] + 2 * zoom_padding)
    # Zoom in on region
    zoom_europe_gdf = europe_gdf.overlay(zoom_frame, how = "intersection")
    # Countries in regions
    zoom_countries_gdf = countries_gdf.overlay(zoom_frame, how = "intersection")
    # Regions
    zoom_regions_gdf = regions_gdf.overlay(zoom_frame, how = "intersection")
    # Piped importers
    zoom_piped_importers_gdf = piped_importers_gdf.overlay(zoom_frame, how = "intersection")
    
    # Set up plot
    zoom_fig, zoom_ax = plt.subplots(figsize = (14 * resolution_level, 14 * resolution_level * zoom_aspect_ratio))
    
    # Plot Europe
    zoom_europe_gdf.plot(
        ax = zoom_ax,
        edgecolor = formatting_dict["edgecolor_sub"],
        linewidth = formatting_dict["linewidth_sub"],
        facecolor = formatting_dict["facecolor_sub"]
    )
        
    # Plot Piped importers
    if not zoom_piped_importers_gdf.empty:
        zoom_piped_importers_gdf.plot(
            ax = zoom_ax,
            edgecolor = formatting_dict["edgecolor_importer"],
            linewidth = formatting_dict["linewidth_importer"],
            facecolor = formatting_dict["facecolor_importer"]
        )
        
    # Plot the country
    regions_gdf.filter(items = [zoom_country], axis = 0).plot(
        ax = zoom_ax,
        edgecolor = formatting_dict["edgecolor_main"],
        linewidth = formatting_dict["linewidth_main"],
        facecolor = formatting_dict["facecolor_main"]
    )
    
    # Save zoom plot
    pickle.dump(zoom_fig, open(TEMP_FOLDER / f"{zoom_country}.pkl", "wb"))
    
    
# Find all scenarios
scenarios = [scenario for scenario in os.listdir(OUTPUT_FOLDER / "scenarios") if os.path.isdir(os.path.join(OUTPUT_FOLDER / "scenarios", scenario)) and os.path.isfile(OUTPUT_FOLDER / "scenarios" /  scenario / "output.xlsx")]

# Iterate over all scenarios
for scenario_index, scenario in enumerate(scenarios):
    # Read in data for scenario
    # Initialise empty dictionary
    output_df_dict = {}
    
    # Read in data from each sheet
    for output_metric, output_params in output_dict.items():
        output_df_dict[output_metric] = pd.read_excel(
            io = OUTPUT_FOLDER / "scenarios" /  scenario / "output.xlsx",
            sheet_name = output_metric,
            skiprows = 4 + output_params[1],
            index_col = output_params[2]
        )
        # Drop Total row
        if output_params[0]:
            output_df_dict[output_metric].drop("Total", inplace = True)
    
    # Piped Import Flows with coordinates
    output_df_dict["Piped Imports"] = piped_importers_connections_df.merge(
        output_df_dict["Piped Imports"],
        left_index = True,
        right_index = True
    )
    
    for metric, metric_params in output_metrics.items():
        # Create folder if needed
        try:
            os.mkdir(OUTPUT_FOLDER / "scenarios" / scenario / metric)
        except FileExistsError:
            pass
        # Parameters
        label_metric = metric_params[0]
        colormap_metric = metric_params[1]
        label_round_digits = metric_params[2]
        
        if isinstance(label_metric, str):
            # Merge with relevant data
            label_gdf = regions_gdf.merge(
                output_df_dict[label_metric],
                left_index = True,
                right_index = True
            )
        if isinstance(colormap_metric, str):
            colormap_gdf = regions_gdf.merge(
                output_df_dict[label_metric],
                left_index = True,
                right_index = True
            )
        
        # Iterate over cycles
        for cycle in params_dict["Cycles"].index.values[1:]:
            # Plot each cycle - load base plot
            scenario_fig = pickle.load(open(TEMP_FOLDER / "base.pkl", "rb"))
            scenario_ax = scenario_fig.axes[0]
            
            if isinstance(label_metric, str):
                # Plot labels
                for region_index, region in label_gdf.iterrows():
                    scenario_ax.annotate(
                        xy = region["labelpoint"],
                        text = f"{region[cycle]: .{label_round_digits}f}",
                        horizontalalignment = "center",
                        fontsize = formatting_dict["label_fontsize"],
                    )
                
            if isinstance(colormap_metric, str):
                # Plot colormap
                colormap_gdf.plot(
                    ax = scenario_ax,
                    column = cycle,
                    #TODO
                    cmap = formatting_dict["cmap_main"],
                    edgecolor = formatting_dict["edgecolor_main"],
                    linewidth = formatting_dict["linewidth_main"]
                )
                # Show color scale if label metric isn't the same as colormap metric
                #TODO add this back
                #if label_metric != colormap_metric:
                cax = scenario_fig.add_axes([0.15, 0.81, 0.7, 0.02])
                cbar = scenario_fig.colorbar(mappable = scenario_ax.collections[-1], cax = cax, orientation = "horizontal")
                for t in cbar.ax.get_xticklabels():
                    t.set_fontsize(formatting_dict["label_fontsize"] * resolution_level)
            else:
                # Plot single color
                colormap_gdf.plot(
                ax = scenario_ax,
                column = cycle,
                facecolor = formatting_dict["facecolor_main"],
                edgecolor = formatting_dict["edgecolor_main"],
                linewidth = formatting_dict["linewidth_main"]
                )
                
            # Connection flows
            for connection_flow_index, connection_flow in output_df_dict["Connections"].iterrows():
                # Only plot non-zeros
                if connection_flow[cycle] != 0:
                    connection_params = params_dict["C Connections"].loc[connection_flow_index[0]]
                    # Arrow direction
                    if connection_flow[cycle] > 0:
                        arrow_dir = "larrow"
                    else:
                        arrow_dir = "rarrow"
                    arrow_dir += ", pad = 0.1"
                    scenario_ax.text(
                        x = connection_params["Longitude"],
                        y = connection_params["Latitude"],
                        rotation = connection_params["Angle"],
                        s = f"{connection_flow[cycle]: .1f}",
                        ha = "center",
                        va = "center",
                        size = formatting_dict["label_fontsize"],
                        bbox = dict(
                            boxstyle = arrow_dir,
                            fc = formatting_dict["connection_arrow_col"],
                            alpha = 0.5,
                            ec = "none"
                        )
                    )
                    
            # LNG flows
            for lng_connection_flow_index, lng_connection_flow in output_df_dict["LNG"].iterrows():
                # Only plot non-zeros
                if lng_connection_flow[cycle] != 0:
                    lng_connection_params = params_dict["C LNG Importers Conn"].loc[lng_connection_flow_index[0]]
                    scenario_ax.text(
                        x = lng_connection_params["Longitude"],
                        y = lng_connection_params["Latitude"],
                        rotation = lng_connection_params["Angle"],
                        s = f"{lng_connection_flow[cycle]: .1f}",
                        ha = "center",
                        va = "center",
                        size = formatting_dict["label_fontsize"],
                        bbox = dict(
                            boxstyle = "larrow, pad = 0.1",
                            fc = formatting_dict["lng_connection_arrow_col"],
                            alpha = 1,
                            ec = "none"
                        )
                    )
            
            # Piped import flows
            for piped_import_flow_index, piped_import_flow in output_df_dict["Piped Imports"].iterrows():
                if piped_import_flow[cycle] > 0:
                    total_piped_importer_flow = output_df_dict["Piped Imports"].groupby(level = ["Importer"]).sum().loc[piped_import_flow_index[1]][cycle]
                    scenario_ax.annotate(
                        text = "",
                        xytext = (piped_import_flow["Longitude"], piped_import_flow["Latitude"]),
                        xy = (piped_import_flow["Destination Longitude"], piped_import_flow["Destination Latitude"]),
                        arrowprops = dict(
                            arrowstyle = "-|>",
                            color = formatting_dict["piped_import_label_col"],
                            connectionstyle = "angle3, angleA = 0, angleB = 90",
                            linewidth = formatting_dict["piped_import_base_linewidth"] * piped_import_flow[cycle] / total_piped_importer_flow,
                            shrinkB = 20 * resolution_level
                        ),
                        fontsize = formatting_dict["label_fontsize"]
                    )
                    boxstyle = f"round, pad = {round(0.5 * piped_import_flow[cycle] / total_piped_importer_flow, 2)}"
                    scenario_ax.text(
                        x = piped_import_flow["Destination Longitude"],
                        y = piped_import_flow["Destination Latitude"],
                        s = f"{piped_import_flow[cycle]: .2f}",
                        ha = "center",
                        va = "baseline",
                        size = formatting_dict["label_fontsize"],
                        bbox = dict(
                            boxstyle = boxstyle,
                            fc = formatting_dict["piped_import_label_col"]
                        )
                    )
            
            plt.draw()

            # Massage data and save file
            data = np.fromstring(scenario_fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(scenario_fig.canvas.get_width_height()[::-1] + (3,))[290 * resolution_level:-265 * resolution_level, 200 * resolution_level:-200 * resolution_level, :]
            img = Image.fromarray(data).convert("RGBA")
            newData = [(255, 255, 255, 0) if item[0] == 255 and item[1] == 255 and item[2] == 255 else item for item in img.getdata()]
            img.putdata(newData)
            img.save(OUTPUT_FOLDER / "scenarios" / scenario / metric / f"{scenario}_{metric}_{cycle}.png", "PNG")
            
            plt.close()