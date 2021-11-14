'''
Module to evaluate scenario results
'''

from numpy.lib.arraysetops import isin
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from params import OUTPUT_FOLDER, TEMP_FOLDER, PARAMS_FOLDER, europe_frame, COUNTRY_SHAPE_FILE, country_translations, piped_importer_countries, output_metrics, formatting_dict, output_dict, resolution_level, zoom_list, zoom_padding
from PIL import Image
import pickle
from shapely.geometry import Polygon, LineString, Point

def reset_angles(df_row):
    '''
    Returns rotated angle
    '''
    if 90 < df_row["Angle"] <= 270:
        if df_row["Angle"] < 180:
            return df_row["Angle"] + 180
        return df_row["Angle"] - 180
    return df_row["Angle"]

def compare_angles(df_row):
    '''
    If the angles don't match, -1 is returned
    '''
    if df_row["Angle"] == df_row["new_angle"]:
        return 1
    return -1


'''
Read in parameters
'''
params_dict = pd.read_excel(io = OUTPUT_FOLDER / "params.xlsx", sheet_name = None, index_col = 0)
# Add proper direction to transit and LNG flows
params_dict["C Connections"]["new_angle"] = params_dict["C Connections"].apply(reset_angles, axis = 1)
params_dict["C Connections"]["dir"] = params_dict["C Connections"].apply(compare_angles, axis = 1)
params_dict["C LNG Importers Conn"]["new_angle"] = params_dict["C LNG Importers Conn"].apply(reset_angles, axis = 1)
# Collect import pipeline routes
temp_import_pipes_gdf = gpd.read_file(PARAMS_FOLDER / "piped_imports.geojson")
import_pipes_gdf = temp_import_pipes_gdf[temp_import_pipes_gdf["geometry"].apply(isinstance, args = (LineString,))].overlay(europe_frame, how = "intersection").merge(
    params_dict["C Piped Importers Conn"],
    left_on = "name",
    right_index = True
).set_index("name")
import_pipes_gdf["points"] = import_pipes_gdf.apply(lambda x: [y for y in x["geometry"].coords], axis=1)
# TODO work on labels more
import_pipes_labels_gdf = temp_import_pipes_gdf[temp_import_pipes_gdf["geometry"].apply(isinstance, args = (Point,))].overlay(europe_frame, how = "intersection").set_index("name")
import_pipes_labels_gdf["points"] = import_pipes_labels_gdf.apply(lambda x: [y for y in x["geometry"].coords], axis=1)

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
crimea_gdf = gpd.GeoDataFrame(geometry = gpd.GeoSeries(russia_gdf.iloc[1]["geometry"])).set_crs(epsg = 4326)
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
fig, ax = plt.subplots(figsize = (14 * resolution_level, 18.9 * resolution_level))

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

ax.set_axis_off()
ax.margins(0, 0)
fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
fig.tight_layout()

# Extent of Europe y axis
europe_dy = ax.get_ylim()[1] - ax.get_ylim()[0]

# Save base plot
pickle.dump(fig, open(TEMP_FOLDER / "base.pkl", "wb"))

# Zoom maps
for zoom_country in zoom_list:
    # Find outer bounds of country shape
    zoom_frame_orig = regions_gdf.loc[zoom_country]["geometry"].bounds
    zoom_aspect_ratio = (zoom_frame_orig[2] - zoom_frame_orig[0] + 2 * zoom_padding) * np.cos((zoom_frame_orig[3] + zoom_frame_orig[1]) / 2 * np.pi / 180) / (zoom_frame_orig[3] - zoom_frame_orig[1] + 2 * zoom_padding)
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
    ).set_crs(epsg = 4326)
    # Zoom in on region
    zoom_europe_gdf = europe_gdf.overlay(zoom_frame, how = "intersection")
    # Countries in regions
    zoom_countries_gdf = countries_gdf.overlay(zoom_frame, how = "intersection")
    # Regions
    zoom_regions_gdf = regions_gdf.overlay(zoom_frame, how = "intersection")
    # Piped importers
    zoom_piped_importers_gdf = piped_importers_gdf.overlay(zoom_frame, how = "intersection")
    
    # Set up plot
    zoom_fig, zoom_ax = plt.subplots(figsize = (14 * resolution_level, 14 * resolution_level / zoom_aspect_ratio))
    
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
    
    zoom_ax.set_axis_off()
    zoom_ax.margins(0, 0)
    zoom_fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    zoom_fig.tight_layout()
    
    # Save zoom plot
    pickle.dump(zoom_fig, open(TEMP_FOLDER / f"{zoom_country}.pkl", "wb"))

# Find all scenarios
scenarios = [scenario for scenario in os.listdir(OUTPUT_FOLDER / "scenarios") if os.path.isdir(os.path.join(OUTPUT_FOLDER / "scenarios", scenario)) and os.path.isfile(OUTPUT_FOLDER / "scenarios" /  scenario / "output.xlsx")]

# Iterate over all scenarios
for scenario_index, scenario in enumerate(scenarios):
    # Read in data for scenario
    # Initialise empty dictionary
    output_df_dict = {}
    
    # Create folder
    try:
        os.mkdir(OUTPUT_FOLDER / "scenarios" / scenario / "Markt")
    except FileExistsError:
        pass
    
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
    
    # Add consolidated transit flows
    output_df_dict["Connections_cons"] = params_dict["C Connections"].merge(
        output_df_dict["Connections"],
        left_index = True,
        right_index = True
    )
    
    # Consolidate LNG flows where terminals sit very close to each other
    output_df_dict["LNG_cons"] = params_dict["C LNG Importers Conn"].merge(
        output_df_dict["LNG"],
        left_index = True,
        right_index = True
    ).groupby(["Latitude", "Longitude", "Angle"]).sum().drop(columns = ["new_angle"]).merge(
        params_dict["C LNG Importers Conn"],
        left_index = True,
        right_on = ["Latitude", "Longitude", "Angle"]
    ).drop_duplicates(["Latitude", "Longitude"]).set_index("Region")
    
    # Consolidate Piped Import flows where end points are the same
    output_df_dict["Piped_import_cons"] = output_df_dict["Piped Imports"].groupby(["Destination Latitude", "Destination Longitude"]).sum().merge(
        output_df_dict["Piped Imports"][["Destination Latitude", "Destination Longitude"]].drop_duplicates(["Destination Latitude", "Destination Longitude"]),
        left_index = True,
        right_on = ["Destination Latitude", "Destination Longitude"]
    ).drop(columns = ["Latitude", "Longitude", "Destination Latitude", "Destination Longitude"])
    
    # Add "Pipeline" (Imports + transit imports)
    output_df_dict["Pipeline"] = output_df_dict["Supply Mix"].loc[pd.IndexSlice[:, ["Piped Imports", "Transit Imports"]], :].groupby(["Region"]).sum()
    # Add "Supply" ("Pipeline" + LNG imports)
    output_df_dict["Supply Mix"] = pd.concat(
        [
            output_df_dict["Supply Mix"],
            output_df_dict["Supply Mix"].loc[pd.IndexSlice[:, ["Pipeline", "LNG Imports"]], :].groupby(["Region"]).sum().assign(Type = "Supply").reset_index().set_index(["Region", "Type"])
        ]
    ).sort_index(level = 0)
    output_df_dict["Supply"] = output_df_dict["Supply Mix"].loc[pd.IndexSlice[:, ["Piped Imports", "Transit Imports", "LNG Imports"]], :].groupby(["Region"]).sum()
    # Add "Piped Share" ("Pipeline"/"Supply")
    output_df_dict["Piped Share"] = output_df_dict["Pipeline"] / output_df_dict["Supply"] * 100
        
    for metric, metric_params in output_metrics.items():
        # Parameters
        label_metric = metric_params.get("label_metric", None)
        colormap_metric = metric_params.get("colormap_metric", None)
        label_round_digits = metric_params.get("label_digits", None)
        colormap_digits = metric_params.get("colormap_digits", 0)
        colormap_percentage = "%%" if metric_params.get("colormap_percentage", False) else ""
        zoom_creation = metric_params.get("zoom_creation", False)
        
        # Create folder if needed
        try:
            os.mkdir(OUTPUT_FOLDER / "scenarios" / scenario / "Markt" / metric)
        except FileExistsError:
            pass
        if zoom_creation:
            zoom_dict = {}
            for zoom_country in zoom_list:
                zoom_dict[zoom_country] = {}
                # Create folders if needed
                try:
                    os.mkdir(OUTPUT_FOLDER / "scenarios" / scenario / zoom_country)
                except FileExistsError:
                    pass
                try:
                    os.mkdir(OUTPUT_FOLDER / "scenarios" / scenario / zoom_country / metric)
                except FileExistsError:
                    pass
        
        
        
        # Merge with relevant data
        # Label data
        if isinstance(label_metric, str):
            label_gdf = regions_gdf.merge(
                output_df_dict[label_metric],
                left_index = True,
                right_index = True
            )
        # Colormap data
        if isinstance(colormap_metric, str):
            colormap_gdf = regions_gdf.merge(
                output_df_dict[colormap_metric],
                left_index = True,
                right_index = True
            )
        
        # Iterate over cycles
        for cycle in params_dict["Cycles"].index.values[1:]:
            # Plot each cycle - load base plot
            scenario_fig = pickle.load(open(TEMP_FOLDER / "base.pkl", "rb"))
            scenario_ax = scenario_fig.axes[0]
            
            if zoom_creation:
                for zoom_country in zoom_list:
                    zoom_dict[zoom_country]["fig"] = pickle.load(open(TEMP_FOLDER / f"{zoom_country}.pkl", "rb"))
                    zoom_dict[zoom_country]["ax"] = zoom_dict[zoom_country]["fig"].axes[0]
            
            
            if isinstance(label_metric, str):
                # Plot labels
                for region_index, region in label_gdf.iterrows():
                    scenario_ax.annotate(
                        xy = region["labelpoint"],
                        text = f"{region[cycle]: ,.{label_round_digits}f}".replace(",", ";").replace(".", ",").replace(";", "."),
                        horizontalalignment = "center",
                        fontsize = formatting_dict["label_fontsize"],
                    )
                
            if isinstance(colormap_metric, str):
                # Plot colormap
                colormap_gdf.plot(
                    ax = scenario_ax,
                    column = cycle,
                    cmap = formatting_dict["cmap_main"],
                    edgecolor = formatting_dict["edgecolor_main"],
                    linewidth = formatting_dict["linewidth_main"]
                )
                # Show color scale if requested
                if metric_params.get("colorscale_orientation", None) == "horizontal":
                    cax = scenario_fig.add_axes([0.03, 0.95, 0.94, 0.02])
                    cbar = scenario_fig.colorbar(mappable = scenario_ax.collections[-1], cax = cax, orientation = "horizontal", format = f"%.{colormap_digits}f{colormap_percentage}")
                    for t in cbar.ax.get_xticklabels():
                        t.set_fontsize(formatting_dict["label_fontsize"])
                elif metric_params.get("colorscale_orientation", None) == "vertical":
                    x0, x1 = scenario_ax.get_xlim()
                    scenario_ax.set_xlim(x0 - 4, x1 + 6)
                    cax = scenario_fig.add_axes([0.88, 0.1, 0.04, 0.8])
                    cbar = scenario_fig.colorbar(mappable = scenario_ax.collections[-1], cax = cax, orientation = "vertical", format = f"%.{colormap_digits}f{colormap_percentage}")
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(formatting_dict["label_fontsize"])
                    scenario_fig.set_figheight(16.5)
                    scenario_fig.set_figwidth(14)
            else:
                # Plot single color
                label_gdf.plot(
                    ax = scenario_ax,
                    column = cycle,
                    facecolor = formatting_dict["facecolor_main"],
                    edgecolor = formatting_dict["edgecolor_main"],
                    linewidth = formatting_dict["linewidth_main"]
                    )
                
            # Piped import flows
            if isinstance(metric_params.get("piped_imports", None), (int, float)):
                for piped_import_flow_index, piped_import_flow in output_df_dict["Piped_import_cons"].iterrows():
                    # Only plot required values
                    if piped_import_flow[cycle] > metric_params["piped_imports"]:
                        segments = import_pipes_gdf.loc[piped_import_flow_index[0]]["points"]
                        segment_count = len(segments)
                        for segment_index, segment in enumerate(segments):
                            if segment_index > 0:
                                arrowstyle = f"-|>, head_width = 0.8, head_length = 0.5" if segment_index == segment_count - 1 else "-"
                                scenario_ax.annotate(
                                    "",
                                    xy = (segment[0], segment[1]),
                                    xytext = (segments[segment_index - 1][0], segments[segment_index - 1][1]),
                                    arrowprops = dict(
                                        arrowstyle = arrowstyle,
                                        linewidth = formatting_dict["pipe_label_fontsize"],
                                        color = formatting_dict["piped_import_label_col"],
                                        alpha = formatting_dict["piped_import_arrow_alpha"]
                                    )
                                )
                    
                    if zoom_creation:
                        for zoom_country in zoom_list:
                            x_range = zoom_dict[zoom_country]["ax"].get_xlim()
                            y_range = zoom_dict[zoom_country]["ax"].get_ylim()
                            if piped_import_flow_index[2] == zoom_country and piped_import_flow[cycle] > metric_params["piped_imports"]:
                                segments = import_pipes_gdf.loc[piped_import_flow_index[0]]["points"]
                                segment_count = len(segments)
                                for segment_index, segment in enumerate(segments):
                                    if segment_index > 0:
                                        arrowstyle = "-"
                                        #arrowstyle = "->, head_width = 0.8, head_length = 0.5" if segment_index == segment_count - 1 else "-"
                                        if (x_range[0] < segment[0] < x_range[1]) and (y_range[0] < segment[1] < y_range[1]):
                                            x0 = segments[segment_index - 1][0]
                                            y0 = segments[segment_index - 1][1]
                                            x1 = segment[0]
                                            y1 = segment[1]
                                            slope = (y1 - y0) / (x1 - x0)
                                            intercept = y1 - slope * x1
                                            # If the x starting coordinate is outside
                                            while (not (x_range[0] <= x0 <= x_range[1])) or (not (y_range[0] <= y0 <= y_range[1])):
                                                if not (x_range[0] <= x0 <= x_range[1]):
                                                    x0 = max(min(x_range[1], x0), x_range[0])
                                                    y0 = slope * x0 + intercept
                                                # If the y starting coordinate is outside
                                                elif not (y_range[0] <= y0 <= y_range[1]):
                                                    y0 = max(min(y_range[1], y0), y_range[0])
                                                    x0 = (y0 - intercept) / slope
                                            zoom_dict[zoom_country]["ax"].annotate(
                                                "",
                                                xytext = (x0, y0),
                                                xy = (x1, y1),
                                                arrowprops = dict(
                                                    arrowstyle = arrowstyle,
                                                    linewidth = formatting_dict["pipe_label_fontsize"] + 10,
                                                    color = formatting_dict["piped_import_label_col"],
                                                    alpha = formatting_dict["piped_import_arrow_alpha"]
                                                )
                                            )
                # Once more for labels
                for piped_import_flow_index, piped_import_flow in output_df_dict["Piped_import_cons"].iterrows():
                    # Only plot required values
                    if piped_import_flow[cycle] > metric_params["piped_imports"]:
                        label_loc = import_pipes_labels_gdf.loc[piped_import_flow_index[0] + "_full"]["points"][0]
                        scenario_ax.text(
                            x = label_loc[0],
                            y = label_loc[1],
                            s = f"{piped_import_flow[cycle]: .{metric_params.get('piped_imports_digits', 0)}f}",
                            ha = "center",
                            va = "baseline",
                            size = formatting_dict["pipe_label_fontsize"],
                            bbox = dict(
                                boxstyle = "round, pad = 0.1",
                                fc = formatting_dict["piped_import_label_col"]
                            ),
                            alpha = formatting_dict["piped_import_label_alpha"]
                        )
                    
                    if zoom_creation:
                        for zoom_country in zoom_list:
                            if piped_import_flow_index[2] == zoom_country and piped_import_flow[cycle] > metric_params["piped_imports"]:
                                label_loc = import_pipes_labels_gdf.loc[piped_import_flow_index[0] + "_close"]["points"][0]
                                zoom_dict[zoom_country]["ax"].text(
                                    x = label_loc[0],
                                    y = label_loc[1],
                                    s = f"{piped_import_flow[cycle]: .{metric_params.get('piped_imports_digits', 0)}f}",
                                    ha = "center",
                                    va = "baseline",
                                    size = formatting_dict["pipe_label_fontsize"] * formatting_dict["zoom_font_factor"],
                                    bbox = dict(
                                        boxstyle = "round, pad = 0.1",
                                        fc = formatting_dict["piped_import_label_col"]
                                    ),
                                    alpha = formatting_dict["piped_import_label_alpha"]
                                )
            
            # Connection flows
            if isinstance(metric_params.get("transits", None), (int, float)):
                # Aggregate flows?
                connection_flow_df = output_df_dict["Connections_cons"].copy()
                if metric_params.get("aggregate_transit", None):
                    connection_flow_df["Flow"] = connection_flow_df["dir"] * connection_flow_df[cycle]
                    connection_flow_df = connection_flow_df.groupby(["Latitude", "Longitude", "new_angle"]).sum()
                else:
                    connection_flow_df["Flow"] = connection_flow_df[cycle]
                    connection_flow_df.set_index(["Latitude", "Longitude", "new_angle"], inplace = True)
                
                for connection_flow_index, connection_flow in connection_flow_df.iterrows():
                    # Only plot required values
                    if abs(connection_flow["Flow"]) > metric_params["transits"]:
                        # Arrow direction
                        if metric_params.get("aggregate_transit", None):
                            if connection_flow["Flow"] > 0:
                                arrow_dir = "larrow"
                            else:
                                arrow_dir = "rarrow"
                        else:
                            if connection_flow["dir"] > 0:
                                arrow_dir = "larrow"
                            else:
                                arrow_dir = "rarrow"
                        arrow_dir += ", pad = 0.1"
                        scenario_ax.text(
                            x = connection_flow_index[1],
                            y = connection_flow_index[0],
                            rotation = connection_flow_index[2],
                            s = f"{abs(connection_flow['Flow']): .{metric_params.get('transit_digits', 0)}f}",
                            ha = "center",
                            va = "center",
                            size = formatting_dict["connection_label_fontsize"],
                            bbox = dict(
                                boxstyle = arrow_dir,
                                fc = formatting_dict["connection_arrow_col"],
                                alpha = formatting_dict["connection_arrow_alpha"],
                                ec = "none"
                            )
                        )
                        if zoom_creation:
                            for zoom_country in zoom_list:
                                marker = params_dict["C Connections"][
                                        (
                                            (params_dict["C Connections"]["Latitude"] == connection_flow_index[0]) &
                                            (params_dict["C Connections"]["Longitude"] == connection_flow_index[1]) &
                                            (params_dict["C Connections"]["new_angle"] == connection_flow_index[2])
                                        )
                                ].iloc[0]
                                if marker["Origin"] == zoom_country or marker["Destination"] == zoom_country:
                                    zoom_dict[zoom_country]["ax"].text(
                                        x = connection_flow_index[1],
                                        y = connection_flow_index[0],
                                        rotation = connection_flow_index[2],
                                        s = f"{abs(connection_flow['Flow']): .{metric_params.get('transit_digits', 0)}f}",
                                        ha = "center",
                                        va = "center",
                                        size = formatting_dict["connection_label_fontsize"] * formatting_dict["zoom_font_factor"],
                                        bbox = dict(
                                            boxstyle = arrow_dir,
                                            fc = formatting_dict["connection_arrow_col"],
                                            alpha = formatting_dict["connection_arrow_alpha"],
                                            ec = "none"
                                        )
                                    )
                        
            # LNG flows
            if isinstance(metric_params.get("lng_imports", None), (int, float)):
                for lng_connection_flow_index, lng_connection_flow in output_df_dict["LNG_cons"].iterrows():
                    # Only plot required values
                    if lng_connection_flow[cycle] > metric_params["lng_imports"]:
                        scenario_ax.text(
                            x = lng_connection_flow["Longitude"],
                            y = lng_connection_flow["Latitude"],
                            rotation = lng_connection_flow["new_angle"],
                            s = f"{lng_connection_flow[cycle]: .{metric_params.get('lng_imports_digits', 0)}f}",
                            ha = "center",
                            va = "center",
                            size = formatting_dict["lng_label_fontsize"],
                            bbox = dict(
                                boxstyle = "larrow, pad = 0.1" if lng_connection_flow["new_angle"] == lng_connection_flow["Angle"] else "rarrow, pad = 0.1",
                                fc = formatting_dict["lng_connection_arrow_col"],
                                alpha = formatting_dict["lng_connection_arrow_alpha"],
                                ec = "none"
                            )
                        )
                    if zoom_creation:
                        for zoom_country in zoom_list:
                            if lng_connection_flow_index == zoom_country:
                                zoom_dict[zoom_country]["ax"].text(
                                    x = lng_connection_flow["Longitude"],
                                    y = lng_connection_flow["Latitude"],
                                    rotation = lng_connection_flow["new_angle"],
                                    s = f"{lng_connection_flow[cycle]: .{metric_params.get('lng_imports_digits', 0)}f}",
                                    ha = "center",
                                    va = "center",
                                    size = formatting_dict["lng_label_fontsize"] * formatting_dict["zoom_font_factor"],
                                    bbox = dict(
                                        boxstyle = "larrow, pad = 0.1" if lng_connection_flow["new_angle"] == lng_connection_flow["Angle"] else "rarrow, pad = 0.1",
                                        fc = formatting_dict["lng_connection_arrow_col"],
                                        alpha = formatting_dict["lng_connection_arrow_alpha"],
                                        ec = "none"
                                    )
                                )
                                
            # Piped export flows
            if isinstance(metric_params.get("piped_exports", None), (int, float)):
                for export_flow_index, export_flow in output_df_dict["Piped Exports"].iterrows():
                    # Only plot required values
                    if export_flow[cycle] < metric_params["piped_exports"]:
                        scenario_ax.text(
                            x = params_dict["C Piped Exporters Conn"].loc[export_flow_index[0]]["Origin Longitude"],
                            y = params_dict["C Piped Exporters Conn"].loc[export_flow_index[0]]["Origin Latitude"],
                            s = f"{abs(export_flow[cycle]): .{metric_params.get('piped_exports_digits', 0)}f}",
                            ha = "center",
                            va = "center",
                            size = formatting_dict["export_label_fontsize"],
                            bbox = dict(
                                boxstyle = "larrow, pad = 0.1",
                                fc = formatting_dict["piped_export_arrow_col"],
                                alpha = formatting_dict["piped_export_arrow_alpha"],
                                ec = "none"
                            )
                        )
                    if zoom_creation:
                        for zoom_country in zoom_list:
                            if export_flow_index[2] == zoom_country:
                                zoom_dict[zoom_country]["ax"].text(
                                    x = params_dict["C Piped Exporters Conn"].loc[export_flow_index[0]]["Origin Longitude"],
                                    y = params_dict["C Piped Exporters Conn"].loc[export_flow_index[0]]["Origin Latitude"],
                                    s = f"{abs(export_flow[cycle]): .{metric_params.get('piped_exports_digits', 0)}f}",
                                    ha = "center",
                                    va = "center",
                                    size = formatting_dict["export_label_fontsize"] * formatting_dict["zoom_font_factor"],
                                    bbox = dict(
                                        boxstyle = "larrow, pad = 0.1",
                                        fc = formatting_dict["piped_export_arrow_col"],
                                        alpha = formatting_dict["piped_export_arrow_alpha"],
                                        ec = "none"
                                    )
                                )
            
            scenario_fig.canvas.draw()

            # Massage data and save file
            data = np.fromstring(scenario_fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(scenario_fig.canvas.get_width_height()[::-1] + (3,))[30 * resolution_level:-30 * resolution_level, 30 * resolution_level:-30 * resolution_level, :]
            h, w = data.shape[:2]
            RGBA = np.dstack((data, np.zeros((h, w), dtype = np.uint8) + 255))
            mWhite = (RGBA[:, :, 0:3] == [255, 255, 255]).all(2)
            RGBA[mWhite] = (255, 255, 255, 0)
            img = Image.fromarray(RGBA)
            img.save(OUTPUT_FOLDER / "scenarios" / scenario / "Markt" / metric / f"Markt_{scenario}_{metric}_{cycle}.png", "PNG")
            
            if zoom_creation:
                for zoom_country in zoom_list:
                    fig_range = zoom_dict[zoom_country]["fig"].canvas.get_width_height()
                    aspect_ratio = fig_range[1] / fig_range[0]
                    zoom_dict[zoom_country]["fig"].canvas.draw()
                    data = np.fromstring(zoom_dict[zoom_country]["fig"].canvas.tostring_rgb(), dtype=np.uint8, sep="")
                    data = data.reshape(zoom_dict[zoom_country]["fig"].canvas.get_width_height()[::-1] + (3,))[int(30 * aspect_ratio * resolution_level):-int(30 * aspect_ratio * resolution_level), int(20 / aspect_ratio * resolution_level):-int(55 / aspect_ratio * resolution_level), :]
                    h, w = data.shape[:2]
                    RGBA = np.dstack((data, np.zeros((h, w), dtype = np.uint8) + 255))
                    mWhite = (RGBA[:, :, 0:3] == [255, 255, 255]).all(2)
                    RGBA[mWhite] = (255, 255, 255, 0)
                    img = Image.fromarray(RGBA)
                    img.save(OUTPUT_FOLDER / "scenarios" / scenario / zoom_country / metric / f"{zoom_country}_{scenario}_{metric}_{cycle}.png", "PNG")
                               
            plt.close()
