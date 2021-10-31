from gasmarketmodel.params import europe_frame
import matplotlib.pyplot as plt

def plot_gdf(edgecolor, facecolor, linewidth, ax, gdf = europe_frame):
    '''
    Plots a GeoDataFrame based on the passed parameters on a given axes
    '''
    gdf.plot(
        edgecolor = edgecolor,
        facecolor = facecolor,
        linewidth = linewidth,
        ax = ax
    )
    