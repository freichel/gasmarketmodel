'''
Import module
All data imports from Excel are defined here
'''

import pandas as pd
import openpyxl

def import_excel_generic_df(databook_path, sheet_name, skiprows, index_col):
    '''
    Function can be used for most imports
    Returns dataframe
    '''
    return pd.read_excel(
        io = databook_path,
        sheet_name = sheet_name,
        skiprows = skiprows,
        index_col = index_col
    )
    
def import_excel_connections_data_df(databook_path, sheet_name, skiprows, index_col, merge_df = None):
    '''
    Import function for connections data
    Separate routine if merge required
    Returns dataframe
    '''
    df = pd.read_excel(
        io = databook_path,
        sheet_name = sheet_name,
        skiprows = skiprows,
        index_col = index_col,
        usecols = "B:C,E:U"
    ).fillna(0)
    temp_index = df.index.names
    df = df.reset_index().dropna().set_index(temp_index)
    if merge_df is not None:
        df = df.reset_index().merge(
            merge_df[["Name", "Importer"]],
            left_on = "Piped Import Connection",
            right_on = "Name",
            how = "left"
        )
        df.drop(columns = ["Name"], inplace = True)
        df = df.reset_index().dropna().set_index(temp_index)
    return df

def import_excel_cell(databook_path, sheet_name, row, col):
    '''
    Returns value of specific cell as str
    '''
    wb = openpyxl.load_workbook(databook_path)
    ws = wb[sheet_name]
    return ws.cell(row, col).value
    