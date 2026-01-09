import pandas as pd

# Pull out variables in a crop array to a dataframe
def variables_to_df(ca, var_names):
    """
    Creates a pandas dataframe from the specified variables of a crop array.  

    Parameters:
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    var_names: list of str
        Names of the variables in the xarray dataset to convert to a dataframe.
    
    Returns:
    -------
    A concatenated pandas dataframe with the specified variables such that each column of the dataframe corresponds to one coordinate dimension in the crop array. Basically the output corresponds to xr.to_dataframe(), but with multiindex flattened.
    """
        # Check if variables exist in the dataset
    for var in var_names:
        if var not in ca:
            raise ValueError(f"'{var}' not found in the provided xarray dataset.")

    # Check if the variables have the same dimensions
    dims = ca[var_names[0]].dims
    for var in var_names[1:]:
        if ca[var].dims != dims:
            raise ValueError(f"Variables do not have matching dimensions. {var_names[0]} has dimensions {dims} while {var} has dimensions {ca[var].dims}")

    # Convert each variable to a dataframe and concatenate them
    dfs = [ca[var].to_dataframe().reset_index(level=list(range(len(dims)))) for var in var_names]
    
    final_df = pd.concat(dfs, axis=1)
    # Drop duplicate columns if any arise due to the reset index operation
    final_df = final_df.loc[:,~final_df.columns.duplicated()]
    
    return final_df