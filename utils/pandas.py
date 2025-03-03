def to_npdict(df):
    """
    Convert a pandas DataFrame to a dict of numpy arrays.
    """
    return {col: df[col].values for col in df.columns}