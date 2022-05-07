import pandas as pd
from pandas_profiling import ProfileReport

def makeReport(path, savetopath):
    df = pd.read_csv(path)
    profile = ProfileReport(df, title="Pandas Profiling Report", missing_diagrams={"Count": False})
    profile.to_file(savetopath)