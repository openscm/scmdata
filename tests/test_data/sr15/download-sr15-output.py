import os.path

import pyam

import scmdata
import scmdata.database

out_path = os.path.join(".")

conn = pyam.iiasa.Connection()
print("Connecting to SR1.5 database")
conn.connect("iamc15")

print("Querying data")

variables_to_download = (
    "*Temperature|Global Mean*",
    "*Temperature|Exceed*",
)
out = []
for v in variables_to_download:
    print(f"Downloading {v}")
    df = pyam.read_iiasa("iamc15", variable=v, region="World", meta=["category"])

    meta = df.meta.drop("exclude", axis="columns")
    df_with_meta = meta.join(df.timeseries().reset_index().set_index(meta.index.names))
    df_scmrun = scmdata.ScmRun(df_with_meta)
    out.append(df_scmrun)

out = scmdata.run_append(out)
out.to_csv("sr15-output.csv")
