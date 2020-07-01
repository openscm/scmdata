import numpy as np
from scmdata import ScmRun

IDX = [2010, 2020, 2030]


start = ScmRun(
    data=np.arange(12).reshape(3, 4),
    index=IDX,
    columns={
        "variable": [
            "Emissions|CO2|Fossil", "Emissions|CO2|AFOLU",
            "Emissions|CO2|Fossil", "Emissions|CO2|AFOLU",
        ],
        "unit": "GtC / yr",
        "region": [
            "World|NH", "World|NH",
            "World|SH", "World|SH",
        ],
        "model": "idealised",
        "scenario": "idealised",
    }
)

start.head()

fos = start.filter(variable="*Fossil")
fos.head()

afolu = start.filter(variable="*AFOLU")
afolu.head()

total = fos.add(afolu, op_cols={"variable": "Emissions|CO2"})
total.head()

nh = start.filter(region="*NH")
nh.head()

sh = start.filter(region="*SH")
sh.head()

world = nh.add(sh, op_cols={"region": "World"})
world.head()


fos_minus_afolu = fos.subtract(afolu, op_cols={"variable": "Emissions|CO2|Fossil - AFOLU"})
fos_minus_afolu.head()

nh_minus_sh = nh.subtract(sh, op_cols={"region": "World|NH - SH"})
nh_minus_sh.head()
