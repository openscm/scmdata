import numpy as np
from scmdata import ScmRun

IDX = [2010, 2020, 2030]


start = ScmRun(
    data=np.arange(18).reshape(3, 6),
    index=IDX,
    columns={
        "variable": [
            "Emissions|CO2|Fossil",
            "Emissions|CO2|AFOLU",
            "Emissions|CO2|Fossil",
            "Emissions|CO2|AFOLU",
            "Cumulative Emissions|CO2",
            "Surface Air Temperature Change",
        ],
        "unit": ["GtC / yr", "GtC / yr", "GtC / yr", "GtC / yr", "GtC", "K"],
        "region": ["World|NH", "World|NH", "World|SH", "World|SH", "World", "World"],
        "model": "idealised",
        "scenario": "idealised",
    },
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

fos_minus_afolu = fos.subtract(
    afolu, op_cols={"variable": "Emissions|CO2|Fossil - AFOLU"}
)
fos_minus_afolu.head()

nh_minus_sh = nh.subtract(sh, op_cols={"region": "World|NH - SH"})
nh_minus_sh.head()

fos_times_afolu = fos.multiply(
    afolu, op_cols={"variable": "Emissions|CO2|Fossil : AFOLU"}
)
fos_times_afolu.head()

warming_per_co2 = start.filter(variable="*Temperature*").divide(
    start.filter(variable="Cumulative*"), op_cols={"variable": "Warming per emissions"}
)
warming_per_co2.head()
