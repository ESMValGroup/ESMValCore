"""
Checks output directory of EMAC simulations for new data
and creates recipe for this data.


Files:
 - temp.yml
```
{earliestDate: '1850-01-31T23:50:00', restartDate: '1920-04-30T23:50:00'}
```
"""

import datetime
import glob
import os
import re
import subprocess
import warnings

import yaml

import logging
from  jinja2 import Template


recipe = """
# ESMValTool
# recipe_python.yml
---
documentation:
  description: |
    Example recipe that plots the mean precipitation and temperature.

  authors:
    - ande_bo
    - righ_ma

  maintainer:
    - broe_bj

  references:
    - acknow_project

  projects:
    - esmval

datasets:
    - {{ DATASET }}

preprocessors:

  preprocessor1:
    extract_levels:
      levels: 85000
      scheme: nearest
    regrid:
      target_grid: reference_dataset
      scheme: linear
    multi_model_statistics:
      span: overlap
      statistics: [mean, median]

diagnostics:

  diagnostic1:
    description: Air temperature and precipitation Python tutorial diagnostic.
    themes:
      - phys
    realms:
      - atmos
    variables:
      ta:
        preprocessor: preprocessor1
        reference_dataset: CanESM2
      pr:
        reference_dataset: MPI-ESM-LR
    scripts:
      script1:
        script: examples/diagnostic.py
        quickplot:
          plot_type: pcolormesh
"""

MIPS = {
        'DECK':{
            'rids' : ["1pctCO2", "abrupt-4xCO2", "historical", "piControl"],
            'base_data_dir' : "/work/bk0988/emac/b309081/DECK"
            },
        'AerChemMIP':{
            'rids' : ['hist-piNTCF-01', 'hist-piNTCF-02', 'hist-piNTCF-03', 'histSST', 'histSST-piNTCF', 'histSST-piCH4'],
#            'rids' : ['hist-piNTCF-01', 'hist-piNTCF-02', 'hist-piNTCF-03', 'histSST'],
            'base_data_dir' : '/work/bd1055/emac/b309081/AerChemMIP'
            }
        }
WORKPATH = "/work/bd0854/b309070/quicklooks/quicklook_workdir"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(logging.DEBUG)
CONSOLE_HANDLER.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(CONSOLE_HANDLER)

# logger.debug("Skipping bad file: %s", f)
# logger.error("Failed to create symlink for: %s", f)


def get_temp_file(rid):
    path = os.path.join(WORKPATH, rid)
    if not os.path.isdir(path):
        raise Exception("Path does not exist: {}".format(path))
    temp_file = os.path.join(path, "temp.yaml")
    if os.path.exists(temp_file):
        return temp_file
    else:
        raise Exception("Tempfile does not exist: {}".format(temp_file))


def get_last_year_processed(rid):
    out = None
    with open(get_temp_file(rid), 'r') as stream:
        try:
            info = yaml.load(stream, Loader=yaml.Loader)
        except:
            raise Exception("Error while loading the yaml file!")
    try:
        out = datetime.datetime.fromisoformat(
            info['restartDate']).strftime("%Y")
    except:
        raise Exception("Bad format of restart date!")
    return int(out)


def get_last_year_available(rid, base_data_dir):
    inpath = os.path.join(base_data_dir, rid, "Amon")
    if not os.path.isdir(inpath):
        raise Exception("No valid path : {0}".format(inpath))

    def _get_years(filename):
        match = re.search(r'_[0-9][0-9][0-9][0-9][0-9][0-9]_', filename)
        if match:
            return match.group(0).replace("_", "")
        else:
            logger.warning("Filename does not contain year pattern : %s", filename)
            return ""

    years = list()
    for item in glob.glob(inpath + "/*nc"):
        res = _get_years(os.path.basename(item))
        if res:
            years.append(res)
    years.sort()
    last = years[-1]
    return int(datetime.datetime.strptime(last, "%Y%m").strftime("%Y"))

def create_recipe( rid, last_processed, last_available, base_data_dir):
    template = Template(recipe)
    content = template.render(DATASET="{0}".format(rid))
    print(content)
    #with open(recipe_name, 'w') as f:
    #    f.write(content)

def run():
    for mip, content in MIPS.items():
        rids = content['rids']
        base_data_dir = content['base_data_dir']
        for rid in rids:
            last_processed = get_last_year_processed(rid)
            last_available = get_last_year_available(rid, base_data_dir)
            logger.debug("Last processed for %s of mip %s: %s", rid, mip, last_processed)
            logger.debug("Last available for %s of mip %s: %s", rid, mip, last_available)
            if last_available - last_processed > 2:
                logger.debug("Create recipe")
                create_recipe( rid,
                    last_processed,
                    last_available,
                    base_data_dir)
                #logger.debug("Start processing")
                # start processing


if __name__ == "__main__":
    run()


