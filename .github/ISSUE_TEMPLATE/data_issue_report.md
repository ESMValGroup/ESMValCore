---
name: Dataset issue report
about: Create a report about a problematic dataset
title: 'Dataset problem: '
labels: ''
assignees: valeriupredoi, zklaus

---

**Describe the dataset issue**
A clear and concise description of what the problem with the data is is. Here are some guidelines that you can use
right out the box by filling in the blanks (note that, if needed we will assist you with writing a fix; also note that
if the problem is with a CMIP6 dataset, we will alert the ESGF and/or the model developers for them to fix the problem
in the long run; fixes for CMIP5 by the model developers are no longer possible):

- Data file has been changed by you in any way (if you answer yes the issue will be void and closed, we are not
supporting custom-made datasets that present problems, it is your resposability to fix those problems):
- Project (CMIP5/CMIP6/OBS/obs4mips/etc):
- Full dataset decription (fill out as many as you know, please):
  - dataset:
  - experiment:
  - mip:
  - grid:
  - type:
  - version:
  - tier:
  - variable used:
  - data version:
- Problems encountered (please describe what the actual problems are: e.g. wrong standard name, issue with dimensions):
- Pointer to existing copy of data on ESGF node (it would be very useful if you could provide a physical
fullpath to the file(s) that are causing the problem, e.g. on CEDA Jasmin or DKRZ):
- Other notes and mentions:

**Assign the right people**
If you are already a member of the ESMValTool GitHub project, please assign Valeriu Predoi (valeriupredoi) and
Klaus Zimmermann (zklaus) to the issue. They will then check the issue raised and propagate it further to the
data model developers.

**Please attach**
  - The recipe that you are trying to run, you can find a copy in the `run` directory in the output directory
  - The `main_log_debug.txt` file, this can also be found in the `run` directory in the output directory
  - Attach a text file with the output of `ncdump -h /path/to/file.nc`

Cheers :beer:
