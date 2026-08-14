---
name: Dataset issue report
about: Create a report about a problematic dataset
title: 'Dataset problem: '
labels: ''
assignees: valeriupredoi

---

**Describe the dataset issue**
A clear and concise description of what the problem with the data is is. Here are some guidelines that you can use
right out the box by filling in the blanks. If needed we will assist you with writing a fix.
Please check the [ESGF errata service](https://errata.ipsl.fr/static/index.html) to
see if the problem has already been reported and report it there if it is not.
Please link to the errata issue in your report here.

- Data file has been changed by you in any way (if you answer yes the issue will be void and closed, we are not
supporting custom-made datasets that present problems, it is your resposability to fix those problems):
- Project (CMIP7/CMIP6/obs4MIPs/CORDEX/OBS/etc.):
- Full dataset description (fill out as many as you know, please):
  - dataset:
  - ensemble:
  - experiment:
  - mip:
  - grid:
  - variable used:
  - version:
- Problems encountered (please describe what the actual problems are: e.g. wrong standard name, issue with dimensions):
- Pointer to existing copy of data on ESGF node (it would be very useful if you could provide a physical
fullpath to the file(s) that are causing the problem, e.g. on CEDA Jasmin or DKRZ):
- Other notes and mentions:

**Please attach**
  - The recipe that you are trying to run, you can find a copy in the `run` directory in the output directory
  - The `main_log_debug.txt` file, this can also be found in the `run` directory in the output directory
  - Attach a text file with the output of `ncdump -h /path/to/file.nc`

Cheers :beer:
