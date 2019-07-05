# Quicklook system

## Prerequisits

Have the ESMValTool with quicklook enhancements available at the postprocessing machine

## Configuration

Adapt the file `template_recipe_emac.yml` for your needs.

## Execution

Call the quicklook system for full years only:

### Normal execution

Usually the quicklook system is called via commandline option 
```
quicklooks --rid <Run identifier> 
    --start <year of last restart> 
    --end <last year computed> 
    --project <DKRZ project to be chared>
    --inpath <Path to input data>
```

