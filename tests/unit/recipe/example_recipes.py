"""Example recipes for the tests of recipe validation."""

RECIPE = """
documentation:
  title: Example recipe
  description: Example recipe.
  authors: [andela_bouwe]

datasets:
  - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: r1i1p1}
  - {dataset: ERA5, project: native6, tier: 3, type: reanaly, version: 1}
  - dataset: ACCESS-CM2
    project: CMIP6
    exp: historical
    ensemble: r1i1p1f1
    grid: gn
    activity: [CMIP, DCPP]

preprocessors:
  default: {}
  pp:
    custom_order: true
    derive:
    extract_levels:
      levels: 85000
      scheme: nearest
    regrid:
      target_grid: 1x1
      scheme: linear
    area_statistics:
      operator: mean
    multi_model_statistics:
      span: overlap
      statistics: [mean]
      exclude: [reference_dataset]
    convert_units:
      units: degC
    mask_landsea: false
    remove_supplementary_variables:

diagnostics:
  example:
    description:
    variables:
      ta:
        mip: Amon
        preprocessor: pp
        timerange: 2000/2002
        reference_dataset: ERA5
        additional_datasets:
          - dataset: CanESM5
            project: CMIP6
            exp: historical
            ensemble: r1i1p1f1
            grid: gn
      pr: {mip: Amon}
    scripts:
      plot:
        script: examples/diagnostic.py
        ancestors: [example/ta]
        quickplot:
          plot_type: pcolormesh
"""


INVALID_RECIPE = """
documentation: {title: Test, description: Test., authors: [andela_bouwe]}
datasets:
  - {dataset: CanESM2, project: CMIP5, exp: historical, ensemble: r1i1p1}
  - {project: CMIP5, tier: 7, exp: historical, ensemble: r1i1p1}
preprocessors:
  base: &base
    area_statistics:
      operator: 1
  derived:
    <<: *base
    convert_units:
diagnostics:
  example:
    variables: {ta: {mip: Amon, preprocessor: derived}}
"""


DUPLICATE_RECIPE = """
documentation: {title: Test, description: Test., authors: [andela_bouwe]}
preprocessors:
  base: &base
    area_statistics:
      operator: mean
    area_statistics:
      operator: max
  pp:
    <<: *base
    convert_units:
      units: K
diagnostics:
  example:
    variables:
      tas:
        mip: Amon
      tas:
        mip: day
    scripts: null
datasets: [{dataset: A, project: CMIP6, exp: x, ensemble: r1i1p1f1, grid: gn}]
"""


AUTHORS = [f"author_{i}" for i in range(12)]
