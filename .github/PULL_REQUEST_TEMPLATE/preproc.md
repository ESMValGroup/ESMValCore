---
name: Core contribution
about: 
title: ''
labels: ''
assignees: ''

---

Before you start, read [contributing.md](https://github.com/ESMValGroup/ESMValTool/blob/version2_development/CONTRIBUTING.md).
If you are making a dataset fix, first read the [documentation](https://esmvaltool.readthedocs.io/projects/esmvalcore/en/latest/esmvalcore/fixing_data.html).

**Tasks**
-  [ ]  Create an issue to explain what you are going to do (and add the link at the bottom)
-  [ ]  Add unit tests
-  [ ]  Public functions should have a numpy-style docstring so they appear properly in the [API documentation](https://esmvaltool.readthedocs.io/projects/esmvalcore/en/latest/api/esmvalcore.html). For all other functions a one line docstring is sufficient.
-  [ ]  If writing a new preprocessor function, please update the [documentation](https://esmvaltool.readthedocs.io/projects/esmvalcore/en/latest/esmvalcore/preprocessor.html)
-  [ ]  Make sure the Circle/CI tests pass
-  [ ]  Make sure the Codacy tests pass
-  [ ]  Please use `yamllint` to check that your YAML files do not contain mistakes
-  [ ]  If you make backward incompatible changes to the recipe format, make a new pull request in the [ESMValTool repository](https://github.com/ESMValGroup/ESMValTool) and add the link below

If you need help with any of the tasks above, please do not hesitate to ask by commenting in the issue or pull request.

---

**Links to info and code**
Fixes {Link to corresponding issue}
(optional) Corresponding Pull request in ESMValTool: {Link to corresponding pull request}
