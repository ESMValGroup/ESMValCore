.. _utils:

*********
Utilities
*********

This section provides extra information on topics that are not part of ESMValTool
code base but are used by ESMValTool directly or indirectly.

Brief introduction to YAML
==========================

While ``.yaml`` or ``.yml`` is a relatively common format, maybe users may not have
encountered this language before. The key information about this format is:

- yaml is a human friendly markup language.
- yaml is commonly used for configuration files (gradually replacing the venerable ``.ini``)
- the syntax is relatively straightforward
- indentation matters a lot (like ``Python``)!
- yaml is case sensitive
- a yaml tutorial is available `here <https://learnxinyminutes.com/docs/yaml/>`_
- a yaml quick reference card is available `here <https://yaml.org/refcard.html>`_
- ESMValTool uses the ``yamllint`` linter `tool <http://www.yamllint.com>`_
