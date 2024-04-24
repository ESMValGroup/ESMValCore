.. _preprocessor_functions:

Preprocessor functions
======================

All preprocessor functions are designed to preserve floating point data types.
For example, input data of type ``float32`` will give output of ``float32``.
However, other data types may change, e.g., data of type ``int`` may give
output of type ``float64``.

.. autodata:: esmvalcore.preprocessor.DEFAULT_ORDER
.. automodule:: esmvalcore.preprocessor
