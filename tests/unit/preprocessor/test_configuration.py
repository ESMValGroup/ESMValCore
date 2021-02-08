"""Tests for the basic configuration of the preprocessor module."""
from esmvalcore.preprocessor import (
    DEFAULT_ORDER,
    FINAL_STEPS,
    INITIAL_STEPS,
    MULTI_MODEL_FUNCTIONS,
    TIME_PREPROCESSORS,
)


def test_non_repeated_keys():
    """Check that there are not repeated keys in the lists."""
    assert len(DEFAULT_ORDER) == len(set(DEFAULT_ORDER))
    assert len(TIME_PREPROCESSORS) == len(set(TIME_PREPROCESSORS))
    assert len(INITIAL_STEPS) == len(set(INITIAL_STEPS))
    assert len(FINAL_STEPS) == len(set(FINAL_STEPS))
    assert len(MULTI_MODEL_FUNCTIONS) == len(set(MULTI_MODEL_FUNCTIONS))


def test_time_preprocessores_default_order_added():
    assert all(
        (time_preproc in DEFAULT_ORDER for time_preproc in TIME_PREPROCESSORS))


def test_multimodel_functions_in_default_order():
    assert all((time_preproc in DEFAULT_ORDER
                for time_preproc in MULTI_MODEL_FUNCTIONS))
