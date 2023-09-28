"""Unit tests for `esmvalcore.preprocessor._shared`."""
import warnings

import iris.analysis
import pytest

from esmvalcore.exceptions import ESMValCoreDeprecationWarning
from esmvalcore.preprocessor._shared import (
    aggregator_accept_weights,
    get_iris_aggregator,
)


@pytest.mark.parametrize('operator', ['gmean', 'GmEaN', 'GMEAN'])
def test_get_iris_aggregator_gmean(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.GMEAN
    assert agg_kwargs == {}


@pytest.mark.parametrize('operator', ['hmean', 'hMeAn', 'HMEAN'])
def test_get_iris_aggregator_hmean(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.HMEAN
    assert agg_kwargs == {}


@pytest.mark.parametrize('operator', ['max', 'mAx', 'MAX'])
def test_get_iris_aggregator_max(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MAX
    assert agg_kwargs == {}


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}, {'weights': False}])
@pytest.mark.parametrize('operator', ['mean', 'mEaN', 'MEAN'])
def test_get_iris_aggregator_mean(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.MEAN
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}])
@pytest.mark.parametrize('operator', ['median', 'mEdIaN', 'MEDIAN'])
def test_get_iris_aggregator_median(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.MEDIAN
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('operator', ['min', 'MiN', 'MIN'])
def test_get_iris_aggregator_min(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.MIN
    assert agg_kwargs == {}


@pytest.mark.parametrize('operator', ['peak', 'pEaK', 'PEAK'])
def test_get_iris_aggregator_peak(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.PEAK
    assert agg_kwargs == {}


@pytest.mark.parametrize('kwargs', [{'percent': 80.0, 'alphap': 0.5}])
@pytest.mark.parametrize('operator', ['percentile', 'PERCENTILE'])
def test_get_iris_aggregator_percentile(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.PERCENTILE
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'alphap': 0.5}])
@pytest.mark.parametrize('operator', ['p10', 'P10.5'])
def test_get_iris_aggregator_pxxyy(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    with pytest.warns(ESMValCoreDeprecationWarning):
        (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.PERCENTILE
    assert agg_kwargs == {'percent': float(operator[1:]), **kwargs}


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}])
@pytest.mark.parametrize('operator', ['rms', 'rMs', 'RMS'])
def test_get_iris_aggregator_rms(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.RMS
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'ddof': 1}])
@pytest.mark.parametrize('operator', ['std', 'STD', 'std_dev', 'STD_DEV'])
def test_get_iris_aggregator_std(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    if operator.lower() == 'std':
        with pytest.warns(ESMValCoreDeprecationWarning):
            (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter(
                'error', category=ESMValCoreDeprecationWarning
            )
            (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.STD_DEV
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('kwargs', [{}, {'weights': True}])
@pytest.mark.parametrize('operator', ['sum', 'SuM', 'SUM'])
def test_get_iris_aggregator_sum(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.SUM
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('operator', ['variance', 'vArIaNcE', 'VARIANCE'])
def test_get_iris_aggregator_variance(operator):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator)
    assert agg == iris.analysis.VARIANCE
    assert agg_kwargs == {}


@pytest.mark.parametrize('kwargs', [{'percent': 10, 'weights': True}])
@pytest.mark.parametrize('operator', ['wpercentile', 'WPERCENTILE'])
def test_get_iris_aggregator_wpercentile(operator, kwargs):
    """Test ``get_iris_aggregator``."""
    (agg, agg_kwargs) = get_iris_aggregator(operator, **kwargs)
    assert agg == iris.analysis.WPERCENTILE
    assert agg_kwargs == kwargs


@pytest.mark.parametrize('operator', ['invalid', 'iNvAliD', 'INVALID'])
def test_get_iris_aggregator_invalid_operator_fail(operator):
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator(operator)


@pytest.mark.parametrize('operator', ['mean', 'mEaN', 'MEAN'])
def test_get_iris_aggregator_no_aggregator_fail(operator, monkeypatch):
    """Test ``get_iris_aggregator``."""
    monkeypatch.setattr(iris.analysis, 'MEAN', 1)
    with pytest.raises(ValueError):
        get_iris_aggregator(operator)


def test_get_iris_aggregator_invalid_kwarg():
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator('max', invalid_kwarg=1)


def test_get_iris_aggregator_missing_kwarg():
    """Test ``get_iris_aggregator``."""
    with pytest.raises(ValueError):
        get_iris_aggregator('percentile')


@pytest.mark.parametrize(
    'aggregator,result',
    [
        (iris.analysis.MEAN, True),
        (iris.analysis.SUM, True),
        (iris.analysis.RMS, True),
        (iris.analysis.WPERCENTILE, True),
        (iris.analysis.MAX, False),
        (iris.analysis.MIN, False),
        (iris.analysis.PERCENTILE, False),
    ],
)
def test_aggregator_accept_weights(aggregator, result):
    """Test ``aggregator_accept_weights``."""
    assert aggregator_accept_weights(aggregator) == result
