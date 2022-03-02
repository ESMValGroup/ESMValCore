"""Unit tests for ``_data_finder.py``."""
import pytest

from esmvalcore._data_finder import _truncate_dates, dates_to_timerange

TEST_DATES_TO_TIMERANGE = [
    (2000, 2000, '2000/2000'),
    (1, 2000, '0001/2000'),
    (2000, 1, '2000/0001'),
    (1, 2, '0001/0002'),
    ('2000', '2000', '2000/2000'),
    ('1', '2000', '0001/2000'),
    (2000, '1', '2000/0001'),
    ('1', 2, '0001/0002'),
    ('*', '*', '*/*'),
    (2000, '*', '2000/*'),
    ('2000', '*', '2000/*'),
    (1, '*', '0001/*'),
    ('1', '*', '0001/*'),
    ('*', 2000, '*/2000'),
    ('*', '2000', '*/2000'),
    ('*', 1, '*/0001'),
    ('*', '1', '*/0001'),
    ('P5Y', 'P5Y', 'P5Y/P5Y'),
    (2000, 'P5Y', '2000/P5Y'),
    ('2000', 'P5Y', '2000/P5Y'),
    (1, 'P5Y', '0001/P5Y'),
    ('1', 'P5Y', '0001/P5Y'),
    ('P5Y', 2000, 'P5Y/2000'),
    ('P5Y', '2000', 'P5Y/2000'),
    ('P5Y', 1, 'P5Y/0001'),
    ('P5Y', '1', 'P5Y/0001'),
    ('*', 'P5Y', '*/P5Y'),
    ('P5Y', '*', 'P5Y/*'),
]


@pytest.mark.parametrize('start_date,end_date,expected_timerange',
                         TEST_DATES_TO_TIMERANGE)
def test_dates_to_timerange(start_date, end_date, expected_timerange):
    """Test ``dates_to_timerange``."""
    timerange = dates_to_timerange(start_date, end_date)
    assert timerange == expected_timerange


TEST_TRUNCATE_DATES = [
    ('2000', '2000', (2000, 2000)),
    ('200001', '2000', (2000, 2000)),
    ('2000', '200001', (2000, 2000)),
    ('200001', '2000', (2000, 2000)),
    ('200001', '200001', (200001, 200001)),
    ('20000102', '200001', (200001, 200001)),
    ('200001', '20000102', (200001, 200001)),
    ('20000102', '20000102', (20000102, 20000102)),
    ('20000102T23:59:59', '20000102', (20000102, 20000102)),
    ('20000102', '20000102T23:59:59', (20000102, 20000102)),
    ('20000102T235959', '20000102T01:02:03', (20000102235959, 20000102010203)),
]


@pytest.mark.parametrize('date,date_file,expected_output', TEST_TRUNCATE_DATES)
def test_truncate_dates(date, date_file, expected_output):
    """Test ``_truncate_dates``."""
    output = _truncate_dates(date, date_file)
    assert output == expected_output
