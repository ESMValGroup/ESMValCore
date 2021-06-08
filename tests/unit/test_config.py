import pytest

from esmvalcore._config._config import _deep_update

TEST_DEEP_UPDATE = [([{}], {}), ([dict(a=1, b=2), dict(a=3)], dict(a=3, b=2)),
                    ([
                        dict(a=dict(b=1, c=dict(d=2)), e=dict(f=4, g=5)),
                        dict(a=dict(b=2, c=3))
                    ], dict(a=dict(b=2, c=3), e=dict(f=4, g=5)))]


@pytest.mark.parametrize('dictionaries, expected_merged', TEST_DEEP_UPDATE)
def test_deep_update(dictionaries, expected_merged):
    merged = dictionaries[0]
    for update in dictionaries[1:]:
        merged = _deep_update(merged, update)
    assert expected_merged == merged
