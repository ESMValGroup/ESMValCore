import pytest

from esmvalcore._recipe import Recipe, _allow_skipping
from esmvalcore._recipe_checks import RecipeError


class TestRecipe:
    def test_expand_ensemble(self):

        datasets = [
            {
                'dataset': 'XYZ',
                'ensemble': 'r(1:2)i(2:3)p(3:4)',
            },
        ]

        expanded = Recipe._expand_ensemble(datasets)

        ensembles = [
            'r1i2p3',
            'r1i2p4',
            'r1i3p3',
            'r1i3p4',
            'r2i2p3',
            'r2i2p4',
            'r2i3p3',
            'r2i3p4',
        ]
        for i, ensemble in enumerate(ensembles):
            assert expanded[i] == {'dataset': 'XYZ', 'ensemble': ensemble}

    def test_expand_ensemble_nolist(self):

        datasets = [
            {
                'dataset': 'XYZ',
                'ensemble': ['r1i1p1', 'r(1:2)i1p1']
            },
        ]

        with pytest.raises(RecipeError):
            Recipe._expand_ensemble(datasets)


VAR_A = {'dataset': 'A'}
VAR_A_REF_A = {'dataset': 'A', 'reference_dataset': 'A'}
VAR_A_REF_B = {'dataset': 'A', 'reference_dataset': 'B'}


TEST_ALLOW_SKIPPING = [
    ([], VAR_A, {}, False),
    ([], VAR_A, {'skip-nonexistent': False}, False),
    ([], VAR_A, {'skip-nonexistent': True}, True),
    ([], VAR_A_REF_A, {}, False),
    ([], VAR_A_REF_A, {'skip-nonexistent': False}, False),
    ([], VAR_A_REF_A, {'skip-nonexistent': True}, False),
    ([], VAR_A_REF_B, {}, False),
    ([], VAR_A_REF_B, {'skip-nonexistent': False}, False),
    ([], VAR_A_REF_B, {'skip-nonexistent': True}, True),
    (['A'], VAR_A, {}, False),
    (['A'], VAR_A, {'skip-nonexistent': False}, False),
    (['A'], VAR_A, {'skip-nonexistent': True}, False),
    (['A'], VAR_A_REF_A, {}, False),
    (['A'], VAR_A_REF_A, {'skip-nonexistent': False}, False),
    (['A'], VAR_A_REF_A, {'skip-nonexistent': True}, False),
    (['A'], VAR_A_REF_B, {}, False),
    (['A'], VAR_A_REF_B, {'skip-nonexistent': False}, False),
    (['A'], VAR_A_REF_B, {'skip-nonexistent': True}, False),
]


@pytest.mark.parametrize('ancestors,var,cfg,out', TEST_ALLOW_SKIPPING)
def test_allow_skipping(ancestors, var, cfg, out):
    """Test ``_allow_skipping``."""
    result = _allow_skipping(ancestors, var, cfg)
    assert result is out
