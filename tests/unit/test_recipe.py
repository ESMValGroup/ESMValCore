import pytest

from esmvalcore._recipe import Recipe
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
