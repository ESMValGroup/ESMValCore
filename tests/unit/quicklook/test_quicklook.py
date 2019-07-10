"""Unit tests for the :func:`esmvalcore.quicklook` module."""

import pytest
from pathlib import Path

import esmvalcore._quicklook as quicklook
import tempfile

def test_create_recipe(tmpdir):
    d = tmpdir.mkdir('test')
    cfg = {
        'quicklook_recipe_dir': None,
        'quicklook_recipes': ['ocean'],
        'quicklook_run_id': 'test123',
        'quicklook_output_dir': d.dirname,
    }
    start = 1900
    end = 1903

    quicklook.create_recipes(cfg, start, end)
    recipe_paths = [
        d.join(cfg['quicklook_run_id'], recipe + '.yml')
        for recipe in cfg['quicklook_recipes']
    ]
    for recipe_path in recipe_paths:
        print(recipe_path)
        assert False


def test_run():
    quicklook.run()
