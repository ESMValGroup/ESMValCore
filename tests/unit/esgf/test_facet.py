"""Test `esmvalcore.esgf.facets`."""
import pyesgf.search

from esmvalcore.esgf import facets


def test_create_dataset_map(monkeypatch, mocker):
    """Test `esmvalcore.esgf.facets.create_dataset_map`."""
    monkeypatch.setattr(facets, 'FACETS', {'CMIP5': facets.FACETS['CMIP5']})

    conn = mocker.create_autospec(
        pyesgf.search.SearchConnection,
        spec_set=True,
        instance=True,
    )
    ctx = mocker.create_autospec(
        pyesgf.search.context.DatasetSearchContext,
        spec_set=True,
        instance=True,
    )
    ctx.facet_counts = {
        'model': {
            'ACCESS1.0': 10,
            'BNU-ESM': 20
        },
    }
    ids = [
        'cmip5.output1.CSIRO-BOM.ACCESS1-0.1pctCO2.3hr.atmos.3hr.r1i1p1.v1'
        '|aims3.llnl.gov',
        'cmip5.output1.BNU.BNU-ESM.rcp45.mon.atmos.Amon.r1i1p1.v20120510'
        '|aims3.llnl.gov',
    ]
    results = [
        pyesgf.search.results.DatasetResult(
            json={
                'id': id_,
                'score': 1.0
            },
            context=None,
        ) for id_ in ids
    ]
    ctx.search.side_effect = [[r] for r in results]
    conn.new_context.return_value = ctx
    mocker.patch.object(facets.pyesgf.search,
                        'SearchConnection',
                        autospec=True,
                        return_value=conn)

    dataset_map = facets.create_dataset_map()
    assert dataset_map == {'CMIP5': {'ACCESS1-0': 'ACCESS1.0'}}
