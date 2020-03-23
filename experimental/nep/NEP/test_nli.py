from . import nli


def test_filter_nli():
    assert not nli.filter_nli("After the war")
