from . import nli


def test_filter_nli():
    assert not nli.filter_nli("After the war")


def test_fix_hypothesis():
    hypothesis = nli.fix_hypothesis("we did not ever eat cake", set("we did ever eat cake".split()))
    assert hypothesis == "we did not eat cake"

    hypothesis = nli.fix_hypothesis("we did ever eat cake", set("we did not ever eat cake".split()))
    assert hypothesis == "we did eat cake"
