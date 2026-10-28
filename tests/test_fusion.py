import pytest

from contributor_copilot.fusion import reciprocal_rank_fusion


def test_items_ranked_high_in_both_lists_win():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [item for item, _ in fused] == ["a", "b", "c", "d"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_item_in_one_list_can_beat_item_low_in_both():
    fused = dict(reciprocal_rank_fusion([["x", "y", "z", "w"], ["y2", "z2", "w2", "w"]], k=1))
    assert fused["x"] == pytest.approx(1 / 2)
    assert fused["w"] == pytest.approx(1 / 5 + 1 / 5)
    assert fused["x"] > fused["w"]


def test_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert [item for item, _ in fused] == ["a", "b"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
