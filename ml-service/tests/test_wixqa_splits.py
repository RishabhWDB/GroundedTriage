from eval.wixqa_splits import DEMO_N, assign, question_id
from rag.kb import QAPair


def pairs():
    return [QAPair(f"expert q{i}", "a", ["d"], "expertwritten") for i in range(60)] + [
        QAPair(f"sim q{i}", "a", ["d"], "simulated") for i in range(50)
    ]


def test_splits_are_disjoint_sized_and_order_free():
    p = pairs()
    a, b = assign(p), assign(list(reversed(p)))
    assert a == b
    demo = [k for k, v in a.items() if v == "demo"]
    assert len(demo) == DEMO_N
    assert all(
        a[question_id("expertwritten", f"expert q{i}")] in ("demo", "p72_dev", "p72_test")
        for i in range(60)
    )
    assert all(
        a[question_id("simulated", f"sim q{i}")] != "demo" for i in range(50)
    )  # demo is expert-only
    sim_dev = sum(a[question_id("simulated", f"sim q{i}")] == "p72_dev" for i in range(50))
    assert sim_dev == 20  # 40% of 50
    exp_dev = sum(
        v == "p72_dev"
        for k, v in a.items()
        if k in {question_id("expertwritten", f"expert q{i}") for i in range(60)}
    )
    assert exp_dev == 16  # 40% of the 40 non-demo expert questions
