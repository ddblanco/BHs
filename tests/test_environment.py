from rotating_bh.environment import environment_report


def test_environment_report_has_required_versions():
    report = environment_report()

    assert report["python"].count(".") >= 1
    assert set(report["packages"]) == {"numpy", "scipy", "sympy", "matplotlib"}
    assert all(report["packages"].values())
