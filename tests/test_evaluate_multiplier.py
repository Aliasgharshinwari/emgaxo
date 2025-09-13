#!/usr/bin/env python3
import pytest
from emgaxo.AppAxO.Evaluate_Multiplier import Compute_Metrics


@pytest.fixture(scope="module")
def default_lut_config():
    return 404750335

def test_error_metrics(default_lut_config):

    metrics = Compute_Metrics(default_lut_config)
    assert metrics["avg_error"] ==  -188.25
    assert metrics["avg_abs_error"] == 188.25
    assert metrics["avg_rel_error"] == -0.03953414422518257
    assert metrics["avg_abs_rel_error"] ==  0.35995859364678046
    assert metrics["max_error"] == 0
    assert metrics["min_error"] == -753
    assert metrics["error_probability"] == 0.92578125


if __name__ == "__main__":
    pytest.main([__file__, "-v"])