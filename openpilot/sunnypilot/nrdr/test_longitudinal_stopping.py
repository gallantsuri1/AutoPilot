import math

import pytest

from openpilot.sunnypilot.nrdr.longitudinal_stopping import CONTROL_DT, compute_stopping_accel


DEFAULTS = {
  "stop_accel": -2.0,
  "stopping_decel_rate": 0.3,
  "v_ego_stopping": 0.3,
  "hold_accel": -0.6,
  "phase_switch_v": 0.15,
  "proximity_scale_m": 8.0,
  "pitch_margin": 1.0,
}


def stopping_accel(last, speed, distance=math.inf, pitch=0.0, **overrides):
  values = DEFAULTS | overrides
  return compute_stopping_accel(
    last,
    values["stop_accel"],
    values["stopping_decel_rate"],
    speed,
    values["v_ego_stopping"],
    values["hold_accel"],
    values["phase_switch_v"],
    values["proximity_scale_m"],
    values["pitch_margin"],
    distance,
    pitch,
  )


def settle(speed, start=-0.6, **kwargs):
  output = start
  for _ in range(8000):
    output = stopping_accel(output, speed, **kwargs)
  return output


def test_above_stopping_window_matches_stock_ramp():
  for last in (0.5, 0.0, -0.3, -1.5, -2.0):
    expected = last if last <= -2.0 else min(last, 0.0) - 0.3 * CONTROL_DT
    assert stopping_accel(last, 5.0) == pytest.approx(expected)


def test_rolling_and_standstill_have_distinct_targets():
  assert settle(0.25, start=-0.1) == pytest.approx(-0.6)
  assert settle(0.0) == pytest.approx(-2.0)


def test_standstill_hold_is_exactly_stop_accel():
  assert settle(0.0, stop_accel=-2.0) == pytest.approx(-2.0)
  assert settle(0.0, stop_accel=-1.0) == pytest.approx(-1.0)
  assert settle(0.0, stop_accel=-3.5) == pytest.approx(-3.5)
  # a target weaker than the current accel can't be reached: the ramp only decelerates,
  # so it freezes in place rather than releasing the brake
  assert settle(0.0, stop_accel=0.0) == pytest.approx(-0.6)


def test_pitch_strengthens_hold_beyond_stop_accel():
  uphill = settle(0.0, pitch=0.2)
  assert uphill < settle(0.0)  # facing uphill holds harder than the flat stop_accel target


def test_close_lead_reduces_ramp_rate():
  far = stopping_accel(-0.1, 0.0)
  near = stopping_accel(-0.1, 0.0, distance=2.0)
  assert abs(near + 0.1) < abs(far + 0.1)


@pytest.mark.parametrize("last,speed,distance,pitch", [
  (math.nan, 0.0, math.inf, 0.0),
  (0.0, math.nan, math.inf, 0.0),
  (-0.5, 0.0, math.nan, math.nan),
])
def test_nonfinite_inputs_produce_finite_output(last, speed, distance, pitch):
  assert math.isfinite(stopping_accel(last, speed, distance, pitch))
