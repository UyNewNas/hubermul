"""Independent arbitrary-precision evaluation; never imports production helpers."""
import mpmath as mp


def oracle(x: float, y: float, radius: float) -> float:
    # 800 digits also covers cancellation involving very large binary64 inputs.
    with mp.workdps(800):
        x, y, r = mp.mpf(float(x)), mp.mpf(float(y)), mp.mpf(float(radius))

        def h(t):
            return t * t / 2 if abs(t) <= r else r * (abs(t) - r / 2)

        return float((h(x + y) - h(x - y)) / 2)
