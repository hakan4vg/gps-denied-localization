from types import SimpleNamespace

from gpsloc import dem


def _frame(h):
    return SimpleNamespace(sensors={} if h is None else {"height": h}, truth=SimpleNamespace(lat=32.5, lon=109.5))


def test_usable_agl_rejects_csv_glitch(monkeypatch):
    monkeypatch.setattr(dem, "elevation", lambda lat, lon, d=None: 505.4)
    assert dem.usable_agl(_frame(840.0)) == 840.0 - 505.4
    assert dem.usable_agl(_frame(499.4)) is None  # 6 m underground
    assert dem.usable_agl(_frame(None)) is None


def test_usable_agl_falls_back_without_tile(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(dem, "elevation", missing)
    assert dem.usable_agl(_frame(450.0)) == 450.0
    assert dem.usable_agl(_frame(0.0)) is None
