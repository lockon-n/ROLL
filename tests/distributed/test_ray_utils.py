from roll.distributed.ray_utils import build_ray_init_kwargs


def test_build_ray_init_kwargs_uses_loopback_for_single_worker(monkeypatch):
    monkeypatch.setenv("ARNOLD_WORKER_NUM", "1")

    kwargs = build_ray_init_kwargs(address=None, namespace="test")

    assert kwargs["address"] is None
    assert kwargs["_node_ip_address"] == "127.0.0.1"
    assert kwargs["namespace"] == "test"


def test_build_ray_init_kwargs_skips_loopback_for_multi_worker(monkeypatch):
    monkeypatch.setenv("ARNOLD_WORKER_NUM", "2")

    kwargs = build_ray_init_kwargs(address=None)

    assert kwargs["address"] is None
    assert "_node_ip_address" not in kwargs


def test_build_ray_init_kwargs_skips_loopback_for_explicit_address(monkeypatch):
    monkeypatch.setenv("ARNOLD_WORKER_NUM", "1")

    kwargs = build_ray_init_kwargs(address="127.0.0.1:6379")

    assert kwargs["address"] == "127.0.0.1:6379"
    assert "_node_ip_address" not in kwargs
