import numpy as np
import pytest


def test_ray_modes_keep_sea_nav_dimensions():
    torch = pytest.importorskip("torch")
    mujoco = pytest.importorskip("mujoco")
    from deploy.go2_onboard.navigation_observation import NavigationObservation
    from .adapters import MuJoCoStateAdapter

    model = mujoco.MjModel.from_xml_path("mujoco_sim/assets/go2/course_mixed.xml")
    data = mujoco.MjData(model)
    data.qpos[2] = 0.45
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = [.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5]
    mujoco.mj_forward(model, data)
    for mode in ("grid2ray", "physical_lidar"):
        adapter = MuJoCoStateAdapter(model, data, [17, 0], "flat", mode)
        state = adapter.read([17, 0])
        assert state.rays.shape == (41,)
        obs = NavigationObservation(torch.device("cpu"))
        value = obs.build(torch.from_numpy(state.gravity).reshape(1, 3),
                          torch.zeros(1, 3), torch.from_numpy(state.linear_velocity).reshape(1, 3),
                          torch.from_numpy(state.angular_velocity).reshape(1, 3),
                          torch.from_numpy(state.rays).reshape(1, 41),
                          torch.from_numpy(state.goal_xy).reshape(1, 2))
        assert tuple(value.shape) == (1, 550)
        assert np.isfinite(state.rays).all()
