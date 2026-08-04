from pathlib import Path

import pytest


def test_all_supported_scenes_load_with_go2_contract():
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).parent / "assets" / "go2"
    scenes = [root / name for name in ("scene.xml", "stair.xml", "stairs_down.xml",
                                       "hfield.xml", "course_flat_obstacle.xml",
                                       "course_mixed.xml", "rough_random_obstacles.xml")]
    for scene in scenes:
        model = mujoco.MjModel.from_xml_path(str(scene))
        assert (model.nq, model.nu) == (19, 12)


def test_rough_random_scene_seed_is_reproducible():
    mujoco = pytest.importorskip("mujoco")
    from mujoco_sim.run import _configure_rough_random_scene, _reset

    scene = Path(__file__).parent / "assets" / "go2" / "rough_random_obstacles.xml"
    model_a = mujoco.MjModel.from_xml_path(str(scene))
    model_b = mujoco.MjModel.from_xml_path(str(scene))
    data_a, data_b = mujoco.MjData(model_a), mujoco.MjData(model_b)
    _reset(model_a, data_a)
    _reset(model_b, data_b)
    meta_a = _configure_rough_random_scene(model_a, data_a, 11)
    meta_b = _configure_rough_random_scene(model_b, data_b, 11)
    assert meta_a == meta_b
    assert len(meta_a["random_obstacles"]) == 5
