from pathlib import Path

import pytest


def test_all_supported_scenes_load_with_go2_contract():
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).parent / "assets" / "go2"
    scenes = [root / name for name in ("scene.xml", "stair.xml", "stairs_down.xml",
                                       "hfield.xml", "course_flat_obstacle.xml",
                                       "course_mixed.xml")]
    for scene in scenes:
        model = mujoco.MjModel.from_xml_path(str(scene))
        assert (model.nq, model.nu) == (19, 12)
