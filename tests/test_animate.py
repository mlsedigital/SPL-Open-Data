import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANIMATE_PATH = REPO_ROOT / "basketball" / "freethrow" / "animate.py"


def _load_animate_module():
    spec = importlib.util.spec_from_file_location("animate_module", ANIMATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trial(path: Path, sampling_rate):
    trial = {
        "sampling_rate": sampling_rate,
        "tracking": [
            {
                "frame": 0,
                "time": 0,
                "data": {
                    "ball": [1.0, 2.0, 3.0],
                    "player": {
                        "RIGHT_HIP": [0.0, 0.0, 1.0],
                        "LEFT_HIP": [1.0, 0.0, 1.0],
                        "RIGHT_KNEE": [0.0, 0.0, 0.5],
                        "LEFT_KNEE": [1.0, 0.0, 0.5],
                    },
                },
            },
            {
                "frame": 1,
                "time": 16,
                "data": {
                    "ball": [1.1, 2.1, 3.1],
                    "player": {
                        "RIGHT_HIP": [0.1, 0.0, 1.0],
                        "LEFT_HIP": [1.1, 0.0, 1.0],
                        "RIGHT_KNEE": [0.1, 0.0, 0.5],
                        "LEFT_KNEE": [1.1, 0.0, 0.5],
                    },
                },
            },
        ],
    }
    path.write_text(json.dumps(trial))


def test_resolve_joint_name_uses_alias_when_needed():
    animate = _load_animate_module()
    available_joints = {"RIGHT_HIP", "LEFT_HIP"}
    assert animate._resolve_joint_name("R_HIP", available_joints) == "RIGHT_HIP"
    assert animate._resolve_joint_name("L_HIP", available_joints) == "LEFT_HIP"
    assert animate._resolve_joint_name("R_WRIST", available_joints) is None


def test_animate_trial_uses_trial_sampling_rate_for_interval(tmp_path, monkeypatch):
    animate = _load_animate_module()
    trial_path = tmp_path / "trial_60fps.json"
    _write_trial(trial_path, sampling_rate=60)

    captured = {}

    def fake_func_animation(fig, update, frames, interval):
        captured["frames"] = frames
        captured["interval"] = interval
        return {"ok": True}

    monkeypatch.setattr(animate, "FuncAnimation", fake_func_animation)

    anim = animate.animate_trial(
        str(trial_path),
        show_court=False,
        notebook_mode=False,
        connections=[("R_HIP", "R_KNEE"), ("L_HIP", "L_KNEE")],
    )

    assert anim == {"ok": True}
    assert captured["frames"] == 2
    assert captured["interval"] == 1000 / 60


def test_animate_trial_falls_back_to_30fps_for_invalid_sampling_rate(tmp_path, monkeypatch):
    animate = _load_animate_module()
    trial_path = tmp_path / "trial_invalid_fps.json"
    _write_trial(trial_path, sampling_rate="bad_value")

    captured = {}

    def fake_func_animation(fig, update, frames, interval):
        captured["interval"] = interval
        return {"ok": True}

    monkeypatch.setattr(animate, "FuncAnimation", fake_func_animation)

    animate.animate_trial(
        str(trial_path),
        show_court=False,
        notebook_mode=False,
        connections=[("R_HIP", "R_KNEE"), ("L_HIP", "L_KNEE")],
    )

    assert captured["interval"] == 1000 / 30
