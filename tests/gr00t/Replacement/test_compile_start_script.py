"""CPU smoke tests for compile settings passed through the training launcher."""

import os
import shutil
import subprocess
from pathlib import Path


def test_start_passes_compile_scopes_to_finetune(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    shutil.copy(repo / "start.sh", tmp_path / "start.sh")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "finetune.sh").write_text(
        'printf "SCOPES=%s,%s,%s\\n" "$GR00T_COMPILE_DIT" '
        '"$GR00T_COMPILE_VL" "$GR00T_COMPILE_TEXT"\n'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("GR00T_")}
    env.update({"GROOT_DATA_ROOT": str(tmp_path / "data"), "GR00T_COMPILE_DIT": "model",
                "GR00T_COMPILE_VL": "1", "GR00T_COMPILE_TEXT": "forwards"})
    result = subprocess.run(["bash", str(tmp_path / "start.sh")], env=env,
                            capture_output=True, text=True, check=True, timeout=10)
    assert "SCOPES=model,1,forwards" in result.stdout
