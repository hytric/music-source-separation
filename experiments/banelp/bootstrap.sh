#!/usr/bin/env bash
set -euo pipefail
runtime_dir= artifact_dir=
while (($#)); do
  case "$1" in
    --runtime-dir) runtime_dir=${2:?}; shift 2;;
    --artifact-dir) artifact_dir=${2:?}; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n $runtime_dir && -n $artifact_dir ]] || { echo 'both --runtime-dir and --artifact-dir are required' >&2; exit 2; }
python_bin=$(command -v python3.12 || true)
[[ -n $python_bin ]] || { echo 'Python 3.12 is required; install/provision it under runtime_dir before bootstrap' >&2; exit 3; }
mkdir -p "$runtime_dir/env" "$runtime_dir/cache/torch" "$runtime_dir/cache/huggingface" "$runtime_dir/tmp" "$artifact_dir/evidence"
export HF_HOME="$runtime_dir/cache/huggingface" TORCH_HOME="$runtime_dir/cache/torch" XDG_CACHE_HOME="$runtime_dir/cache" TMPDIR="$runtime_dir/tmp"
"$python_bin" -m venv "$runtime_dir/env"
"$runtime_dir/env/bin/python" -m pip install --upgrade 'pip==25.1.1'
"$runtime_dir/env/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r "$(dirname "$0")/requirements.lock"
"$runtime_dir/env/bin/python" -m pip freeze --all > "$artifact_dir/evidence/requirements.resolved.lock"
"$runtime_dir/env/bin/python" - <<'PY' > "$artifact_dir/evidence/environment.json"
import importlib.metadata as md, json, platform, sys
mods=('numpy','scipy','soundfile','torch','torchaudio','demucs','norbert')
print(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{m:md.version(m) for m in mods}},indent=2,sort_keys=True))
PY
"$runtime_dir/env/bin/python" - <<'PY'
import numpy, scipy, soundfile, torch, torchaudio, demucs, norbert
print('all research imports passed')
PY
