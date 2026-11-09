#!/usr/bin/env bash
# Build gpuRIR (AGPL-3.0, GPU image-source RIR simulator) into the Banelp research venv.
# Everything lives under --runtime-dir; nothing is installed system-wide and nothing is added to the repository.
# There is no CPU fallback: on failure this exits non-zero and the RIR engine is NOT substituted.
set -euo pipefail
runtime_dir= artifact_dir= cuda=12.6 arch=75 rev=06da27dd
while (($#)); do
  case "$1" in
    --runtime-dir) runtime_dir=${2:?}; shift 2;;
    --artifact-dir) artifact_dir=${2:?}; shift 2;;
    --cuda) cuda=${2:?}; shift 2;;
    --arch) arch=${2:?}; shift 2;;
    --gpurir-rev) rev=${2:?}; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n $runtime_dir && -n $artifact_dir ]] || { echo 'both --runtime-dir and --artifact-dir are required' >&2; exit 2; }
R=$(realpath -m "$runtime_dir"); A=$(realpath -m "$artifact_dir"); P=$R/env/bin/python
TC=$R/gpurir-toolchain SRC=$R/src/gpuRIR MM=$R/tools/micromamba LOGS=$A/evidence/gpurir-build
[[ -x $P ]] || { echo "research venv missing: run bootstrap.sh first ($P)" >&2; exit 3; }
mkdir -p "$R/tools" "$R/src" "$R/tmp" "$LOGS" "$A/evidence"
export TMPDIR=$R/tmp MAMBA_ROOT_PREFIX=$R/tools/mamba-root PIP_CACHE_DIR=$R/cache/pip XDG_CACHE_HOME=$R/cache
step() { echo "== $1"; }
run() { local name=$1; shift; "$@" >"$LOGS/$name.log" 2>&1 || { echo "step '$name' failed; see $LOGS/$name.log" >&2; tail -n 25 "$LOGS/$name.log" >&2; exit 1; }; }

step micromamba
if [[ ! -x $MM ]]; then
  run micromamba-download curl -fsSL --retry 3 -o "$MM.part" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
  chmod +x "$MM.part"; mv "$MM.part" "$MM"
fi
mm_sha=$(sha256sum "$MM" | cut -d' ' -f1)

step toolchain
if [[ ! -x $TC/bin/nvcc ]]; then
  run toolchain "$MM" create -y -p "$TC" -c conda-forge --override-channels "cuda-version=$cuda" cuda-nvcc cuda-cudart-dev \
    libcufft-dev libcurand-dev 'gxx_linux-64=13' 'cmake>=3.23,<4' patchelf
fi
CXXBIN=$TC/bin/x86_64-conda-linux-gnu-g++
[[ -x $CXXBIN ]] || { echo "conda g++ missing: $CXXBIN" >&2; exit 1; }
export PATH=$TC/bin:$PATH CUDACXX=$TC/bin/nvcc CUDAHOSTCXX=$CXXBIN CXX=$CXXBIN CUDAToolkit_ROOT=$TC

step source
if [[ ! -d $SRC/.git ]]; then run clone git clone https://github.com/DavidDiazGuerra/gpuRIR "$SRC"; fi
run checkout git -C "$SRC" checkout --force "$rev"
if [[ -f $SRC/.gitmodules ]]; then run submodule git -C "$SRC" submodule update --init --recursive; fi
commit=$(git -C "$SRC" rev-parse HEAD)
[[ -f $SRC/third_party/pybind11/CMakeLists.txt ]] || { echo "pybind11 sources missing under $SRC/third_party" >&2; exit 1; }
git -C "$SRC" checkout --force -- CMakeLists.txt
sed -i "s/CUDA_ARCHITECTURES all-major/CUDA_ARCHITECTURES $arch/" "$SRC/CMakeLists.txt"
git -C "$SRC" diff -- CMakeLists.txt > "$LOGS/cmake-arch.patch" || true
[[ -s $LOGS/cmake-arch.patch ]] || { echo "CMake architecture patch did not apply" >&2; exit 1; }

step build
build_path=pip
if ! "$P" -m pip install --no-deps --force-reinstall "$SRC" >"$LOGS/pip-build.log" 2>&1; then
  echo "pip build failed (see $LOGS/pip-build.log); trying direct cmake build" >&2
  build_path=cmake-fallback; rm -rf "$SRC/build"
  run cmake-configure cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" -DCUDAToolkit_ROOT="$TC" -DPython_EXECUTABLE="$P" -DPYTHON_EXECUTABLE="$P"
  run cmake-build cmake --build "$SRC/build" -j8
  site=$("$P" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  mkdir -p "$site/gpuRIR"; cp "$SRC/gpuRIR/__init__.py" "$site/gpuRIR/"
  so=$(find "$SRC/build" -name 'gpuRIR_bind*.so' | head -n1); [[ -n $so ]] || { echo "no gpuRIR_bind*.so produced" >&2; exit 1; }
  cp "$so" "$site/gpuRIR/"
fi

step verify
modfile=$("$P" - <<'PY'
import importlib.util
print(importlib.util.find_spec("gpuRIR").origin)
PY
)
moddir=$(dirname "$modfile")
for so in "$moddir"/*.so; do patchelf --set-rpath "$TC/lib" "$so"; done
env -u LD_LIBRARY_PATH "$P" - >"$LOGS/import.log" 2>&1 <<'PY' || { echo "import gpuRIR failed; see $LOGS/import.log" >&2; cat "$LOGS/import.log" >&2; exit 1; }
import inspect, gpuRIR
print(inspect.signature(gpuRIR.simulateRIR))
for name in ("activateMixedPrecision", "activateLUT", "beta_SabineEstimation", "att2t_SabineEstimator", "t2n", "simulateRIR"):
    assert callable(getattr(gpuRIR, name)), name
PY
for so in "$moddir"/*.so; do ldd "$so" >>"$LOGS/ldd.log" 2>&1 || true; done
! grep -q 'not found' "$LOGS/ldd.log" || { echo "unresolved shared libraries; see $LOGS/ldd.log" >&2; exit 1; }

step record
BUILD_PATH=$build_path COMMIT=$commit MM_SHA=$mm_sha ARCH=$arch CUDA=$cuda TC=$TC MODDIR=$moddir PATCH=$LOGS/cmake-arch.patch "$P" - "$TC/build-record.json" "$A/evidence/gpurir-build-record.json" <<'PY'
import hashlib, json, os, subprocess, sys, datetime, glob
def out(*cmd): return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
tc = os.environ["TC"]
record = {"gpurir_commit": os.environ["COMMIT"], "gpurir_license": "AGPL-3.0", "source_url": "https://github.com/DavidDiazGuerra/gpuRIR",
          "arch_patch": {"cuda_architectures": os.environ["ARCH"], "diff": open(os.environ["PATCH"]).read()},
          "build_path": os.environ["BUILD_PATH"],
          "nvcc_version": [l for l in subprocess.run([f"{tc}/bin/nvcc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines() if "release" in l][0].strip(),
          "gxx_version": out(f"{tc}/bin/x86_64-conda-linux-gnu-g++", "--version"), "cmake_version": out(f"{tc}/bin/cmake", "--version"),
          "cuda_version": os.environ["CUDA"], "micromamba_sha256": os.environ["MM_SHA"],
          "extension_sha256": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sorted(glob.glob(os.environ["MODDIR"] + "/*.so"))},
          "python": sys.version.split()[0], "torch_cuda": __import__("torch").version.cuda,
          "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
for path in sys.argv[1:]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(record, indent=2, sort_keys=True))
PY
echo "gpuRIR $commit ready ($build_path); build record: $A/evidence/gpurir-build-record.json"
