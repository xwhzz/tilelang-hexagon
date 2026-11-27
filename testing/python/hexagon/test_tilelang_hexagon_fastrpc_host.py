"""Execute the generated one-shot driver against a native FastRPC stub."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tilelang.engine.param import KernelParam
from tilelang.hexagon.runtime import fastrpc


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a native C compiler is required")
    root = tmp_path_factory.mktemp("oneshot-driver")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fastrpc, "PD_SETUP", "")
        plans = fastrpc._plan([KernelParam("int32", [])], [])
        (root / "driver.c").write_text(fastrpc.gen_host("test", plans))
    headers = {
        "AEEStdErr.h": "#define AEE_SUCCESS 0\n",
        "remote.h": "#include <stdint.h>\ntypedef uint64_t remote_handle64;\n",
        "dsp_capabilities_utils.h": """#define MAX_DOMAIN_URI_SIZE 32
typedef struct { const char *uri; } domain;
static domain* get_domain(int id) {
  static char uri[32]; static domain d = {uri};
  snprintf(uri, sizeof(uri), "?domain=%d", id);
  return &d;
}
""",
        "test.h": """#define test_URI "test"
static int test_open(const char *uri, remote_handle64 *handle) {
  *handle = strtoul(strchr(uri, '=') + 1, 0, 10); return 0;
}
static int test_close(remote_handle64 handle) {
  (void)handle; return getenv("TL_TEST_CLOSE_ERROR") ? 0x1005 : 0;
}
static int test_run(remote_handle64 handle, int32_t value) {
  printf("domain=%llu value=%d\\n", (unsigned long long)handle, value); return 0;
}
""",
    }
    for name, text in headers.items():
        (root / name).write_text(text)
    executable = root / "driver"
    subprocess.run([compiler, "-std=c11", "-I", str(root), str(root / "driver.c"), "-o", str(executable)], check=True, timeout=30)
    return executable


@pytest.mark.parametrize("domain", [3, 4])
def test_one_shot_uses_runtime_domain_and_preserves_kernel_arguments(driver: Path, domain):
    result = subprocess.run([str(driver), str(domain), "-37"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"domain={domain} value=-37\n"


@pytest.mark.parametrize("args", [[], ["3"], ["-1", "0"], ["3x", "0"], [str(1 << 31), "0"], ["", "0"]])
def test_one_shot_rejects_bad_domain_or_argument_count_before_rpc(driver: Path, args):
    result = subprocess.run([str(driver), *args], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert not result.stdout


def test_one_shot_propagates_close_failure(driver, monkeypatch):
    monkeypatch.setenv("TL_TEST_CLOSE_ERROR", "1")
    result = subprocess.run([str(driver), "3", "-37"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "close failed: 0x1005" in result.stderr


@pytest.mark.parametrize(("value", "word"), [(0.375, 0x3EC00000), (-0.5, -1090519040), (-0.0, -(1 << 31)), (2**-149, 1)])
def test_one_shot_driver_transports_float32_bits(driver, value, word):
    (plan,) = fastrpc._plan([KernelParam("float32", [])], [])
    result = subprocess.run([str(driver), "3", str(plan.scalar_word(value))], capture_output=True, text=True, check=True, timeout=5)
    assert result.stdout == f"domain=3 value={word}\n"


def test_cooperative_dispatch_preserves_policy_and_first_error(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a native C++ compiler is required")
    harness = Path(__file__).with_name("cooperative_dispatch_host.cc").read_text()
    source = tmp_path / "cooperative_dispatch.cc"
    executable = tmp_path / "cooperative_dispatch"
    source.write_text(harness.replace("// GENERATED_RUN_KERNEL", fastrpc.RUN_KERNEL))
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True, timeout=30)
    for mode in ("ordinary", "cooperative", "resource-free"):
        for failure in ("none", "prepare-error", "body-error", "finish-error", "body-error-finish-error"):
            subprocess.run([str(executable), f"{mode}-{failure}"], check=True, timeout=5)
    subprocess.run([str(executable), "ordinary-take-error"], check=True, timeout=5)
    for failure in ("give-error", "body-error-give-error", "body-error-give-error-finish-error"):
        subprocess.run([str(executable), f"ordinary-{failure}"], check=True, timeout=5)
