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
static int test_close(remote_handle64 handle) { (void)handle; return 0; }
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
