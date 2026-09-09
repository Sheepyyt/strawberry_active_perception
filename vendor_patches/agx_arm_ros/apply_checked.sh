#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${script_directory}/../.." && pwd)"
vendor_directory="${project_directory}/nero_ws/src/agx_arm_ros"
patch_file="${script_directory}/nero_v111_safety.patch"
expected_commit="22a9cf6c5ad2fd2e0743531936bc5dab007fa5bc"
expected_sha256="394542caff32f5d147a7a33059a0fa078e96416276f0a61d9fed62b48e8104d4"

actual_commit="$(git -C "${vendor_directory}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "AGX 版本不匹配：期望 ${expected_commit}，实际 ${actual_commit}" >&2
    exit 2
fi

actual_sha256="$(sha256sum "${patch_file}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "安全补丁 SHA256 不匹配，拒绝应用。" >&2
    exit 2
fi

# A zero-context patch is safe here only because both the vendor commit and the
# patch digest were checked above. Suppress the expected reverse-check error on
# a clean checkout so first-time installation output stays readable.
if git -C "${vendor_directory}" apply --unidiff-zero --reverse --check \
    "${patch_file}" 2>/dev/null; then
    echo "AGX 安全补丁已经完整存在，无需重复应用。"
    exit 0
fi

if [[ -n "$(git -C "${vendor_directory}" status --porcelain)" ]]; then
    echo "AGX 子模块含有其它修改，拒绝自动覆盖。" >&2
    exit 2
fi

git -C "${vendor_directory}" apply --unidiff-zero --check "${patch_file}"
git -C "${vendor_directory}" apply --unidiff-zero "${patch_file}"
echo "AGX 安全补丁已应用；请运行对应测试。"
