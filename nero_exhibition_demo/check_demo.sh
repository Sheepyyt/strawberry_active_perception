#!/usr/bin/env bash
set -eo pipefail

demo_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd "${demo_directory}/.." && pwd)"

source /opt/ros/jazzy/setup.bash
source "${project_directory}/.venv/bin/activate"
source "${project_directory}/nero_ws/install/setup.bash"
set -u

export PYTHONDONTWRITEBYTECODE=1

python -m flake8 \
    --config "${demo_directory}/setup.cfg" \
    "${demo_directory}"/*.py
python "${demo_directory}/test_demo_profiles.py" -v
python "${demo_directory}/test_profile_isolation.py" -v
python "${demo_directory}/show_demo.py" poses --preview-only
python "${demo_directory}/show_demo.py" trajectory --preview-only
