#!/bin/zsh
# Clona SemIf en el commit evaluado e instala su venv (Python 3.12; 3.14 falla con numpy 2.2.6).
set -e
cd "$(dirname "$0")"
[ -d semif ] || git clone https://github.com/TheoLeeCJ/SemIf.git semif
git -C semif checkout -q 23cf1f3
uv venv --python 3.12 semif/.venv
(cd semif && . .venv/bin/activate && uv pip install -e '.[test,mlx]')
