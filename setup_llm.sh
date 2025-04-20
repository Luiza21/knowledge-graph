# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv_llm --python=3.10
source .venv_llm/bin/activate
uv python install 3.10
uv pip install -r requirements_llm.txt --prerelease=allow