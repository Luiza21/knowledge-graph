# Run this once for installation

# curl -LsSf https://astral.sh/uv/install.sh | sh
# uv venv .venv --python=3.9 & source .venv/bin/activate
# uv python install 3.9
# uv pip install -U pip
# uv pip install -r requirements.txt --prerelease=allow

# python -m spacy download en_core_web_lg
# python -m coreferee install en


# Run this to create graph
uv run --python 3.10 make_graph.py