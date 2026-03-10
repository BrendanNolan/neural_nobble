if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install conan
else
    source .venv/bin/activate
fi
