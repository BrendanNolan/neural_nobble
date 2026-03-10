if not test -d .venv
    python3 -m venv .venv
    source .venv/bin/activate.fish
    pip install conan
else
    source .venv/bin/activate.fish
end
