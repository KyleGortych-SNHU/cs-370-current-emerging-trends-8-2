{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.python312Packages.virtualenv
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
    if [ ! -d .venv ]; then
      python -m venv .venv
      .venv/bin/pip install numpy tensorflow keras matplotlib ipykernel jupyterlab
    fi
    source .venv/bin/activate
  '';
}
