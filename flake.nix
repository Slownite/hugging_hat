{
  description = "A Python project structure using uv, nix flakes, justfile, and click";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgsFor = system: import nixpkgs { inherit system; };
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python313
              uv
              just
            ];
            
            shellHook = ''
              export UV_PROJECT_ENVIRONMENT=.venv
              # Ensures uv uses the python from nixpkgs
              export UV_PYTHON=${pkgs.python313}/bin/python
              # Put the project venv's bin on PATH so installed scripts (e.g. `hh`) are available
              export PATH="$PWD/.venv/bin:$PATH"
            '';
          };
        });
    };
}
