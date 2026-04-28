{
  description = "C++ flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      python =
        pkgs.python313.withPackages
        (ps:
          with ps; [
            ipython
            numpy
            matplotlib
            polars
            fastexcel
          ]);
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = [python];
      };
    });
}
