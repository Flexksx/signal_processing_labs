{
  description = "Program summary monitoring streamlit flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonPackages = pkgs.python3Packages;
      in
      {
        # Define the development shell for each system
        devShell = pkgs.mkShell {
          buildInputs = [
            pkgs.python3
            pythonPackages.pandas
            pythonPackages.python-dotenv
            pythonPackages.numpy
            pythonPackages.plotly
            pythonPackages.black
                      ];

          shellHook = ''
            echo "Welcome to the development shell for ${system}!"
          '';
        };

        pythonEnv = pkgs.python3.withPackages (
          ps: with ps; [
            pandas
            numpy
            python-dotenv
            plotly
            black
          ]
        );
      }
    );
}
