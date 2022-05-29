NVIDIA's Warp for Houdini
=========================

Exposes NVIDIA's Warp framework to Houdini.

NVIDIA's Warp provides access to running high-performance kernels either on
the CPU or GPU, all with the convenience of writing the code in Python.

See the official reference pages for more information:

* home page: <https://developer.nvidia.com/warp-python>
* documentation: <https://nvidia.github.io/warp>
* code repository: <https://github.com/NVIDIA/warp>


## Warning

This HDA is very much *NOT* production-ready and hasn't been thoroughly tested.

Use at your own risk!


## Build

An HDA file is already provided but it is also possible to build it anew using
the script `./bin/nvidiawarp`.


## Installation

1. Install Warp using the [official instructions][installation].
2. Make Warp's Python package available to Houdini—running `import warp` from
   within Houdini's shell should not error out.
3. Add the `./houdini` folder to the `HOUDINI_PATH` environment variable.


## Usage

The help page for the node comes with a couple of examples, check them out!


## Repository

<https://github.com/christophercrouzet/nvidia-warp-houdini>


## License

[Unlicense][unlicense].


[installation]: https://github.com/NVIDIA/warp#installing
[unlicense]: https://unlicense.org
