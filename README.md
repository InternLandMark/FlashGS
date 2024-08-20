# FlashGS
FlashGS is an efficient CUDA Python library, enabling real-time 3D Gaussian Splatting (3DGS) based rendering especially for large-scale and high-resolution scenes.

## Hardware Requirement
NVIDIA's server-grade and consumer-grade GPUs should work for our implementation. We have conducted our experiments on an NVIDIA A100, V100, RTX 2080ti, RTX 3090 and RTX 4090 GPUs.

## Directories
* `csrc/`: Our CUDA C++ implementation of FlashGS. The optimized rendering kernels are under `csrc/cuda_rasterizer/`.
* `flash_3dgs/`: An example to show how to use the installed FlashGS library.
* `setup.py`: A Python script to build, package, and install the FlashGS library. 
* `requirements.txt`: Record some software dependencies when installing FlashGS.

## Installation
You can follow the following steps to setup on your machine:
* Clone the FlashGS project from this page.
* Download the dependencies as we recommend.
* Use `python setup.py install` to install FlashGS library.
* Run the provided example with your datasets to test.
