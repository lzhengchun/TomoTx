# TomoTx

Transformer for the analysis of Sinogram for Computed Tomography.

This repo hosts code for the paper, `Masked Sinogram Model with Transformer for ill-Posed Computed Tomography Reconstruction: a Preliminary Study`, in prepration.
It is not progressing fast because of short of hands, if one really wants to see the draft, it's [here](https://arxiv.org/abs/2209.01356).

- ```dataset``` folder has the code and instruction to generate the synthesized used in the code/paper. 
- ```SinoTx``` (i.e., the Masked Sinogram Model) and each application folder has a ```config``` folder that contains yaml configuration files for different experiment scenarios. 
- One is expected to only modify the configuration file in most reproduce/evaluation cases. The default yaml files come from my own experiment setup, at least filenames to datasets need adjustment to run in your environment.

## Citation

If you find my code useful for your research, consider cite:

```
@misc{https://doi.org/10.48550/arxiv.2209.01356,
  doi = {10.48550/ARXIV.2209.01356},
  url = {https://arxiv.org/abs/2209.01356},
  author = {Liu, Zhengchun and Kettimuthu, Rajkumar and Foster, Ian},
  title = {Masked Sinogram Model with Transformer for ill-Posed Computed Tomography Reconstruction: a Preliminary Study},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

