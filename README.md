# multigpu_diffusion_localai

A custom LocalAI backend comprised of various multi-GPU Diffusion solutions.

### DO NOT use this software to generate illegal or otherwise harmful content.
### DO NOT use this software to generate illegal or otherwise harmful content.
### DO NOT use this software to generate illegal or otherwise harmful content.

## Notes:
- Clone this repo inside LocalAI/backend/python folder.
- Run `setup.sh`.
- Edit `config.json`.
- Check `sample` folder for a sample model configuration.

## Currently Supported Models:
- Stable Diffusion 1.5 (sd1) (AsyncDiff or DistriFuser)
- Stable Diffusion 2 (sd2) (AsyncDiff or DistriFuser)
- Stable Diffusion 3 (sd3) (AsyncDiff or xDiT)
- Stable Diffusion XL (sdxl) (AsyncDiff or DistriFuser)
- Stable Diffusion Upscaler (sdup) (AsyncDiff)
- Stable Video Diffusion (svd) (AsyncDiff)
- FLUX.1 (flux) (xDiT)

## Read More:
- [AsyncDiff](https://github.com/czg1225/AsyncDiff).
- [DistriFuser](https://github.com/mit-han-lab/distrifuser).
- [xDiT](https://github.com/xdit-project/xDiT).

