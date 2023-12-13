# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import torch
from typing import List
from diffusers import AutoPipelineForText2Image

MODEL_NAME = "stabilityai/sdxl-turbo"
MODEL_CACHE = "model-cache"


class Predictor(BasePredictor):
    def setup(self) -> None:
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.vae.use_tiling = False

        from typing import Optional
        import diffusers
        from torch import Tensor
        from torch.nn import functional as F
        from torch.nn import Conv2d
        from torch.nn.modules.utils import _pair

        def asymmetricConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
            self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
            self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
            working = F.pad(input, self.paddingX, mode='circular')
            working = F.pad(working, self.paddingY, mode='circular')
            return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        targets = [pipe.vae, pipe.text_encoder, pipe.unet, ]
        conv_layers = []
        for target in targets:
            for module in target.modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(module)

        for cl in conv_layers:
            if isinstance(cl, diffusers.models.lora.LoRACompatibleConv) and cl.lora_layer is None:
                cl.lora_layer = lambda *x: 0

            cl._conv_forward = asymmetricConv2DConvForward.__get__(cl, torch.nn.Conv2d)

        if torch.cuda.is_available():
            pipe.to("cuda")
        self.pipe = pipe
        self.pipe.vae.enable_tiling()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="21 years old girl,short cut,beauty,dusk,Ghibli style illustration"
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="3d, cgi, render, bad quality, normal quality",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1, le=4, default=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        width: int = Input(
            description="Width of the output image",
            ge=1, le=1024, default=512,
        ),
        height: int = Input(
            description="Height of the output image",
            ge=1, le=1024, default=512,
        ),
        final_width: int = Input(
            description="Width of the final output image",
            ge=1, le=1024, default=512,
        ),
        final_height: int = Input(
            description="Height of the final output image",
            ge=1, le=1024, default=512,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": 0,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
        }

        output = self.pipe(**common_args)

        output_paths = []
        for i, image in enumerate(output.images):
            if final_width != width or final_height != height:
                image = image.resize((final_width, final_height))
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
        