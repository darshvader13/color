import modal
import io
import base64
from pathlib import Path
from fastapi import UploadFile, File

app = modal.App("deoldify-colorization")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6", "libxrender-dev")
    .run_commands(
        "git clone https://github.com/darshvader13/color.git /root/color",
    )
    .pip_install(
        "numpy==1.21.6",
        "torch==1.11.0",
        "torchvision==0.12.0",
        "fastai==1.0.60",
        "Pillow==9.3.0",
        "opencv-python>=4.2.0.32",
        "matplotlib",
        "wandb",
        "tensorboardX>=1.6",
        "ffmpeg-python",
        "yt-dlp",
        "ipywidgets",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
    )
)

volume = modal.Volume.from_name("model-storage", create_if_missing=False)

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},
    keep_warm=0,
)
class MangaColorizer:

    @modal.enter()
    def setup(self):
        import sys
        import warnings
        import torch
        warnings.filterwarnings("ignore")
        
        sys.path.insert(0, '/root/color')
        sys.path.insert(0, '/root/color/deoldify')
        
        from deoldify.visualize import ModelImageVisualizer
        from deoldify.filters import MasterFilter, ColorizerFilter
        from deoldify.generators import gen_learner_new
        from deoldify.dataset import get_dummy_databunch
        
        print("Loading generator...")
        print(f"Looking for models in: /models")
        print(f"Available files: {list(Path('/models').glob('*'))}")

        self.generator = gen_learner_new(
            data=get_dummy_databunch(),
            gen_loss=torch.nn.functional.l1_loss,
            root_folder=Path('/'),
            weights_name='ColorizeArtistic_gen'
        ).load('finetuned_default_generator', with_opt=False)
        
        self.rf_factor = 10
        filtr = MasterFilter([ColorizerFilter(learn=self.generator)], render_factor=self.rf_factor)
        self.vis = ModelImageVisualizer(filtr, results_dir='/tmp/results')
        
        print("Model loaded successfully!")
    
    @modal.method()
    def colorize(self, image_bytes: bytes) -> bytes:
        from PIL import Image
        import io
        
        input_image = Image.open(io.BytesIO(image_bytes))
        
        input_image = input_image.resize((280, 400), Image.Resampling.LANCZOS)
        
        temp_path = '/tmp/input_image.jpg'
        input_image.save(temp_path)
        
        result_image = self.vis.get_transformed_image(
            path=temp_path,
            render_factor=self.rf_factor,
            watermarked=False
        )
        
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG', quality=95)
        output_bytes = output_buffer.getvalue()
        
        return output_bytes


@app.function(image=image)
@modal.web_endpoint(method="POST")
async def colorize_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    colorizer = MangaColorizer()
    result_bytes = colorizer.colorize.remote(image_bytes)
    
    image_base64 = base64.b64encode(result_bytes).decode('utf-8')
    
    return {"image_base64": image_base64, "status": "success"}
