import modal
import io
import base64
from pathlib import Path
from fastapi import UploadFile, File

# Create Modal app
app = modal.App("deoldify-colorization")

# Create image with dependencies
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
        # Add compatible FastAPI and Pydantic for Modal endpoints
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
    )
)

# Create volume for model storage
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
        """Load the model when container starts"""
        import sys
        import warnings
        import torch
        warnings.filterwarnings("ignore")
        
        # Add paths
        sys.path.insert(0, '/root/color')
        sys.path.insert(0, '/root/color/deoldify')
        
        # Import DeOldify modules explicitly
        from deoldify.visualize import ModelImageVisualizer
        from deoldify.filters import MasterFilter, ColorizerFilter
        from deoldify.generators import gen_learner_new
        from deoldify.dataset import get_dummy_databunch
        
        # Load generator
        print("Loading generator...")
        print(f"Looking for models in: /models")
        print(f"Available files: {list(Path('/models').glob('*'))}")

        self.generator = gen_learner_new(
            data=get_dummy_databunch(),
            gen_loss=torch.nn.functional.l1_loss,
            root_folder=Path('/'),
            weights_name='ColorizeArtistic_gen'
        ).load('finetuned_default_generator', with_opt=False)
        
        # Setup filter and visualizer
        self.rf_factor = 10
        filtr = MasterFilter([ColorizerFilter(learn=self.generator)], render_factor=self.rf_factor)
        self.vis = ModelImageVisualizer(filtr, results_dir='/tmp/results')
        
        print("Model loaded successfully!")
    
    @modal.method()
    def colorize(self, image_bytes: bytes) -> bytes:
        """
        Colorize a manga panel
        
        Args:
            image_bytes: Input image as bytes
            
        Returns:
            Colorized image as bytes
        """
        from PIL import Image
        import io
        
        # Load image from bytes
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Resize to target dimensions (280, 400)
        input_image = input_image.resize((280, 400), Image.Resampling.LANCZOS)
        
        # Save temporarily
        temp_path = '/tmp/input_image.jpg'
        input_image.save(temp_path)
        
        # Run colorization
        result_image = self.vis.get_transformed_image(
            path=temp_path,
            render_factor=self.rf_factor,
            watermarked=False
        )
        
        # Convert PIL Image to bytes
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG', quality=95)
        output_bytes = output_buffer.getvalue()
        
        return output_bytes


@app.function(image=image)
@modal.web_endpoint(method="POST")
async def colorize_endpoint(file: UploadFile = File(...)):
    """
    Web endpoint for colorizing manga panels
    
    Returns JSON with base64 encoded image
    """
    # Read uploaded file
    image_bytes = await file.read()
    
    # Colorize using the model
    colorizer = MangaColorizer()
    result_bytes = colorizer.colorize.remote(image_bytes)
    
    # Encode to base64
    image_base64 = base64.b64encode(result_bytes).decode('utf-8')
    
    # Return as JSON
    return {"image_base64": image_base64, "status": "success"}
