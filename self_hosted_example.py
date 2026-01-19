"""EXAMPLE WORKFLOW WITH JUGGERNAUT MODEL + IP-ADAPTER---THIS WORKFLOW ASSUMES SELF-HOSTING MODEL"""


import torch
import os
import json
import random
from pathlib import Path
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import numpy as np

class ConsistentCharacterGenerator:
    def __init__(self, 
                 base_model="RunDiffusion/Juggernaut-XL-v9",
                 output_dir="character_dataset",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading {base_model}...")
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)
        
        # Use DDIM scheduler for better face consistency
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # Load IP-Adapter FaceID for maximum character consistency
        print("Loading IP-Adapter FaceID...")
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sdxl.bin",
            image_encoder_folder=None
        )
        
        # Set IP-Adapter influence (0.6-0.8 for good balance)
        self.pipeline.set_ip_adapter_scale(0.7)
        
        # Enable memory optimization
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        
        # Define character prompts for variety
        self.character_prompts = {
            "portraits": [
                "portrait of a beautiful woman, {expression}, professional photography, high quality, detailed face, natural lighting",
                "close-up portrait, {expression}, soft lighting, cinematic, 85mm lens",
                "headshot, {expression}, studio lighting, professional headshot",
            ],
            "expressions": ["smiling", "serious", "thoughtful", "confident", "playful", "elegant"],
            "environments": [
                "in a cozy cafe", "in a modern office", "outdoor natural light", 
                "studio background", "urban setting", "home environment"
            ],
            "styles": [
                "professional photography", "cinematic style", "natural lighting",
                "soft focus", "high contrast", "warm tones", "cool tones"
            ]
        }
    
    def generate_character_dataset(self, reference_image_path, num_images=80, batch_size=4):
        """Generate consistent character images for LoRA training"""
        
        print(f"Loading reference image: {reference_image_path}")
        reference_image = load_image(reference_image_path)
        
        # Save reference image
        reference_image.save(self.output_dir / "reference_image.jpg")
        
        # Generate images in batches
        generated_images = []
        metadata = []
        
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            print(f"Generating batch {batch_start//batch_size + 1}/{(num_images-1)//batch_size + 1}")
            
            # Generate batch
            batch_images, batch_metadata = self._generate_batch(
                reference_image, batch_start, batch_end
            )
            
            generated_images.extend(batch_images)
            metadata.extend(batch_metadata)
            
            # Save batch
            self._save_batch(batch_images, batch_metadata, batch_start)
        
        # Save metadata
        self._save_metadata(metadata)
        
        print(f"Dataset generation complete! Saved {len(generated_images)} images to {self.output_dir}")
        return generated_images, metadata
    
    def _generate_batch(self, reference_image, start_idx, end_idx):
        """Generate a batch of images"""
        
        batch_images = []
        batch_metadata = []
        
        # Generate different variations
        for i in range(start_idx, end_idx):
            # Randomize parameters
            expression = random.choice(self.character_prompts["expressions"])
            environment = random.choice(self.character_prompts["environments"])
            style = random.choice(self.character_prompts["styles"])
            portrait_type = random.choice(self.character_prompts["portraits"])
            
            # Build prompt
            prompt = portrait_type.format(expression=expression)
            prompt += f", {environment}, {style}"
            
            # Add some randomization
            seed = random.randint(0, 1000000)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Vary IP-Adapter influence slightly for diversity
            ip_scale = 0.6 + random.random() * 0.2  # 0.6-0.8
            
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    ip_adapter_image=reference_image,
                    num_inference_steps=30,  # Good balance of quality/speed
                    guidance_scale=7.5,  # Standard guidance
                    ip_adapter_scale=ip_scale,
                    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality, blurry, out of focus",
                    generator=generator,
                    width=1024,
                    height=1024,
                )
            
            image = result.images[0]
            batch_images.append(image)
            
            # Store metadata
            metadata = {
                "image_id": f"character_{i:04d}",
                "prompt": prompt,
                "seed": seed,
                "ip_adapter_scale": ip_scale,
                "expression": expression,
                "environment": environment,
                "style": style
            }
            batch_metadata.append(metadata)
        
        return batch_images, batch_metadata
    
    def _save_batch(self, images, metadata, batch_start):
        """Save a batch of images"""
        
        batch_dir = self.output_dir / f"batch_{batch_start//4:03d}"
        batch_dir.mkdir(exist_ok=True)
        
        for img, meta in zip(images, metadata):
            filename = f"{meta['image_id']}.jpg"
            img.save(batch_dir / filename, quality=95)
    
    def _save_metadata(self, metadata):
        """Save metadata for training"""
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a simple text file with prompts for easy viewing
        prompts_path = self.output_dir / "prompts.txt"
        with open(prompts_path, 'w') as f:
            for meta in metadata:
                f.write(f"{meta['image_id']}: {meta['prompt']}\n")
    
    def cleanup(self):
        """Cleanup and optimize memory"""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache()


# 🎯 **USAGE EXAMPLE**
def main():
    """Main function to generate your character dataset"""
    
    # Initialize generator
    generator = ConsistentCharacterGenerator(
        base_model="RunDiffusion/Juggernaut-XL-v9",
        output_dir="ai_influencer_dataset",
        device="cuda"
    )
    
    # Generate dataset
    # Replace with your reference character image
    reference_image_path = "path/to/your/character_reference.jpg"
    
    try:
        images, metadata = generator.generate_character_dataset(
            reference_image_path=reference_image_path,
            num_images=80,  # Generate 80 images
            batch_size=4    # Process 4 at a time
        )
        
        print("✅ Dataset generation complete!")
        print(f"📁 Images saved to: {generator.output_dir}")
        print(f"📊 Total images: {len(images)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        generator.cleanup()


# 🛠️ **ADVANCED CONFIGURATION OPTIONS**

def create_advanced_generator():
    """Create generator with custom settings"""
    
    # For even more consistency, you can use IP-Adapter Plus Face
    # instead of regular FaceID
    
    from transformers import CLIPVisionModelWithProjection
    
    # Load with custom image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16
    )
    
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        image_encoder=image_encoder,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Use IP-Adapter Plus Face (even better consistency)
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
    )
    
    return pipeline


if __name__ == "__main__":
    main()
