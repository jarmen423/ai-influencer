import replicate
import requests
import json
import os
import time
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import random

class ReplicateCharacterGenerator:
    """Generate consistent character images using Replicate's IP-Adapter API"""
    
    def __init__(self, api_token=None, output_dir="character_dataset"):
        """
        Initialize the generator with Replicate API
        
        Args:
            api_token: Replicate API token (get from https://replicate.com/account/api-tokens)
            output_dir: Directory to save generated images
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found. Get it from https://replicate.com/account/api-tokens")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Replicate client
        self.client = replicate.Client(api_token=self.api_token)
        
        # Model selection - best options for your use case
        self.models = {
            "sdxl_face": "lucataco/ip_adapter-sdxl-face:5d91b5d68c53164b108768eb8053b130427f8c422c3b5ebbe5e6020b34066508",
            "instant_id_plus": "zsxkib/instant-id-ipadapter-plus-face:32402fb5c493d883aa6cf098ce3e4cc80f1fe6871f6ae7f632a8dbde01a3d161"
        }
        
        # Choose the best model for your needs
        self.model_version = self.models["sdxl_face"]  # Best balance of quality/cost
        
        # Character prompt variations
        self.character_variations = {
            "expressions": ["smiling", "serious", "thoughtful", "confident", "playful", "elegant", "natural", "warm"],
            "environments": ["studio background", "natural outdoor light", "cozy indoor", "urban setting", "soft background blur"],
            "styles": ["professional photography", "cinematic lighting", "soft focus", "high contrast", "warm tones", "cool tones"],
            "poses": ["portrait", "3/4 view", "close-up", "side profile", "looking up", "looking down"]
        }
    
    def upload_reference_image(self, image_path):
        """Upload reference image to Replicate"""
        # Replicate can accept URLs or base64 encoded images
        # Method 1: Upload to a temporary URL service
        # Method 2: Use base64 encoding
        
        with open(image_path, 'rb') as f:
            img_data = f.read()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Create data URL
        img_type = "image/jpeg" if image_path.endswith('.jpg') or image_path.endswith('.jpeg') else "image/png"
        data_url = f"data:{img_type};base64,{img_base64}"
        
        return data_url
    
    def generate_character_dataset(self, reference_image_path, num_images=80, cost_estimate=True):
        """
        Generate consistent character images for LoRA training
        
        Args:
            reference_image_path: Path to your reference character image
            num_images: Number of images to generate (60-80 recommended)
            cost_estimate: Show cost estimate before starting
            
        Returns:
            List of generated image paths and metadata
        """
        
        # Upload reference image
        print("📤 Uploading reference image...")
        reference_url = self.upload_reference_image(reference_image_path)
        
        # Cost estimate
        if cost_estimate:
            estimated_cost = num_images * 0.095  # ~$0.095 per run
            print(f"💰 Estimated cost: ${estimated_cost:.2f} for {num_images} images")
            print("Continue? (y/n): ", end="")
            if input().lower() != 'y':
                print("❌ Cancelled")
                return []
        
        print(f"🎬 Starting generation of {num_images} images...")
        
        metadata = []
        generated_paths = []
        
        for i in range(num_images):
            print(f"🖼️  Generating image {i+1}/{num_images}")
            
            # Generate varied prompt for consistency + diversity
            prompt = self._create_varied_prompt()
            
            try:
                # Run inference
                output = self.client.run(
                    self.model_version,
                    input={
                        "image": reference_url,
                        "prompt": prompt,
                        "num_inference_steps": 30,  # Good balance of quality/speed
                        "guidance_scale": 7.5,       # Standard guidance
                        "negative_prompt": "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality, blurry, out of focus",
                        "width": 1024,
                        "height": 1024,
                        "ip_adapter_scale": 0.7,     # Consistency level
                        "style_strength": 0.6        # Style influence
                    }
                )
                
                # Save image
                image_path = self._save_image(output, i, prompt)
                generated_paths.append(image_path)
                
                # Store metadata
                metadata.append({
                    "image_id": f"character_{i:04d}",
                    "prompt": prompt,
                    "file_path": str(image_path),
                    "generation_number": i
                })
                
                print(f"✅ Saved: {image_path.name}")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error generating image {i+1}: {str(e)}")
                continue
        
        # Save metadata
        self._save_metadata(metadata)
        
        print(f"🎉 Dataset generation complete!")
        print(f"📁 Saved {len(generated_paths)} images to {self.output_dir}")
        print(f"💰 Total cost: ${len(generated_paths) * 0.095:.2f}")
        
        return generated_paths, metadata
    
    def _create_varied_prompt(self):
        """Create varied prompts for consistency + diversity"""
        
        # Base character description (keep consistent)
        base_character = "beautiful woman, consistent face, same person"
        
        # Random variations
        expression = random.choice(self.character_variations["expressions"])
        environment = random.choice(self.character_variations["environments"])
        style = random.choice(self.character_variations["styles"])
        pose = random.choice(self.character_variations["poses"])
        
        # Build prompt
        prompt = f"{base_character}, {pose}, {expression}, {environment}, {style}, professional photography, high quality, detailed face"
        
        return prompt
    
    def _save_image(self, output_url, index, prompt):
        """Save generated image from Replicate URL"""
        
        # Download image
        response = requests.get(output_url)
        img = Image.open(BytesIO(response.content))
        
        # Save with consistent naming
        filename = f"character_{index:04d}.jpg"
        image_path = self.output_dir / filename
        
        # Save with high quality
        img.save(image_path, quality=95)
        
        return image_path
    
    def _save_metadata(self, metadata):
        """Save generation metadata for LoRA training"""
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a simple text file with prompts
        prompts_path = self.output_dir / "prompts.txt"
        with open(prompts_path, 'w') as f:
            for meta in metadata:
                f.write(f"{meta['image_id']}: {meta['prompt']}\n")
    
    def get_status(self):
        """Check API status and credits"""
        try:
            # This is a simple check - you might need to implement your own
            return {"status": "connected", "api_token": "set" if self.api_token else "not set"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# 🚀 **ALTERNATIVE: Hugging Face Inference API**
class HuggingFaceCharacterGenerator:
    """Generate character images using Hugging Face Inference API"""
    
    def __init__(self, api_token=None, output_dir="hf_character_dataset"):
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # HF Inference API endpoint
        self.api_base = "https://api-inference.huggingface.co/models"
        
        # Custom endpoint for IP-Adapter models (you'd need to deploy one)
        self.model_id = "your-deployed-model"  # You'd deploy this yourself
        
    def generate_image(self, reference_image_path, prompt):
        """Generate single image via HF API"""
        
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # This would require a custom deployed model
        # HF doesn't have pre-built IP-Adapter + SDXL like Replicate
        
        return "Not implemented - requires custom deployment"


# 🎯 **USAGE EXAMPLE**
def main():
    """Main function to generate your character dataset"""
    
    # Get API token
    api_token = input("Enter your Replicate API token (or press Enter to use REPLICATE_API_TOKEN env var): ").strip()
    if not api_token:
        api_token = None  # Will use env var
    
    # Initialize generator
    generator = ReplicateCharacterGenerator(
        api_token=api_token,
        output_dir="ai_influencer_dataset"
    )
    
    # Check status
    status = generator.get_status()
    print(f"Status: {status}")
    
    # Generate dataset
    reference_image = input("Path to your reference character image: ").strip()
    
    if not os.path.exists(reference_image):
        print(f"❌ File not found: {reference_image}")
        return
    
    # Generate 80 images for LoRA training
    try:
        paths, metadata = generator.generate_character_dataset(
            reference_image_path=reference_image,
            num_images=80,
            cost_estimate=True
        )
        
        print(f"✅ Dataset generation complete!")
        print(f"📁 Check your output directory: {generator.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
