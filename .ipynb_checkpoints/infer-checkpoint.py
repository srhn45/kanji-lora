#!/usr/bin/env python
# inference_lora.py

import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import argparse

def load_lora_model(lora_path, base_model="runwayml/stable-diffusion-v1-5", device=None):
    """
    Load base model with LoRA weights applied
    
    Args:
        lora_path: Path to the trained LoRA model directory
        base_model: Base Stable Diffusion model to use
        device: Device to load model on (cuda/cpu)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading base model: {base_model}")
    print(f"Using device: {device}")
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Check if LoRA path exists
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    
    print(f"Loading LoRA weights from: {lora_path}")
    
    # Load LoRA weights using PEFT
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    # Merge LoRA weights with base model for faster inference
    # Note: merge_and_unload() permanently merges LoRA weights into the model
    # If you want to keep them separate, remove this line
    pipe.unet = pipe.unet.merge_and_unload()
    
    return pipe

def generate_single_image(pipe, prompt, output_path=None, num_inference_steps=50, guidance_scale=7.5, height=256, width=256):
    """
    Generate a single image from a prompt
    
    Args:
        pipe: Loaded pipeline with LoRA
        prompt: Text prompt
        output_path: Path to save image (optional)
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        height: Image height
        width: Image width
    """
    print(f"Generating: '{prompt}'")
    
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Saved to: {output_path}")
    
    return image

def generate_batch(pipe, prompts, output_dir="./generated_kanji", num_inference_steps=50, guidance_scale=7.5):
    """
    Generate multiple images from a list of prompts
    
    Args:
        pipe: Loaded pipeline with LoRA
        prompts: List of text prompts
        output_dir: Directory to save images
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {len(prompts)} images to {output_dir}")
    
    generated_images = []
    for i, prompt in enumerate(prompts):
        # Create a safe filename
        safe_name = prompt[:50].replace(' ', '_').replace(',', '').replace('/', '_')
        filename = f"{i:03d}_{safe_name}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Generate image
        image = generate_single_image(
            pipe, 
            prompt, 
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        generated_images.append((prompt, image))
    
    print(f"\nGenerated {len(generated_images)} images in {output_dir}")
    return generated_images

def create_comparison_grid(images_with_prompts, output_path="comparison.png", cols=3):
    """
    Create a grid of generated images for comparison
    
    Args:
        images_with_prompts: List of tuples (prompt, image)
        output_path: Path to save grid image
        cols: Number of columns in grid
    """
    from PIL import ImageDraw, ImageFont
    
    images = [img for _, img in images_with_prompts]
    prompts = [prompt for prompt, _ in images_with_prompts]
    
    # Calculate grid dimensions
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    
    # Get image dimensions
    img_width, img_height = images[0].size
    
    # Create grid image
    grid_width = cols * img_width
    grid_height = rows * (img_height + 30)  # Extra space for text
    
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_image)
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Paste images and add text
    for idx, (image, prompt) in enumerate(zip(images, prompts)):
        row = idx // cols
        col = idx % cols
        
        x = col * img_width
        y = row * (img_height + 30)
        
        # Paste image
        grid_image.paste(image, (x, y))
        
        # Add prompt text
        text = prompt[:40] + "..." if len(prompt) > 40 else prompt
        draw.text((x + 5, y + img_height + 5), text, fill='black', font=font)
    
    # Save grid
    grid_image.save(output_path)
    print(f"Comparison grid saved to: {output_path}")
    return grid_image

def main():
    parser = argparse.ArgumentParser(description="Generate images using trained LoRA model")
    parser.add_argument("--lora_path", default="./lora-kanji/checkpoint-3", 
                       help="Path to trained LoRA model directory")
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5",
                       help="Base Stable Diffusion model")
    parser.add_argument("--prompt", default="intern",
                       help="Single prompt to generate")
    parser.add_argument("--prompts_file", 
                       help="Text file with one prompt per line (overrides --prompt)")
    parser.add_argument("--output_dir", default="./generated_kanji",
                       help="Directory to save generated images")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="CFG guidance scale")
    parser.add_argument("--height", type=int, default=256,
                       help="Image height")
    parser.add_argument("--width", type=int, default=256,
                       help="Image width")
    parser.add_argument("--batch_mode", action="store_true",
                       help="Generate batch of images from training prompts")
    parser.add_argument("--create_grid", action="store_true",
                       help="Create comparison grid of generated images")
    parser.add_argument("--device", 
                       help="Device to use (cuda/cpu), defaults to auto-detect")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with LoRA
    try:
        pipe = load_lora_model(args.lora_path, args.base_model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Determine prompts to generate
    prompts = ['internship']
    
    if args.prompts_file:
        # Read prompts from file
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.batch_mode:
        # Use prompts from your training data
        prompts = [
            "music, comfort, ease",
            "lottery, raffle",
            "straw raincoat",
            "bottom, base, slap, bang, all of a sudden",
            "guess, presume, surmise, judge, understand",
            "hawk",
            "feel ashamed",
            "flounder, flatfish",
            "dainty, get thin, taper, slender, narrow, detailed, precise",
            "crow, raven",
            "happening to meet",
            "red"
        ]
    else:
        # Single prompt
        prompts = [args.prompt]
    
    # Generate images
    generated = generate_batch(
        pipe, 
        prompts, 
        output_dir=args.output_dir,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale
    )
    
    # Create comparison grid if requested
    if args.create_grid and len(generated) > 1:
        grid_path = os.path.join(args.output_dir, "comparison_grid.png")
        create_comparison_grid(generated, grid_path)
    
    print("\n" + "="*50)
    print("Generation complete!")
    print(f"Images saved to: {args.output_dir}")
    print("="*50)

def interactive_mode():
    """Interactive mode for testing different prompts"""
    import sys
    
    print("Interactive LoRA Inference Mode")
    print("-" * 40)
    
    # Get LoRA path
    lora_path = input("Path to LoRA model [./lora-kanji/checkpoint-3]: ").strip()
    if not lora_path:
        lora_path = "./lora-kanji/checkpoint-3"
    
    # Load model
    try:
        pipe = load_lora_model(lora_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nModel loaded! Enter prompts to generate images.")
    print("Commands:")
    print("  :q - Quit")
    print("  :s N - Change steps (default: 50)")
    print("  :g N - Change guidance scale (default: 7.5)")
    print("  :d DIR - Change output directory")
    print("-" * 40)
    
    num_steps = 50
    guidance_scale = 7.5
    output_dir = "./interactive_output"
    os.makedirs(output_dir, exist_ok=True)
    
    counter = 1
    while True:
        user_input = input(f"\nPrompt [{counter}]: ").strip()
        
        if user_input.lower() == ':q':
            print("Goodbye!")
            break
        elif user_input.startswith(':s '):
            try:
                num_steps = int(user_input[3:])
                print(f"Steps set to: {num_steps}")
            except:
                print("Invalid steps value")
            continue
        elif user_input.startswith(':g '):
            try:
                guidance_scale = float(user_input[3:])
                print(f"Guidance scale set to: {guidance_scale}")
            except:
                print("Invalid guidance scale value")
            continue
        elif user_input.startswith(':d '):
            output_dir = user_input[3:]
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory set to: {output_dir}")
            continue
        
        if not user_input:
            print("Please enter a prompt")
            continue
        
        # Generate image
        filename = f"gen_{counter:03d}.png"
        output_path = os.path.join(output_dir, filename)
        
        try:
            image = generate_single_image(
                pipe,
                user_input,
                output_path=output_path,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            # Show image info
            print(f"Generated: {filename} ({image.size[0]}x{image.size[1]})")
            counter += 1
            
        except Exception as e:
            print(f"Error generating image: {e}")

if __name__ == "__main__":
    # You can run either main mode or interactive mode
    # Uncomment the one you want to use
    
    # Standard command-line mode
    main()
    
    # Interactive mode (uncomment to use)
    #interactive_mode()