import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from tqdm import tqdm
import argparse

class KanjiDataset(Dataset):
    def __init__(self, metadata_path, image_folder, tokenizer, size=256):
        self.tokenizer = tokenizer
        self.size = size
        
        # Load metadata
        self.data = []
        with open(metadata_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_folder, item["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Tokenize text
        text = item["text"]
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0]
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.jsonl")
    parser.add_argument("--image_folder", default=".")
    parser.add_argument("--output_dir", default="./lora-kanji")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load model
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,  # Use float32 to avoid dtype issues
        safety_checker=None
    ).to(args.device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )
    
    # Add LoRA to UNet
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()
    
    # Freeze other components
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    # Create dataset
    dataset = KanjiDataset(args.metadata, args.image_folder, pipe.tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr)
    
    # Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.epochs * len(dataloader)
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    
    # Training loop
    for epoch in range(args.epochs):
        pipe.unet.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Get batch
            pixel_values = batch["pixel_values"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            
            # Encode images
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=args.device
            ).long()
            
            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            with torch.no_grad():
                encoder_hidden_states = pipe.text_encoder(input_ids)[0]
            
            # Predict noise
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            pipe.unet.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")
    
    # Save final model
    pipe.unet.save_pretrained(args.output_dir)
    print(f"Final model saved to {args.output_dir}")
    
    # Test generation
    print("\nTesting generation...")
    pipe.unet.eval()
    with torch.no_grad():
        for prompt in ["music, "serhan"]:
            image = pipe(prompt, num_inference_steps=20).images[0]
            image.save(os.path.join(args.output_dir, f"test_{prompt[:10]}.png"))
            print(f"Generated image for: {prompt}")

if __name__ == "__main__":
    main()