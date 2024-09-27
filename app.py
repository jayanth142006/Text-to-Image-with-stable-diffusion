import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import customtkinter as ctk

# Set up the GUI using customtkinter
app = ctk.CTk()
app.geometry("400x240")
app.title("Stable Diffusion GUI")

# Determine if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
def load_pipeline():
    try:
        if device == "cuda":
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                revision="fp16",  # If you're using fp16 on GPU
                torch_dtype=torch.float16
            ).to(device)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32  # Use float32 for CPU
            ).to(device)
        
        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

pipe = load_pipeline()

# Function to generate images
def generate():
    if pipe:
        prompt = entry.get()  # Get the text input from the GUI
        try:
            # Autocast is only necessary if you're using a GPU
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    image = pipe(prompt).images[0]
            else:
                image = pipe(prompt).images[0]

            # Save the image
            image.save("output.png")
            print("Image generated and saved as output.png")
        except Exception as e:
            print(f"Error during image generation: {e}")
    else:
        print("Pipeline is not loaded correctly.")

# Setting up the GUI layout
label = ctk.CTkLabel(app, text="Enter a prompt for Stable Diffusion:")
label.pack(pady=20)

entry = ctk.CTkEntry(app, width=300)
entry.pack(pady=10)

generate_button = ctk.CTkButton(app, text="Generate", command=generate)
generate_button.pack(pady=20)

# Start the application
app.mainloop()

