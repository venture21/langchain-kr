# To run this code you need to install the following dependencies:
# pip install google-genai gradio pillow

import os
import io
import gradio as gr
from google import genai
from google.genai import types
from PIL import Image


def generate_image(prompt):
    """Generate an image based on the prompt using Gemini API"""
    
    # Initialize the client
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    model = "gemini-2.0-flash-exp"
    
    # Prepare the contents with the user's prompt
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    # Configure generation to return both image and text
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )
    
    generated_images = []
    generated_text = []
    
    try:
        # Generate content
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or len(chunk.candidates) == 0
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
                or len(chunk.candidates[0].content.parts) == 0
            ):
                continue
            
            # Process each part in the response
            for part in chunk.candidates[0].content.parts:
                # Check for image data
                if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                    data_buffer = part.inline_data.data
                    
                    # Convert bytes to PIL Image
                    image = Image.open(io.BytesIO(data_buffer))
                    generated_images.append(image)
                
                # Check for text data
                elif hasattr(part, 'text') and part.text:
                    generated_text.append(part.text)
        
        # Return the first generated image and combined text
        if generated_images:
            text_response = " ".join(generated_text) if generated_text else "Image generated successfully!"
            return generated_images[0], text_response
        else:
            combined_text = " ".join(generated_text) if generated_text else "No image was generated."
            return None, combined_text
    
    except Exception as e:
        return None, f"Error generating image: {str(e)}"


def create_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(title="Gemini Image Generator") as demo:
        gr.Markdown("# 🎨 Gemini Image Generator")
        gr.Markdown("Generate images using Google's Gemini model")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="e.g., A cute cat wearing a space suit on Mars",
                    lines=3
                )
                
                generate_btn = gr.Button("🚀 Generate Image", variant="primary")
                
                # Examples
                gr.Examples(
                    examples=[
                        ["A banana wearing a costume"],
                        ["A futuristic city with flying cars at sunset"],
                        ["A cute robot playing guitar in a rock concert"],
                        ["A magical forest with glowing mushrooms and fairies"],
                        ["A steampunk-style coffee machine"],
                        ["A photorealistic portrait of a tiger in the jungle"],
                        ["An abstract painting in the style of Kandinsky"],
                    ],
                    inputs=prompt_input,
                    label="Example Prompts"
                )
            
            with gr.Column(scale=1):
                # Output section
                image_output = gr.Image(
                    label="Generated Image",
                    type="pil"
                )
                
                text_output = gr.Textbox(
                    label="Response",
                    lines=3,
                    interactive=False
                )
        
        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_image,
            inputs=prompt_input,
            outputs=[image_output, text_output]
        )
        
        # Also allow generating by pressing Enter in the prompt input
        prompt_input.submit(
            fn=generate_image,
            inputs=prompt_input,
            outputs=[image_output, text_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📝 Notes:
        - Make sure to set your `GEMINI_API_KEY` environment variable
        - The model used is `gemini-2.0-flash-exp` with image generation capabilities
        - Image generation may take a few moments
        - Some prompts may not generate images depending on content policies
        """)
    
    return demo


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️ Warning: GEMINI_API_KEY environment variable is not set!")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        print("Or in Windows: set GEMINI_API_KEY=your-api-key")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)