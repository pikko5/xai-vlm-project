from gradcam import gradCAM
from utils import normalize, getAttMap, viz_attn, load_image
import clip
import urllib.request
import torch
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate attention maps using Grad-CAM and CLIP.")
    parser.add_argument("--image_url", required=True, help="URL of the input image.")
    parser.add_argument("--image_caption", required=True, help="Caption for the image.")
    parser.add_argument("--clip_model", choices=["RN50", "RN101", "RN50x4", "RN50x16"], default="RN50", help="CLIP model to use.")
    parser.add_argument("--saliency_layer", choices=["layer4", "layer3", "layer2", "layer1"], default="layer4", help="Saliency layer to use.")
    parser.add_argument("--output_path", default="attention_map.png", help="Path to save the output PNG image.")
    parser.add_argument("--blur", action="store_true", help="Apply blurring to the attention map.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)

    image_path = 'image.png'
    urllib.request.urlretrieve(args.image_url, image_path)

    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_np = load_image(image_path, model.visual.input_resolution)
    text_input = clip.tokenize([args.image_caption]).to(device)

    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        getattr(model.visual, args.saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    viz_attn(image_np, attn_map, output_path=args.output_path, blur=args.blur)

if __name__ == "__main__":
    main()