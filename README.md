# XAI VLM Research Project 

Explainable AI Project for VLM

## How to Use

We use gradcam on CLIP. 

First, install all requirements:

```
pip install -r requirements.txt
```

Then on the `main.py` file, run:

```
python main.py --image_url "your_image_url" --image_caption "your image caption" --clip_model "RN50" --saliency_layer "layer4" --output_path "my_output.png" --blur
```

Each of these arguments are:
1. `image_url`: URL to the image you would like to use.
2. `image_caption`: Captions for the image.
3. `clip_model`: Types of CLIP Models. Only supports ResNET CLIP Models Right now. More features to come up soon.

Arguments for `clip_model`:

`["RN50", "RN101", "RN50x4", "RN50x16"]`

4. `saliency_layer`: the layer which we perform GradCAM. 

Arguments for `saliency_layer`:

`["layer4", "layer3", "layer2", "layer1"]`

5. `output_path`: image output
6. `blur`: apply Gaussian blur to input image.


## Feature Checklist

### Research

- [ ] Scouring for Codebase: Still WIP. 

### Engineering
- [x] Scouring for Compute Resource: We'll use GPU this time via colab + vast.ai.
- [ ] Add FastAPI integration for better serving into Cloud VMs.
- [ ] Dockerize the code for better deployment into Cloud VMs.