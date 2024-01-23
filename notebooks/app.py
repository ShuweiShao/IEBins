"""
  This is a Hugging Face Spaces demo for IEBins: Iterative Elastic Bins for Monocular Depth Estimation.
  Please refer to the [paper](https://arxiv.org/abs/2309.14137), [github](https://github.com/ShuweiShao/IEBins),
  or [poster](https://nips.cc/media/PosterPDFs/NeurIPS%202023/70695.png?t=1701662442.5228624) for more details.
  This demo is heavily based on [LiheYoung/Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything).
  I'm learning Gradio and this is my first Space. I'm still working on improving the performance of this demo.
  Any suggestions are welcome!

  Author: Umut (Hope) YILDIRIM <hope@umutyildirim.com>
"""

import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import spaces
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
import tempfile
from gradio_imageslider import ImageSlider
import matplotlib.pyplot as plt

from iebins.networks.NewCRFDepth import NewCRFDepth
from iebins.util.transfrom import Resize, NormalizeImage, PrepareForNet
from iebins.utils import post_process_depth, flip_lr

css = """
#img-display-container {
    max-height: 100vh;
    }
#img-display-input {
    max-height: 80vh;
    }
#img-display-output {
    max-height: 80vh;
    }
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NewCRFDepth(version='large07', inv_depth=False,
                    max_depth=10, pretrained=None).to(DEVICE).eval()
model.train()
num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("== Total number of parameters: {}".format(num_params))
num_params_update = sum([np.prod(p.shape)
                        for p in model.parameters() if p.requires_grad])
print("== Total number of learning parameters: {}".format(num_params_update))

model = torch.nn.DataParallel(model)
checkpoint = torch.load('checkpoints/nyu_L.pth',
                        map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model'])
print("== Loaded checkpoint '{}'".format('checkpoints/nyu_L.pth'))

title = "# IEBins: Iterative Elastic Bins for Monocular Depth Estimation"
description = """Demo for **IEBins: Iterative Elastic Bins for Monocular Depth Estimation**.
Please refer to the [paper](https://arxiv.org/abs/2309.14137), [github](https://github.com/ShuweiShao/IEBins), or [poster](https://nips.cc/media/PosterPDFs/NeurIPS%202023/70695.png?t=1701662442.5228624) for more details."""
acknowledgement = """This demo is heavily based on [LiheYoung/Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything). I'm learning Gradio and this is my first Space. I'm still working on improving the performance of this demo. Any suggestions are welcome!"""

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])


@spaces.GPU
@torch.no_grad()
def predict_depth(model, image):
    return model(image)


with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        input_image = gr.Image(label="Input Image",
                               type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(
            label="Depth Map with Slider View", elem_id='img-display-output', position=0.5,)
    raw_file = gr.File(label="Download Depth Map")
    submit = gr.Button("Submit")

    def on_submit(image):
        original_image = image.copy()

        # Resize the image
        image = cv2.resize(image, (640, 480))

        # Normalize the image
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])(image)

        with torch.no_grad():
            image = torch.autograd.Variable(image.unsqueeze(0))
            print("== Processing image")
            pred_depths_r_list, _, _ = model(image)
            image_flipped = flip_lr(image)
            pred_depths_r_list_flipped, _, _ = model(image_flipped)
            pred_depth = post_process_depth(
                pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])
            print("== Finished processing image")

            # Convert the PyTorch tensor to a NumPy array and squeeze
            pred_depth = pred_depth.cpu().numpy().squeeze()

            # Continue with your file saving operations
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.imsave(tmp.name, pred_depth, cmap='jet')

            return [(original_image, tmp.name), tmp.name]

    submit.click(on_submit, inputs=[input_image], outputs=[
                 depth_image_slider, raw_file])

    example_files = os.listdir('examples')
    example_files.sort()
    example_files = [os.path.join('examples', filename)
                     for filename in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[
                           depth_image_slider, raw_file], fn=on_submit, cache_examples=False)
    gr.Markdown(acknowledgement)


if __name__ == '__main__':
    demo.queue().launch()
