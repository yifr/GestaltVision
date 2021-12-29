import os
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import numpy as np

def make_video(sequence, titles=[], output_name=None, output_dir="figures", format="mp4", fps=8):
    """
    Takes an array or tensor of images (T x C x W x H) and converts it into
    an mp4 or gif, which gets saved to an output directory

    Params:
        sequence: list or torch.tensor: input sequence
        output_name: str: name of output video / gif (without the extension)
        output_dir: str: directory to output file
        format: str: format to write video (either "gif" or "mp4")
        fps: int: frames per second

    """

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    columns = 1

    if type(sequence) == list:
        columns = len(sequence)
        sequence = np.array(sequence)
    elif type(sequence) == torch.tensor:
        sequence = sequence.detach().cpu().numpy()

    fig, ax = plt.subplots(1, columns, figsize=(12, 8))

    if sequence.shape[1] == 3 or sequence.shape[1] == 1:
        sequence = sequence.transpose(0, 2, 3, 1)

    imgs = []
    if columns > 1:
        for column in range(columns):
            im = ax[column].imshow(sequence[0][0])
            imgs.append(im)
        if len(titles) == columns:
            for column in range(columns):
                ax[column].set_title(titles[column])
    else:
        im = ax.imshow(sequence[0])

    def init():
        if columns > 1:
            for column in range(columns):
                imgs[column].set_array(sequence[0][0])
            return imgs
        else:
            im.set_array(sequence[0])
            return [im]

    def update(frame):
        if columns > 1:
            for column in range(columns):
                imgs[column].set_array(sequence[column][frame])
            return imgs
        else:
            im.set_array(sequence[frame])
            return [im]
        return [im]

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=sequence.shape[0])

    if format == "gif":
        writer = "imagemagick"
    elif format == "mp4":
        writer = "ffmpeg"

    if output_dir:
        ani.save(os.path.join(output_dir, output_name + "." + format), writer=writer, fps=fps)
    return fig, ani
