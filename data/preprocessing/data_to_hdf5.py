import os
import sys
from PIL import Image
import numpy as np
import h5py as hp
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="Root dataset directory")
parser.add_argument("--dataset_name", type=str, help="Name of the output hdf5 file")

args = parser.parse_args()


def clevr_to_hdf5():
    dataset_root = os.path.join(args.root_dir, "CLEVR_v1.0", "images")

    fname = os.path.join(args.root_dir, args.dataset_name + ".hdf5")
    f = hp.File(fname, "w")

    group = f.create_group("images")
    data_splits = ["train", "test", "val"]

    for data_split in data_splits:
        data_path = os.path.join(dataset_root, data_split)
        image_names = os.listdir(data_path)
        print(f"{len(image_names)} images in {data_split} split")
        sys.stdout.flush()

        dataset_shape = (len(image_names), 3, 256, 256)
        dataset = group.create_dataset(data_split, dataset_shape)

        for i, img_name in tqdm.tqdm(enumerate(image_names)):
            img_path = os.path.join(data_path, img_name)
            img = np.asarray(Image.open(img_path).convert("RGB").resize((256, 256)))
            img = np.transpose(img, [2, 0, 1]) / 255
            dataset[i] = img

        # print(f"{data_split} dataset statistics: ")
        # print("Channel Mean: ", dataset.mean(axis=(0, 2, 3)))
        # print("Channel Std: ", dataset.std(axis=(0, 2, 3)))

    f.close()


if __name__ == "__main__":
    clevr_to_hdf5()
