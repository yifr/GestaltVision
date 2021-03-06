import os
import sys
from glob import glob
from PIL import Image
import numpy as np
import h5py as hp
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="current dataset location")
parser.add_argument("--top_level", type=str, help="top level directory (for hierarchical dataset)")
parser.add_argument("--sub_level", type=str, help="top level directory (for hierarchical dataset)")
parser.add_argument("--output_dir", type=str, help="Root dataset directory")
parser.add_argument("--output_name", type=str, help="Name of the output hdf5 file")
parser.add_argument("--image_passes", type=str, nargs="+", help="Image passes to use")
parser.add_argument("--resize", type=int, default=128, help="resize images to size")
parser.add_argument("--modify", action="store_true", help="whether to write new or modify existing dataset")
args = parser.parse_args()

def gestalt_to_hdf5():
    root_dir = os.path.join(args.output_dir, args.top_level)
    if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    new_dataset_path = os.path.join(root_dir, args.sub_level + ".hdf5")
    if os.path.exists(new_dataset_path) and args.modify:
        mode = "a"
        print("Adding additional data to existing file", new_dataset_path)
    else:
        mode = "w"
        print("Creating new dataset: ", new_dataset_path)

    logfile_path = os.path.join(args.data_dir, args.top_level, args.sub_level + "_log.txt")
    with open(logfile_path, "w") as logfile:
        with hp.File(new_dataset_path, mode, libver="latest") as h5_data:
            data_path_pattern = os.path.join(args.data_dir, args.top_level, "*" + args.sub_level)
            data_dir = glob(data_path_pattern)[0]
            files = os.listdir(data_dir)
            files = sorted(files)
            print(f"Loading {len(files)} files from {args.top_level}/{args.sub_level}")

            for i, f in tqdm(enumerate(files)):
                scene = f"scene_{i:03d}"
                log_text = scene + "\n\t"
                if h5_data.get(scene):
                    scene_group = h5_data[scene]
                else:
                    scene_group = h5_data.create_group(f"scene_{i:03d}")
                print(f)
                for image_pass in args.image_passes:
                    print(f"Loading {image_pass}...")
                    pass_path = os.path.join(data_dir, f, image_pass)
                    if not os.path.exists(pass_path):
                        print("No images found at", pass_path)
                        continue

                    if len(os.listdir(pass_path)) == 0:
                        print("No images found at", pass_path)
                        continue

                    if scene_group.get(image_pass):
                        print("Data already exists for image pass", image_pass)
                        log_text += image_pass + "\t"
                        continue

                    image_names = glob(os.path.join(pass_path, "Image*.png"))
                    image_paths = sorted(image_names)
                    images = []
                    for image_path in image_paths:
                        img = Image.open(image_path).convert("RGB")
                        if image_pass == "masks":
                            resample = Image.NEAREST
                        else:
                            resample = Image.BICUBIC
                        img = img.resize((args.resize, args.resize), resample=resample)
                        img = np.array(img).astype(np.uint8)
                        images.append(img)

                    images = np.stack(images, axis=0)
                    print(f"Pass: {image_pass}, Shape: {images.shape}")

                    pass_data = scene_group.create_dataset(image_pass, images.shape)
                    pass_data[...] = images
                    log_text += image_pass + "\t"

                logfile.write(log_text + "\n")

            logfile.write("\nDONE\n")


def clevr_to_hdf5():
    dataset_root = os.path.join(args.root_dir, "CLEVR_v1.0", "images")

    fname = os.path.join(args.root_dir, args.dataset_name + ".hdf5")
    with hp.File(fname, "w") as f:

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



if __name__ == "__main__":
    gestalt_to_hdf5()
