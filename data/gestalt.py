import os
import h5py as h5
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from glob import glob

IMAGE_PASS_OPTS = ["images", "masks", "depths", "flows", "normals"]
STATE_PASS_OPTS = ["location", "center_of_mass", "shape_params", "bounding_box"]

class Gestalt(Dataset):
    """
    Data Loader for gestalt data

    Args:
        root_dir: str: Root data directory
        top_level: List[str]: Top level hierarchy to pull data from
                    Current options: ["voronoi", "noise", "wave"]
        n_objs: List[str]: Second level hierarchy (how many objects. Will pull data from any
                            sub-directory ending in listed string
                    Current options: [1, 2, 3, 4]
        passes: List[str]: Which image passes to load
                    Image pass options: ["images", "flows", "masks", "depths", "normals"]
                    Config pass options: ["shape_params", "location_2d" (same as center of mass), "location_3d", "rotation"]
        frames_per_scene: int: How many frames to sample per scene
        frame_sampling: str: How to sample frames
                    Current options: "even" (sample at regular interval),
                                    "consecutive" (will sample blocks of consecutive frames until scene ends)
        resolution: Tuple[int]: resolution to resize images to
        transforms: Dict[str:List[transforms]]: Dictionary mapping pass name to list of any additional transforms to apply
        training: Bool: whether to supply training or testing data
        train_test_split: float: percentage of scenes to allocate to testing set
        random_seed: int: random seed for train/test split
        color_channels: str: "RGB" or "L" for black and white
    """
    def __init__(self,
                 root_dir="/om2/user/yyf/CommonFate/scenes/",
                 top_level=["voronoi", "noise"],
                 sub_level=["1", "2", "3"],
                 passes=["images", "masks", "flows"],
                 frames_per_scene=10,
                 frame_sampling_method="consecutive",
                 resolution=(128,128),
                 transforms={},
                 training=True,
                 train_test_split=.2,
                 random_seed=42,
                 color_channels="RGB"
                ):

        self.root_dir = root_dir
        if self.root_dir.endswith(".hdf5"):
            self.use_h5 = True
        else:
            self.use_h5 = False

        self.top_level = top_level
        self.sub_level = sub_level
        self.passes = passes
        self.frames_per_scene = frames_per_scene
        self.frame_sampling_method = frame_sampling_method
        self.resolution = resolution
        self.transforms = transforms
        self.train_test_split = train_test_split
        self.random_seed = random_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training = training
        self.color_channels = color_channels

        self.train_scenes, self.test_scenes = self.get_scene_paths()
        self.real_frames_per_scene = len(glob(os.path.join(self.train_scenes[0], "images", "*.png")))
        if self.frame_sampling_method == "consecutive":
            self.scene_splits = int(self.real_frames_per_scene / self.frames_per_scene)
        else:
            self.scene_splits = 1

    def get_scene_paths(self):
        train_scenes = []
        test_scenes = []
        for top in self.top_level:
            for sub in self.sub_level:
                data_path_pattern = os.path.join(self.root_dir, top, "*" + sub)
                data_dir = glob(data_path_pattern)[0]
                files = os.listdir(data_dir)
                files = sorted(files)
                # split scenes into train and test
                num_test = int(len(files) * self.train_test_split)

                np.random.seed(self.random_seed)
                test_idxs = sorted(np.random.choice(range(len(files)), num_test, replace=False))
                test_idx = 0
                for i, f in enumerate(files):
                    fpath = os.path.join(data_dir, f)
                    if test_idx >= len(test_idxs) or i != test_idxs[test_idx]:
                        train_scenes.append(fpath)
                    else:
                        test_scenes.append(fpath)
                        test_idx += 1

        return train_scenes, test_scenes

    def get_scenes(self, scene_idx=-1):
        if self.training:
            scenes = self.train_scenes
        else:
            scenes = self.test_scenes

        if scene_idx > -1:
            return scenes[scene_idx]
        else:
            return scenes

    def load_image_data(self, image_pass, scene_idx, frame_idxs, color_channels="RGB", normalize=True):
        if self.use_h5:
            # TODO
            with open(self.root_dir, "r") as f:
                data = h5.File(f, mode="r")
                data.close()
                return
        else:
            scene = self.get_scenes(scene_idx)
            pass_dir = os.path.join(scene, image_pass)
            paths = [os.path.join(pass_dir, f"Image{idx:04d}.png") for idx in frame_idxs]
            images = []
            for path in paths:
                img = Image.open(path).convert(color_channels)
                if image_pass == "masks":
                    resample = Image.NEAREST
                else:
                    resample = Image.BICUBIC
                img = img.resize(self.resolution, resample=resample)
                img = np.array(img).astype(np.uint8)
                if len(img.shape) < 3:
                    img = img[..., np.newaxis]  # add 1 channel for depths / masks
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                images.append(img)

            images = torch.stack(images, dim=0)
            if normalize:
                images = images / 255.0
            return images.to(self.device)

    def load_config_data(self, config_pass, scene, frame_idxs):
        scene = self.get_scene(scene_idx)
        config_path = os.path.join(scene, "scene_config.pkl")
        with open(config_path, "rb") as f:
            config_data = pickle.load(f)["objects"]

        data = []
        for obj in config_data:
            obj_params = config_data[obj]
            if obj_params.get("child_params"): # ignore parent shapes
                continue

            if config_pass == "shape_params":
                data.append(obj_params["shape_params"])
            elif config_pass == "center_of_mass":
                config_pass = "location"
            elif config_pass == "bounding_box":
                # TODO
                pass
            else:
                config_data = obj_params[config_pass][frame_idxs]
                data.append(config_data)

        data = torch.from_numpy(np.array(data))
        return data



    def __len__(self):
        scenes = self.get_scenes()
        length = len(scenes) * self.frames_per_scene * self.scene_splits
        return length

    def get_info(self, idx):
        scene = int(idx / self.scene_splits)
        if self.frame_sampling_method == "consecutive":
            scene_block = idx % self.scene_splits
            start_frame = self.frames_per_scene * scene_block
            frame_idxs = np.arange(start_frame + 1, start_frame + self.frames_per_scene + 1)
        elif self.frame_sampling_method == "even":
            interval = int(self.real_frames_per_scene / self.frames_per_scene)
            if interval * self.frames_per_scene > self.real_frames_per_scene:
                print(f"Cannot use even sampling method for {self.frames_per_scene} frames in a scene \
                        with {self.real_frames_per_scene} total frames")
                raise ValueError
            frame_idxs = np.arange(1, self.real_frames_per_scene + 1, interval)
        else:
            frame_idxs = np.arange(1, self.real_frames_per_scene + 1)

        scene_dir = self.get_scenes(scene)
        return scene, frame_idxs, scene_dir

    def __getitem__(self, idx):
        scene, frame_idxs, scene_dir = self.get_info(idx)

        # Load image passes
        data = {}
        for image_pass in self.passes:
            res = []
            if image_pass in IMAGE_PASS_OPTS:
                color_channels = "RGB"
                normalize = True

                res = self.load_image_data(image_pass, scene, frame_idxs, color_channels, normalize)
            elif image_pass in STATE_PASS_OPTS:
                res = self.load_config_data(image_pass, scene, frame_idxs)
            else:
                print(f"Unknown pass specified: {_pass}")
                raise ValueError

            data[image_pass] = res

        return data

