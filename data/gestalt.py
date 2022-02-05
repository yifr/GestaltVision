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
STATE_PASS_OPTS = ["location", "center_of_mass", "shape_params", "bounding_boxes"]


class Gestalt(Dataset):
    """
    Data Loader for gestalt data

    Args:
        root_dir: str: Root data directory
        top_level: List[str]: Top level hierarchy to pull data from
                    Current options: ["voronoi", "noise", "wave"]
        sub_level: List[str]: Second level hierarchy (how many objects. Will pull data from any
                            sub-directory ending in listed string
                    Current options: [1, 2, 3, 4]
        passes: List[str]: Which image passes to load
                    Image pass options: ["images", "flows", "masks", "depths", "normals"]
                    Config pass options: ["shape_params", "location_2d" (same as center of mass), "location_3d", "rotation"]
        max_objects: int: Maximum number of objects to appear in a scene
        frames_per_scene: int: How many frames to sample per scene
        frame_sampling: str: How to sample frames
                    Current options: "even" (sample at regular interval),
                                    "consecutive" (will sample blocks of consecutive frames until scene ends)
        resolution: Tuple[int]: resolution to resize images to
        transforms: Dict[str:List[transforms]]: Dictionary mapping pass name to list of any additional transforms to apply
        training: Bool: whether to supply training or testing data
        train_split: float: percentage of scenes to allocate to training set
        random_seed: int: random seed for train/test split
        color_channels: str: "RGB" or "L" for black and white
        masks_as_rgb: bool: if True returns masks in RGB format, otherwise returns masks in black and white
        shuffle: whether or not to shuffle loaded data
    """

    def __init__(
        self,
        root_dir="/om/user/yyf/CommonFate/scenes/",
        top_level=["voronoi", "noise"],
        sub_level=["superquadric_2", "superquadric_3"],
        passes=["images", "masks", "flows"],
        max_num_objects=5,
        frames_per_scene=6,
        frame_sampling_method="consecutive",
        resolution=(128, 128),
        transforms={},
        training=True,
        train_split=0.9,
        random_seed=42,
        color_channels="RGB",
        masks_as_rgb=True,
        shuffle=False
    ):

        self.root_dir = root_dir
        if self.root_dir.endswith(".hdf5"):
            self.use_h5 = True
        else:
            self.use_h5 = False

        self.top_level = top_level
        self.sub_level = sub_level
        self.passes = passes
        self.max_num_objects = max_num_objects
        self.frames_per_scene = frames_per_scene
        self.frame_sampling_method = frame_sampling_method
        self.resolution = resolution
        self.transforms = transforms
        self.train_split = train_split
        self.random_seed = random_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training = training
        self.color_channels = color_channels
        self.shuffle = shuffle
        self.masks_as_rgb = masks_as_rgb

        self.train_scenes, self.test_scenes = self.get_scene_paths()
        print(f"{len(self.train_scenes)} # training scenes, {len(self.test_scenes)} # test scenes")

        if self.use_h5:
            with h5.File(self.root_dir, "r", swmr=True, libver="latest") as f:
                if len(self.train_scenes) > 1:
                    scene = f[self.train_scenes[0]]
                else:
                    scene = f[self.test_scenes[0]]

                images = scene["images"]["images"][:]
                self.real_frames_per_scene = images.shape[0]
        else:
            if len(self.train_scenes) > 1:
                scene = self.train_scenes[0]
            else:
                scene = self.test_scenes[0]

            self.real_frames_per_scene = len(
                glob(os.path.join(scene, "images", "*.png"))
            )
        if self.frame_sampling_method == "consecutive":
            self.scene_splits = int(self.real_frames_per_scene / self.frames_per_scene)
        else:
            self.scene_splits = 1

    def list_files(self, top_level, sub_level):
        if self.use_h5:
            with h5.File(self.root_dir, "r", swmr=True, libver="latest") as f:
                keys = list(f[top_level][sub_level].keys())
                return [f"{top_level}/{sub_level}/{key}" for key in keys]
        else:
            data_path_pattern = os.path.join(self.root_dir, top_level, sub_level)
            if not os.path.exists(data_path_pattern):
                raise ValueError(f"Could not find data at path: {data_path_pattern}")

            data_dir = glob(data_path_pattern)[0]
            files = os.listdir(data_dir)
            return [os.path.join(data_dir, f) for f in files]

    def get_scene_paths(self):
        train_scenes = []
        test_scenes = []
        for top in self.top_level:
            for sub in self.sub_level:
                files = self.list_files(top, sub)
                files = sorted(files)
                # split scenes into train and test
                num_test = int(len(files) * (1 - self.train_split))

                np.random.seed(self.random_seed)
                test_idxs = sorted(
                    np.random.choice(range(len(files)), num_test, replace=False)
                )
                test_idx = 0
                for i, f in enumerate(files):
                    if test_idx >= len(test_idxs) or i != test_idxs[test_idx]:
                        train_scenes.append(f)
                    else:
                        test_scenes.append(f)
                        test_idx += 1

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(train_scenes)
            np.random.seed(self.random_seed)
            np.random.shuffle(test_scenes)

        return train_scenes, test_scenes

    def get_scenes(self, scene_idx=-1):
        """ Returns path info for a given scene

            Params:
            -------
            scene_index: int
                if < 0, return all scene path info
                otherwise, return specified index
        """
        if self.training:
            scenes = self.train_scenes
        else:
            scenes = self.test_scenes

        if scene_idx > -1:
            return scenes[scene_idx]
        else:
            return scenes

    def load_image_data(
        self, image_pass, scene_idx, frame_idxs, color_channels="RGB", normalize=True
    ):
        scene = self.get_scenes(scene_idx)
        images = []
        for idx in frame_idxs:
            if self.use_h5:
                with h5.File(self.root_dir, "r", swmr=True, libver="latest") as f:
                    # Access image_pass twice bc of weird hdf5 external link thing
                    img = f[scene][image_pass][image_pass][idx]
            else:
                pass_dir = os.path.join(scene, image_pass)
                path = os.path.join(pass_dir, f"Image{idx:04d}.png")
                if image_pass == "masks":
                    resample = Image.NEAREST
                    if not self.masks_as_rgb:
                        img = Image.open(path).convert("L")
                    else:
                        img = Image.open(path).convert(color_channels)
                else:
                    img = Image.open(path).convert(color_channels)
                    resample = Image.BICUBIC

                img = img.resize(self.resolution, resample=resample)
                img = np.array(img).astype(np.uint8)

            if len(img.shape) < 3:
                img = img[..., np.newaxis]  # add 1 channel for depths / masks
            img = torch.from_numpy(img).permute(2, 0, 1).float()

            # Unwrap masks so each object has its own mask slot
            if image_pass == "masks":
                unique_masks = []
                masks = img.unique()
                for i in range(self.max_num_objects):
                    if i >= len(masks):
                        m = torch.zeros_like(img)
                    else:
                        mask = masks[i]
                        m = torch.where(img == mask, img, torch.zeros_like(img))
                        mask_idxs = m > 0
                        m[mask_idxs] = 255.0
                    unique_masks.append(m)
                img = torch.stack(unique_masks, 1)

            images.append(img)

        images = torch.stack(images, dim=0)

        if normalize:
            images = images / 255.0

        if image_pass == "masks":
            images = images.permute(2, 0, 1, 3, 4) # N_Objects x T x C x H x W

        return images

    def load_config_data(self, config_pass, scene, frame_idxs):
        scene = self.get_scenes(scene)
        if config_pass == "bounding_boxes":
            target = {}
            bounding_boxes = []
            areas = []
            all_masks = []
            for frame_idx in frame_idxs:
                mask_path = os.path.join(scene, "masks", f"Image{frame_idx:04d}.png")
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize(self.resolution, resample=Image.NEAREST)

                mask = np.array(mask)
                obj_ids = np.unique(mask)
                obj_ids = obj_ids[1:] # remove background

                masks = mask == obj_ids[:, None, None]
                num_objs = len(obj_ids)
                boxes = []

                for obj_id in obj_ids:
                    pos = np.where(mask == obj_id)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    area = (ymax - ymin) * (xmax - xmin)
                    if area < 1 or xmin >= xmax or ymin >= ymax:
                        print(mask_path, "INCORRECT MASKS")
                        print("area", area, "coords", ((xmin, ymin), (xmax, ymax)))
                        if xmin >= xmax:
                            xmax += 1
                        if ymin >= ymax:
                            ymax += 1

                    boxes.append([xmin, ymin, xmax, ymax])

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                masks = torch.as_tensor(masks, dtype=torch.uint8)

                if boxes.shape[0] == 0:
                    print("returning empty boxes for scene", mask_path)
                    return {"masks": torch.tensor([]),
                            "boxes": torch.tensor([]),
                            "labels": torch.tensor([])}

                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                bounding_boxes.append(boxes)
                areas.append(area)
                all_masks.append(masks)

            bounding_boxes = torch.stack(bounding_boxes, dim=0)
            areas = torch.stack(areas, dim=0)
            masks = torch.stack(all_masks, dim=0)
            labels = torch.ones((masks.shape[1], ), dtype=torch.int64)
            iscrowd = torch.zeros((bounding_boxes.shape[1], ), dtype=torch.int64)

            if labels is None:
                print(mask_path)

            targets = {"boxes": bounding_boxes, "areas": areas,
                    "labels": labels,
                    "iscrowd": iscrowd,
                    }
            return targets

        config_path = os.path.join(scene, "scene_config.pkl")
        with open(config_path, "rb") as f:
            config_data = pickle.load(f)["objects"]

        data = []
        for obj in config_data:
            obj_params = config_data[obj]
            if obj_params.get("child_params"):  # ignore parent shapes
                continue

            if config_pass == "shape_params":
                data.append(obj_params["shape_params"])
            elif config_pass == "center_of_mass":
                config_pass = "location"
            else:
                config_data = obj_params[config_pass][frame_idxs]
                data.append(config_data)

        data = torch.from_numpy(np.array(data))
        return data.to(self.device)

    def __len__(self):
        if self.training:
            length = len(self.train_scenes) * self.scene_splits
        else:
            length = len(self.test_scenes) * self.scene_splits
        return length

    def get_info(self, idx):
        scene = idx // self.scene_splits

        if self.frame_sampling_method == "consecutive":
            scene_block = idx % self.scene_splits
            start_frame = self.frames_per_scene * scene_block
            frame_idxs = np.arange(
                start_frame + 1, start_frame + self.frames_per_scene + 1
            )
        elif self.frame_sampling_method == "even":
            interval = int(self.real_frames_per_scene / self.frames_per_scene)
            if interval * self.frames_per_scene > self.real_frames_per_scene:
                print(
                    f"Cannot use even sampling method for {self.frames_per_scene} frames in a scene \
                        with {self.real_frames_per_scene} total frames"
                )
                raise ValueError
            frame_idxs = np.arange(1, self.real_frames_per_scene + 1, interval)
        else:
            frame_idxs = np.arange(1, self.real_frames_per_scene + 1)

        scene_dir = self.get_scenes(scene)
        return scene, frame_idxs, scene_dir

    def __getitem__(self, idx):
        scene, frame_idxs, scene_dir = self.get_info(idx)
        data = {}

        data["scene"] = scene
        data["frame_idxs"] = frame_idxs
        data["scene_dir"] = scene_dir

        # Load image passes
        for image_pass in self.passes:
            res = []
            if image_pass in IMAGE_PASS_OPTS:
                color_channels = "RGB"
                normalize = True

                # Flows are zero indexed, everything else is 1 indexed
                if image_pass == "flows":
                    frame_idxs = frame_idxs - 1
                    if frame_idxs[-1] == (self.real_frames_per_scene - 1):
                        frame_idxs[-1] = frame_idxs[-2]

                res = self.load_image_data(
                    image_pass, scene, frame_idxs, color_channels, normalize
                )
            elif image_pass in STATE_PASS_OPTS:
                res = self.load_config_data(image_pass, scene, frame_idxs)
                if image_pass == "bounding_boxes":
                    for key in res:
                        data[key] = res[key]
                    continue
            else:
                print(f"Unknown pass specified: {image_pass}")
                raise ValueError

            data[image_pass] = res


        return data

class MaskRCNNLoader(Dataset):
    def __init__(self, root_dir="/om2/user/yyf/CommonFate/scenes",
                 passes=["images", "bounding_boxes"],
                 top_level=["test_voronoi", "test_noise", "test_wave"],
                 sub_level=[f"superquadric_{i}" for i in range(1, 5)],
                 frames_per_scene=64,
                 training=True,
                 resolution=(128, 128)):
        super(Dataset, self).__init__()

        self.data = Gestalt(root_dir=root_dir,
                            passes=passes,
                            top_level=top_level,
                            sub_level=sub_level,
                            frames_per_scene=frames_per_scene,
                            training=training,
                            resolution=resolution,
                            masks_as_rgb=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data.__getitem__(idx)
        images = batch["images"]
        targets = batch

        return images, targets

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data = DataLoader(
        Gestalt(
            root_dir="/om2/user/yyf/CommonFate/scenes/",
            passes=["images", "flows", "depths", "masks"],
            top_level=["test_voronoi", "test_noise", "test_wave"],
            sub_level=["superquadric_1", "superquadric_2", "superquadric_3", "superquadric_4"],
            frames_per_scene=64
        ),
        batch_size=1,
    )
    print(len(data))
    batch = next(iter(data))
    print(batch.keys())
    print(
        f"images: %s, masks: %s, flows: %s, depths: %s"
        % (
            batch["images"].shape,
            batch["masks"].shape,
            batch["flows"].shape,
            batch["depths"].shape,
        )
    )
    print(
        f"image mean: %f, mask mean: %f, flow mean: %f, depth mean: %f"
        % (
            batch["images"].mean(),
            batch["masks"].mean(),
            batch["flows"].mean(),
            batch["depths"].mean(),
        )
    )
    print("len dataset: ", len(data))
    data.dataset.training = False
    print("len test: ", len(data))

