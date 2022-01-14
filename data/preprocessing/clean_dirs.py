import tqdm
import os

root = "/om2/user/yyf/CommonFate/scenes/"
texs = ["noise"]

for tex in texs:
    for i in range(4, 5):
        split = f"superquadric_{i}"
        path = os.path.join(root, tex, split)
        scenes = os.listdir(path)
        print("Processing scenes for ", path)
        for scene in tqdm.tqdm(scenes):
            scene_path = os.path.join(path, scene)
            scene_path_dirs = os.listdir(scene_path)
            if len(scene_path_dirs) < 1:
                print("Deleting scene: ", scene_path)
                os.rmdir(scene_path)
