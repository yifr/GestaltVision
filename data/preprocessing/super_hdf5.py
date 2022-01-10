import os
import h5py

base_dir = "/om2/user/yyf/CommonFate/scenes/"

with h5py.File(base_dir + "gestalt.hdf5", "w", libver="latest") as f:
    for group_name in ["voronoi", "noise"]:
        group = f.require_group(group_name)
        for i in range(1, 5):
            dataset_path = os.path.join(group_name, f"superquadric_{i}.hdf5")
            group[f"superquadric_{i}"] = h5py.ExternalLink(dataset_path, "/")


    print(f["voronoi/superquadric_2/scene_1232/images"]["images"])
    print(f["noise/superquadric_2/scene_1232/images"]["images"])
    print(f["voronoi/superquadric_3/scene_1232/images"]["images"])
    print(f["noise/superquadric_3/scene_1232/images"]["images"])
