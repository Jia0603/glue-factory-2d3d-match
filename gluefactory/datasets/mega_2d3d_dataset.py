import torch
import numpy as np
import h5py
import pickle
import random
from pathlib import Path
from torch.utils.data import Dataset
from gluefactory.datasets.base_dataset import BaseDataset
import torch
import torch.nn.functional as F

def load_scene_list(split_file):
    with open(split_file, "r") as f:
        scenes = [l.strip() for l in f if len(l.strip()) > 0]
    return scenes

def load_query_names(query_names_path):
    with open(query_names_path, "r") as f:
        return [l.strip() for l in f]

def load_query_cams(query_pose_path):

    query_pose_dict = {}
    with open(query_pose_path, 'r') as f:
        for line in f:
            item = line.strip().split()
            name = item[0]

            qvec = list(map(float, item[1:5]))      # 4 numbers
            tvec = list(map(float, item[5:8]))      # 3 numbers

            camera_id = item[8]
            model = item[9]
            width = int(item[10])
            height = int(item[11])
            params = list(map(float, item[12:]))

            query_pose_dict[name] = {
                "qvec": qvec,
                "tvec": tvec,
                "intrinsics": {
                    "camera_id": camera_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
            }
    return query_pose_dict

def load_depth(depth_path):
    with h5py.File(depth_path, 'r') as f:
        depth = f['depth'][:]
    return depth
def sample_depth_bilinear(depth_map, u, v):
    """
    depth_map: (H, W)
    u, v: arrays (N,)
    return: depth values (N,)
    """

    H, W = depth_map.shape

    # normalize to [-1,1] for grid_sample
    u_norm = 2.0 * u / (W - 1) - 1.0
    v_norm = 2.0 * v / (H - 1) - 1.0

    grid = torch.from_numpy(
        np.stack([u_norm, v_norm], axis=-1)
    ).float().unsqueeze(0).unsqueeze(0)  # (1,1,N,2)

    depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)

    sampled = F.grid_sample(
        depth_tensor,
        grid,
        align_corners=True,
        mode='bilinear'
    )

    return sampled.squeeze().numpy()
def qvec2rotmat(qvec):
    """Convert quaternion vector to rotation matrix.
    Args:
        qvec: Quaternion vector (4,).
    Returns:
        Rotation matrix (3, 3).
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    return R


class Torch2D3D(Dataset):
    """
    2D-3D dataset for LightGlue training.
    Each sample = one query image and its visible 3D points.
    """

    def __init__(self, samples, root, reproj_thresh=3.0, depth_rel_thresh=0.1, max_2d=2048, max_3d=8192):

        self.samples = samples
        self.root = Path(root)
        self.reproj_thresh = reproj_thresh
        self.depth_rel_thresh = depth_rel_thresh
        self.max_3d = max_3d
        self.max_2d = max_2d

        # Preload scene info to reduce repeated file I/O
        self.scene_data = {}
        for scene in set(s[0] for s in samples):
            query_path = self.root / "query_sets" / scene
            query_pose_file = query_path / "query_image_cameras.txt"

            feats_2d_path = self.root / "sfm" / scene / "feats-superpoint-n2048.h5"
            feats_3d_path = (self.root / "midterm_results" / scene / "points3D_feats_cache.h5")
            covis_path = self.root / "midterm_results" / scene / "covisibility_results.pkl"
            depth_path = self.root.parent / "datasets/megadepth/depth_undistorted" / scene

            query_cams = load_query_cams(query_pose_file)

            with open(covis_path, "rb") as f:
                covis_dict = pickle.load(f)

            # store per-scene static data
            self.scene_data[scene] = {
                "query_cams": query_cams,
                "covis": covis_dict,
                "feats_2d_path": feats_2d_path,
                "feats_3d_path": feats_3d_path,
                "depth_path": depth_path,
                }

        print(f"Loaded {len(self.samples)} samples.")


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        while True:
            scene, query = self.samples[idx]
            scene_info = self.scene_data[scene]
            # load 2D features
            query_feats = self.load_query_features(query, scene_info["feats_2d_path"])
            # load visible 3D points and the averaged features
            visible_p3d = scene_info["covis"][query]["unique_points"]
            p3d_feats = self.load_3d_features(visible_p3d, scene_info["feats_3d_path"])

            if p3d_feats["keypoints"].shape[0] > 0:
                # if the 3D set is not empty
                break
            idx = random.randint(0, len(self) - 1)

        camera = scene_info["query_cams"][query]
        depth_map = load_depth(scene_info["depth_path"] / f"{Path(query).stem}.h5")
        matches0, matches1 = self.compute_ground_truth_matches(
            query_feats, p3d_feats, camera, depth_map, self.reproj_thresh, self.depth_rel_thresh,
        )
        width = camera['intrinsics']['width']
        height = camera['intrinsics']['height']
        size0 = torch.tensor([width, height]).float()
        size1 = torch.tensor([1.0, 1.0, 1.0]).float()
        kpts3d = torch.from_numpy(p3d_feats["keypoints"]).float()
        desc3d = torch.from_numpy(p3d_feats["descriptors"]).float()
        matches1 = torch.from_numpy(matches1).long()
        kpts2d = torch.from_numpy(query_feats["keypoints"]).float()
        desc2d = torch.from_numpy(query_feats["descriptors"].T).float()
        matches0 = torch.from_numpy(matches0).long()

        # padding
        kpts3d, desc3d, mask1, matches1 = self.pad_to_max(
            kpts3d, desc3d, matches1, self.max_3d
        )
        kpts2d, desc2d, mask0, matches0 = self.pad_to_max(
            kpts2d, desc2d, matches0, self.max_2d
        )

        return {
            "keypoints0": kpts2d,
            "descriptors0": desc2d,
            "keypoints1": kpts3d,
            "descriptors1": desc3d,
            "mask0": mask0,
            "mask1": mask1,
            "gt_matches0": matches0,
            "gt_matches1": matches1,
            "size0": size0,
            "size1": size1,
        }
    
    def pad_to_max(self, kpts, desc, matches, maxN):
        """
        Pad inputs to maxN
        """

        N = kpts.shape[0]
        d = kpts.shape[1]
        D = desc.shape[1]

        if N > maxN:
            kpts = kpts[:maxN]
            desc = desc[:maxN]
            matches = matches[:maxN]
            N = maxN

        kpts_pad = torch.zeros(maxN, d)
        desc_pad = torch.zeros(maxN, D)
        mask = torch.zeros(maxN, dtype=torch.bool)
        matches_pad = torch.full((maxN,), -1, dtype=torch.long)

        kpts_pad[:N] = kpts
        desc_pad[:N] = desc
        mask[:N] = True
        matches_pad[:N] = matches

        return kpts_pad, desc_pad, mask, matches_pad
    def compute_ground_truth_matches(
            self, query_feats, p3d_feats, camera, depth_map=None, reproj_thresh=3.0, depth_rel_thresh=0.1
            ):
        """
        return:
            matches0: (N2D,)
            matches1: (N3D,)
        """

        kpts2d = query_feats["keypoints"]      # (N2D, 2)
        pts3d = p3d_feats["keypoints"]       # (N3D, 3)

        N2D = kpts2d.shape[0]
        N3D = pts3d.shape[0]

        matches0 = -np.ones(N2D, dtype=int)
        matches1 = -np.ones(N3D, dtype=int)

        # pose
        R = qvec2rotmat(camera["qvec"])
        t = np.array(camera["tvec"]).reshape(3, 1)

        # intrinsics
        fx, fy, cx, cy = camera["intrinsics"]["params"][:4]# assuming PINHOLE: fx fy cx cy
        width = camera["intrinsics"]["width"]
        height = camera["intrinsics"]["height"]

        # project all 3D points
        X = pts3d.T  # (3, N3D)
        X_cam = R @ X + t  # (3, N3D)

        z = X_cam[2]
        valid = z > 0 # check depth > 0

        X_cam = X_cam[:, valid]
        z = z[valid]

        u = fx * (X_cam[0] / z) + cx
        v = fy * (X_cam[1] / z) + cy

        valid_proj = ((u >= 0) & (u < width) & (v >= 0) & (v < height))

        u = u[valid_proj]
        v = v[valid_proj]    
        z = z[valid_proj]

        valid_indices = np.where(valid)[0][valid_proj]

        # check depth consistency
        if depth_map is not None:

            depth_real = sample_depth_bilinear(depth_map, u, v)

            valid_depth = depth_real > 0
            rel_error = np.full_like(depth_real, np.inf)
            rel_error[valid_depth] = (
                np.abs(z[valid_depth] - depth_real[valid_depth])
                / depth_real[valid_depth]
            ).flatten()

            depth_mask = (rel_error <= depth_rel_thresh)

            u = u[depth_mask]
            v = v[depth_mask]
            z = z[depth_mask]
            valid_indices = valid_indices[depth_mask]

        projected = np.stack([u, v], axis=1)

        # find the nearest 2D keypoint
        for idx3d, proj_pt in zip(valid_indices, projected):

            dists = np.linalg.norm(kpts2d - proj_pt, axis=1)
            min_idx = np.argmin(dists)

            if dists[min_idx] < reproj_thresh:

                if matches0[min_idx] == -1: # in case to rewrite, only register the first matched pair
                    matches0[min_idx] = idx3d
                    matches1[idx3d] = min_idx

        return matches0, matches1
    
    def load_3d_features(self, points3d, h5_path):

        point3d_feature_dict = {}
        descriptors =[]
        keypoints = []
        scores = []
        with h5py.File(h5_path, "r") as f_h5:
            for id in points3d:
                id = str(id)
                if id not in f_h5:
                    print(f"WARNING: {id} not found in {h5_path}")
                    continue
                ds = f_h5[id]
                descriptors.append(ds["descriptors"][:].reshape(1,256))
                keypoints.append(ds["keypoints"][:].reshape(1,3))
                scores.append(ds["scores"][:])
                
        if len(descriptors) == 0:
            point3d_feature_dict["descriptors"] = np.zeros((0, 256), dtype=np.float32)
            point3d_feature_dict["keypoints"] = np.zeros((0, 3), dtype=np.float32)
            point3d_feature_dict["scores"] = np.array([], dtype=np.float32)
        else:
            point3d_feature_dict["descriptors"] = np.vstack(descriptors)
            point3d_feature_dict["scores"] = scores
            point3d_feature_dict["keypoints"] = np.vstack(keypoints)

        # print(f"Collected descriptors for {np.shape(point3d_feature_dict['keypoints'])[0]} 3D points.")

        return point3d_feature_dict
           
    def load_query_features(self, img_name, h5_path):

        query_feature_dict = {}
        with h5py.File(h5_path, "r") as f_h5:

            if img_name not in f_h5:
                print(f"WARNING: {img_name} not found in {h5_path}")
                return query_feature_dict
            ds = f_h5[img_name]

            query_feature_dict["descriptors"] = ds["descriptors"][:]
            query_feature_dict["scores"] = ds["scores"][:]
            query_feature_dict["keypoints"] = ds["keypoints"][:]

        # print(f"Collected descriptors for query image {img_name}.")
        
        return query_feature_dict

    
class MegaDepth2D3D(BaseDataset):
    """
    BaseDataset wrapper for GlueFactory.
    Manages train/val/test split and data loader creation.
    """

    default_conf = {
        "batch_size": 1,
        "reproj_thresh": 3.0,
        "depth_rel_thresh": 0.1,
        "root": "/proj/vlarsson/outputs",
        "split_train": "splits/train.txt",
        "split_val": "splits/val.txt",
        "split_test": "splits/test.txt",
        "train_list": None,
        "val_list": None,
        "test_list": None,
    }

    def _init(self, conf):
        self.root = conf.root
        self.reproj_thresh = conf.reproj_thresh
        self.depth_rel_thresh = conf.depth_rel_thresh
        # Build torch datasets
        self.train_dataset = self._build_dataset(conf.split_train)
        self.val_dataset = self._build_dataset(conf.split_val)
        self.test_dataset = self._build_dataset(conf.split_test)

    def _build_dataset(self, split_file):
        scenes = load_scene_list(Path(self.root) / split_file)
        samples = []
        for scene in scenes:
            query_path = Path(self.root) / "query_sets" / scene
            query_names_file = query_path / "query_image_names_clean.txt"
            if not query_names_file.exists():
                continue
            queries = load_query_names(query_names_file)
            for q in queries:
                samples.append((scene, q))
        return Torch2D3D(samples, self.root, self.reproj_thresh, self.depth_rel_thresh)
    
    def get_dataset(self, split):
        if split == "train":
            return self.train_dataset
        elif split == "val":
            return self.val_dataset
        elif split == "test":
            return self.test_dataset
        else:
            raise ValueError(f"Unknown split {split}")
        


