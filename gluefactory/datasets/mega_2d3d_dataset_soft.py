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
    Change in soft threshold version.
    """

    def __init__(self, samples, root, pos_reproj_thresh=3.0, neg_reproj_thresh=5.0, 
                 pos_depth_thresh=0.1, neg_depth_thresh=0.25, max_2d=2048, max_3d=8192):

        self.samples = samples
        self.root = Path(root)
        
        # New Soft Thresholds
        self.pos_reproj_thresh = pos_reproj_thresh
        self.neg_reproj_thresh = neg_reproj_thresh
        self.pos_depth_thresh = pos_depth_thresh
        self.neg_depth_thresh = neg_depth_thresh
        
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
        depth_file = scene_info["depth_path"] / f"{Path(query).stem}.h5"
        depth_map = load_depth(depth_file) if depth_file.exists() else None
        
        # Call the new vectorized soft threshold function
        matches0, matches1 = self.compute_ground_truth_matches(
            query_feats, p3d_feats, camera, depth_map
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
    
    def compute_ground_truth_matches(self, query_feats, p3d_feats, camera, depth_map=None):
        """
        Vectorized PyTorch implementation of GT matching with Soft Thresholds.
        """

        IGNORE_FEATURE = -2
        UNMATCHED_FEATURE = -1

        kpts2d = torch.from_numpy(query_feats["keypoints"]).float() # (N2D, 2)
        pts3d = torch.from_numpy(p3d_feats["keypoints"]).float()    # (N3D, 3)

        N2D, N3D = kpts2d.shape[0], pts3d.shape[0]

        # Initialize with -1
        matches0 = torch.full((N2D,), UNMATCHED_FEATURE, dtype=torch.long)
        matches1 = torch.full((N3D,), UNMATCHED_FEATURE, dtype=torch.long)

        if N3D == 0:
            return matches0.numpy(), matches1.numpy()

        # Pose and intrinsics
        R = torch.from_numpy(qvec2rotmat(camera["qvec"])).float()
        t = torch.tensor(camera["tvec"]).float().view(3, 1)

        fx, fy, cx, cy = camera["intrinsics"]["params"][:4]
        width, height = camera["intrinsics"]["width"], camera["intrinsics"]["height"]

        # Vectorized 3D projection
        X = pts3d.T # (3, N3D)
        X_cam = R @ X + t # (3, N3D)
        
        z = X_cam[2, :]
        valid_z = z > 0
        
        # Avoid division by zero for invalid z
        z_safe = torch.where(valid_z, z, torch.ones_like(z))
        
        u = fx * (X_cam[0, :] / z_safe) + cx
        v = fy * (X_cam[1, :] / z_safe) + cy

        valid_proj = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        # Depth check arrays
        has_valid_depth = torch.zeros(N3D, dtype=torch.bool)
        rel_error = torch.full((N3D,), float('inf'))

        if depth_map is not None:
            # Only sample depth for points that landed inside the image bounds
            u_np, v_np = u[valid_proj].numpy(), v[valid_proj].numpy()
            
            if len(u_np) > 0:
                depth_real = sample_depth_bilinear(depth_map, u_np, v_np)
                depth_real_t = torch.from_numpy(depth_real).float()
                
                valid_d = depth_real_t > 0
                
                z_valid = z[valid_proj]
                rel_err_valid = torch.full_like(depth_real_t, float('inf'))
                rel_err_valid[valid_d] = torch.abs(z_valid[valid_d] - depth_real_t[valid_d]) / depth_real_t[valid_d]
                
                # Scatter back to the original N3D sized arrays
                rel_error[valid_proj] = rel_err_valid
                has_valid_depth[valid_proj] = valid_d

        # Combine u,v into (N3D, 2)
        projected = torch.stack([u, v], dim=1)
        
        # Vectorized Distance Matrix Calculation
        dist_matrix = torch.cdist(projected.unsqueeze(0), kpts2d.unsqueeze(0)).squeeze(0) # -> (N3D, N2D)
        
        # Mask out points that projected behind the camera or off-screen
        dist_matrix[~valid_proj] = float('inf')

        # Get minimum distance to a 2D point for every 3D point
        min_dists, min_indices = torch.min(dist_matrix, dim=1) # (N3D,)

        # Filter only points that passed the loose negative threshold
        valid_mask = min_dists <= self.neg_reproj_thresh
        valid_3d_indices = torch.where(valid_mask)[0]
        
        # Loop over only the valid subset to prevent assignment collisions
        for idx3d in valid_3d_indices:
            min_idx_2d = min_indices[idx3d]
            dist = min_dists[idx3d]
            r_err = rel_error[idx3d]
            valid_d = has_valid_depth[idx3d]

            if valid_d:
                # STRICT MATCH
                if dist <= self.pos_reproj_thresh and r_err <= self.pos_depth_thresh:
                    if matches0[min_idx_2d] in [UNMATCHED_FEATURE, IGNORE_FEATURE]:
                        matches0[min_idx_2d] = idx3d
                        matches1[idx3d] = min_idx_2d
                # IGNORE MATCH
                elif dist <= self.neg_reproj_thresh and r_err <= self.neg_depth_thresh:
                    if matches0[min_idx_2d] == UNMATCHED_FEATURE:
                        matches0[min_idx_2d] = IGNORE_FEATURE
                    if matches1[idx3d] == UNMATCHED_FEATURE:
                        matches1[idx3d] = IGNORE_FEATURE
            else:
                # MISSING DEPTH -> IGNORE MATCH
                if dist <= self.neg_reproj_thresh:
                    if matches0[min_idx_2d] == UNMATCHED_FEATURE:
                        matches0[min_idx_2d] = IGNORE_FEATURE
                    if matches1[idx3d] == UNMATCHED_FEATURE:
                        matches1[idx3d] = IGNORE_FEATURE

        return matches0.numpy(), matches1.numpy()
    
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
        # Updated to dual thresholds
        "pos_reproj_thresh": 3.0,
        "neg_reproj_thresh": 5.0,
        "pos_depth_thresh": 0.10,
        "neg_depth_thresh": 0.25,
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
        self.pos_reproj_thresh = conf.pos_reproj_thresh
        self.neg_reproj_thresh = conf.neg_reproj_thresh
        self.pos_depth_thresh = conf.pos_depth_thresh
        self.neg_depth_thresh = conf.neg_depth_thresh
        
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
                
        return Torch2D3D(
            samples, self.root, 
            pos_reproj_thresh=self.pos_reproj_thresh, 
            neg_reproj_thresh=self.neg_reproj_thresh,
            pos_depth_thresh=self.pos_depth_thresh,
            neg_depth_thresh=self.neg_depth_thresh
        )
    
    def get_dataset(self, split):
        if split == "train":
            return self.train_dataset
        elif split == "val":
            return self.val_dataset
        elif split == "test":
            return self.test_dataset
        else:
            raise ValueError(f"Unknown split {split}")