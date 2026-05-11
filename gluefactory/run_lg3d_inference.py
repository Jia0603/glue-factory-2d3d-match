import torch
from tqdm import tqdm
from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
import numpy as np
from omegaconf import OmegaConf

def run_nn_eval():

    conf = {
        "model": {
        "name": "lightglu3d_bicross",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": "/home/x_jiagu/glue-factory/outputs/training/fix_pad_nll.5_gaussian_0.05_all/checkpoint_best.tar",
        },
        "data": {
            "name": "mega_2d3d_dataset_soft_fix_pad",
            "root": "/proj/vlarsson/outputs",
            "split_val": "splits/val.txt", 
            "batch_size": 1,
            "num_workers": 4,
            "pos_reproj_thresh": 3.0,
            "neg_reproj_thresh": 8.0
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"filter_threshold:{conf['model']['filter_threshold']}")
    print(f"pos_reproj_thresh:{conf['data']['pos_reproj_thresh']}; neg_reproj_thresh:{conf['data']['neg_reproj_thresh']}.")

    dataset = get_dataset(conf["data"]["name"])(conf["data"])
    loader = dataset.get_data_loader("val")

    ModelClass = get_model(conf["model"]["name"])
    full_conf = OmegaConf.merge(ModelClass.default_conf, conf["model"])
    model = ModelClass(full_conf).to(device).eval()
    # model = get_model(conf["model"]["name"])(conf["model"]).to(device).eval()

    results = {
        "num_matches": [],
        "precision": [],
        "recall": []
    }

    print("Starting Evaluation...")
    for data in tqdm(loader):
        
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        with torch.no_grad():
            pred = model(data)

        matches = pred['matches0']
        gt_matches = data['gt_matches0']
        
        mask = ((matches > -1) & (gt_matches >= -1)).float()
        correct_matches = ((matches == gt_matches) * mask).sum(1)
        
        num_pred = (matches > -1).sum().item()
        recall_mask = (gt_matches > -1).float()
        precision = correct_matches / (1e-8 + mask.sum(1))
        recall =  ((matches == gt_matches) * recall_mask).sum(1) / (1e-8 + recall_mask.sum(1))
        
        results["num_matches"].append(num_pred)
        results["precision"].append(precision.item())
        results["recall"].append(recall.item())

    print("\n" + "="*30)
    print(f"lg3d Results:")
    print(f"Average Matches:   {np.mean(results['num_matches']):.2f}")
    print(f"Match Precision: {np.mean(results['precision']):.4f}")
    print(f"Match Recall:    {np.mean(results['recall']):.4f}")
    print("="*30)

if __name__ == "__main__":
    run_nn_eval()