import torch
from tqdm import tqdm
from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
import numpy as np
from omegaconf import OmegaConf

def run_nn_eval():

    conf = {
        "model": {
            "name": "matchers.nearest_neighbor_matcher",
            "mutual_check": True,
            "ratio_thresh": None,
            "distance_thresh": 0.75,
        },
        "data": {
            "name": "mega_2d3d_dataset_soft",
            "root": "/proj/vlarsson/outputs",
            "split_val": "splits/test.txt", 
            "batch_size": 1,
            "num_workers": 4,
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = get_dataset(conf["data"]["name"])(conf["data"])
    loader = dataset.get_data_loader("val")

    ModelClass = get_model(conf["model"]["name"])
    full_conf = OmegaConf.merge(ModelClass.default_conf, conf["model"])
    model = ModelClass(full_conf).to(device).eval()
    print(f"Verified Threshold: {model.conf.ratio_thresh}")
    # model = get_model(conf["model"]["name"])(conf["model"]).to(device).eval()

    results = {
        "num_matches": [],
        "precision": [],
        "recall": []
    }

    print("Starting NN Evaluation...")
    for data in tqdm(loader):
        
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        with torch.no_grad():
            pred = model(data)

        # matches = pred['matches0'][0] # [N]
        # gt_matches = data['gt_matches0'][0] # [N]
        matches = pred['matches0']
        gt_matches = data['gt_matches0']
        
        mask = ((matches > -1) & (gt_matches >= -1)).float()
        correct_matches = ((matches == gt_matches) * mask).sum(1)
        # correct_matches = (matches == gt_matches) & (gt_matches > -1)
        
        num_pred = (matches > -1).sum().item()
        # num_gt = (gt_matches > -1).sum().item()
        # num_correct = correct_matches.sum().item()

        # precision = num_correct / (1e-8 + num_pred)
        # recall = num_correct / (1e-8 + num_gt)

        recall_mask = (gt_matches > -1).float()
        precision = correct_matches / (1e-8 + mask.sum(1))
        recall =  ((matches == gt_matches) * recall_mask).sum(1) / (1e-8 + recall_mask.sum(1))
        
        results["num_matches"].append(num_pred)
        results["precision"].append(precision)
        results["recall"].append(recall)

    print("\n" + "="*30)
    print(f"NN Baseline Results:")
    print(f"Average Matches:   {np.mean(results['num_matches']):.2f}")
    print(f"Match Precision: {np.mean(results['precision']):.4f}")
    print(f"Match Recall:    {np.mean(results['recall']):.4f}")
    print("="*30)

if __name__ == "__main__":
    run_nn_eval()