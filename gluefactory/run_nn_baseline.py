import torch
from tqdm import tqdm
from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
import numpy as np

def run_nn_eval():

    conf = {
        "model": {
            "name": "matchers.nearest_neighbor_matcher",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
        },
        "data": {
            "name": "mega_2d3d_dataset",
            "root": "/proj/vlarsson/outputs",
            "split_val": "splits/val.txt", 
            "batch_size": 1,
            "num_workers": 4,
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = get_dataset(conf["data"]["name"])(conf["data"])
    loader = dataset.get_data_loader("val")

    model = get_model(conf["model"]["name"])(conf["model"]).to(device).eval()

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

        matches = pred['matches0'][0] # [N]
        gt_matches = data['gt_matches0'][0] # [N]
        
        valid_matches = (matches > -1)
        correct_matches = (matches == gt_matches) & (gt_matches > -1)
        
        num_pred = valid_matches.sum().item()
        num_gt = (gt_matches > -1).sum().item()
        num_correct = correct_matches.sum().item()

        precision = num_correct / num_pred if num_pred > 0 else 0
        recall = num_correct / num_gt if num_gt > 0 else 0
        

        results["num_matches"].append(num_pred)
        results["precision"].append(precision)
        results["recall"].append(recall)

    print("\n" + "="*30)
    print(f"NN Baseline Results (Ratio={conf['model']['ratio_threshold']}):")
    print(f"Average Matches:   {np.mean(results['num_matches']):.2f}")
    print(f"Average Precision: {np.mean(results['precision']):.4f}")
    print(f"Average Recall:    {np.mean(results['recall']):.4f}")
    print("="*30)

if __name__ == "__main__":
    run_nn_eval()