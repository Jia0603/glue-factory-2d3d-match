import torch
from tqdm import tqdm
from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
import numpy as np

def run_nn_grid_search():

    data_conf = {
        "name": "mega_2d3d_dataset",
        "root": "/proj/vlarsson/outputs",
        "split_val": "splits/val.txt", 
        "batch_size": 1,
        "num_workers": 4,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Initializing Dataset...")
    dataset = get_dataset(data_conf["name"])(data_conf)
    loader = dataset.get_data_loader("val")

    # Define the grid
    mutual_checks = [True, False]
    ratio_thresholds = [None, 0.5, 0.7, 0.8, 0.9]
    distance_thresholds = [None, 0.5, 0.75, 1.0]

    all_results = []
    best_f1 = -1
    best_config = ""

    total_runs = len(mutual_checks) * len(ratio_thresholds) * len(distance_thresholds)
    current_run = 1

    print(f"\nStarting Grid Search: {total_runs} combinations to test.")

    # Iterate over all parameter combinations
    for mc in mutual_checks:
        for rt in ratio_thresholds:
            for dt in distance_thresholds:
                
                print(f"\nRun {current_run}/{total_runs} | Mutual: {mc} | Ratio: {rt} | Dist: {dt}...")
                
                # Initialize model
                model_conf = {
                    "name": "matchers.nearest_neighbor_matcher",
                    "mutual_check": mc,
                    "ratio_thresh": rt,
                    "distance_thresh": dt,
                }
                model = get_model(model_conf["name"])(model_conf).to(device).eval()

                scene_precisions = []
                scene_recalls = []
                scene_matches = []

                # Evaluate on the validation scenes
                for data in tqdm(loader, leave=False):
                    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    
                    with torch.no_grad():
                        pred = model(data)

                    matches = pred['matches0'][0] 
                    gt_matches = data['gt_matches0'][0] 
                    
                    valid_matches = (matches > -1)
                    correct_matches = (matches == gt_matches) & (gt_matches > -1)
                    
                    num_pred = valid_matches.sum().item()
                    num_gt = (gt_matches > -1).sum().item()
                    num_correct = correct_matches.sum().item()

                    precision = num_correct / num_pred if num_pred > 0 else 0
                    recall = num_correct / num_gt if num_gt > 0 else 0
                    
                    scene_precisions.append(precision)
                    scene_recalls.append(recall)
                    scene_matches.append(num_pred)

                # Calculate metrics
                avg_precision = np.mean(scene_precisions)
                avg_recall = np.mean(scene_recalls)
                avg_matches = np.mean(scene_matches)
                
                # Calculate F1 Score to balance precision and recall
                if (avg_precision + avg_recall) > 0:
                    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                else:
                    f1_score = 0.0

                print(f"Result -> Matches: {avg_matches:.1f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {f1_score:.4f}")

                # Save results
                config_str = f"Mutual:{str(mc)[0]} | Ratio:{rt} | Dist:{dt}"
                all_results.append((f1_score, avg_precision, avg_recall, avg_matches, config_str))

                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_config = config_str

                current_run += 1

    # Print Summary
    all_results.sort(reverse=True, key=lambda x: x[0]) # Sort by F1 Score descending

    print("\n" + "="*75)
    print("Grid Search Results")
    print("="*75)
    print(f"{'Config':<30} | {'F1':<6} | {'Prec':<6} | {'Recall':<6} | {'Matches':<6}")
    print("-" * 75)
    for res in all_results:
        print(f"{res[4]:<30} | {res[0]:.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.1f}")
    print("="*75)
    print(f"Best Config: {best_config}")
    print("="*75)

if __name__ == "__main__":
    run_nn_grid_search()