import os
import torch
from collections import defaultdict

def evaluate_model(model, data_loader_val, device="cuda" if torch.cuda.is_available() else "cpu", superclass_categories=None):
    model = model.to(device)
    model.eval()
    correct_predictions=0
    total_images=0
    with torch.no_grad():
        for batch in data_loader_val:
            images, ground_truth,paths = batch
            images = images.to(device)
            ground_truth = ground_truth.to(device)
            output = model(images)
            probabilities = output.softmax(dim=1)
            _, predicted_class_ids = torch.topk(probabilities, k=1)
            batch_probs = probabilities.cpu().numpy()
            total_images += images.size(0)
            
            for i in range(batch_probs.shape[0]):  # Loop through each item in the batch
                batch_valid_pairs = []
                valid_pairs = [(cls, probs) for cls,probs in  zip(superclass_categories,batch_probs[i]) if cls is not None]
                probs_by_class = defaultdict(list)
                for cls, probs in valid_pairs:
                    probs_by_class[cls].append(probs)
                avg_probs = {cls: sum(probs)/len(probs) for cls, probs in probs_by_class.items()}
                max_class=max(avg_probs, key=avg_probs.get)
                ground_truth=os.path.basename(os.path.dirname(paths[i]))
                top1_correct=(max_class == ground_truth)
                correct_predictions += top1_correct
    accuracy = correct_predictions / total_images
    return accuracy
