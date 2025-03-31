import os
from laionc.config import default_config, load_class_mappings
from laionc.data import get_dataloader
from laionc.evaluation import evaluate_model

def run_evaluation(
    model,
    batch_size=None,
    dataset_location=None,
    num_workers=None,
    augmentation_type=None,
    intensity_level=None
):
    config = default_config.copy()
    config.update({
        "batch_size": batch_size or config["batch_size"],
        "dataset_location": dataset_location or config["dataset_location"],
        "num_workers": num_workers or config["num_workers"]
    })
   ## add corruptions
    superclass_categories = load_class_mappings(
        config["index_class_path"], config["super_class_path"]
    )
    corruption_list = [augmentation_type] if augmentation_type else config["corruption_types"]
    intensity_list = [intensity_level] if intensity_level else [1, 2, 3, 4, 5]
    results = {}

    for corruption in corruption_list:
        for severity in intensity_list:
            data_path = os.path.join(config["dataset_location"], corruption, f"intensity_{severity}")
            try:
                dataloader = get_dataloader(
                    config["batch_size"],
                    config["num_workers"],
                    dataset_location=data_path
                )
                acc = evaluate_model(
                    model=model,
                    data_loader_val=dataloader,
                    superclass_categories=superclass_categories
                )
                print(acc)
                print(f"{corruption} | intensity_level {severity}: {acc*100:.2f}%")
                if corruption not in results:
                    results[corruption] = {}
                results[corruption][f"intensity_level_{severity}"] = acc

            except Exception as e:
                print(f"⚠️ Skipped {corruption} intensity level {severity} due to error: {e}")

    return results

