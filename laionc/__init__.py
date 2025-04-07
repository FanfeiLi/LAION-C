import os
from laionc.config import default_config, load_class_mappings
from laionc.data import get_dataloader
from laionc.evaluation import evaluate_model

def run_evaluation(
    model,
    batch_size=None,
    dataset_location=None,
    num_workers=None,
    model_name="Model",
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
        # Validate corruption type
    if augmentation_type:
        if augmentation_type not in config["corruption_types"]:
            raise ValueError(
                f"Invalid augmentation_type '{augmentation_type}'. "
                f"Must be one of: {config['corruption_types']}"
            )
        corruption_list = [augmentation_type]
    else:
        corruption_list = config["corruption_types"]

    superclass_categories = load_class_mappings(
        config["index_class_path"], config["super_class_path"]
    )
    valid_intensities = [1, 2, 3, 4, 5]
    if intensity_level:
        if intensity_level not in valid_intensities:
            raise ValueError(
                f"Invalid intensity_level '{intensity_level}'. "
                f"Must be one of: {valid_intensities}"
            )
        intensity_list = [intensity_level]
    else:
        intensity_list = valid_intensities
    results = {}
   
    for corruption in corruption_list:
        for severity in intensity_list:
            data_path = os.path.join(config["dataset_location"], corruption, f"intensity_level_{severity}")
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
                print(f"{model_name} | {corruption} | severity {severity}: {acc*100:.2f}%")
                #results[f"{corruption}_severity_{severity}"] = acc
                if corruption not in results:
                    results[corruption] = {}
                results[corruption][f"severity_{severity}"] = acc

            except Exception as e:
                print(f"⚠️ Skipped {corruption} intensity level {severity} due to error: {e}")

    return results





