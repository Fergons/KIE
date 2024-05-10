from data_processing_layoutxlm import id2ner, ner2id, ner_tags, label_names, label_to_ner
from data_processing_layoutxlm import normalize_bbox, unnormalize_bbox
from data_processing_layoutxlm import load_and_process_alto, preprocess_data
from data_processing_layoutxlm import DataCollatorForTokenClassification
import pandas as pd
from pathlib import Path
import datasets
import numpy as np
import os 
from transformers import LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, LayoutLMv2ForTokenClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch


OUTPUT_DIR = Path("layoutxlm_train_size_experiments")
DATA_DIR = Path("data")
DATASET = DATA_DIR / "VHA"
DATASET_ALTO = DATASET / "alto"
DATASET_IMAGES = DATASET / "images"
TEST_FILE_ALTO = DATASET_ALTO / "1.xml"
TEST_FILE_IMAGE = DATASET_IMAGES / "1.jpg"
DATASET_CSV = list(DATASET.glob("*.csv"))[0]
dataset = pd.read_csv(DATASET_CSV)


metric = datasets.load_metric("seqeval")
return_entity_level_metrics = True

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2ner[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2ner[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    




def run_experiment(run_name, train: pd.DataFrame, test_dataset: datasets.Dataset, train_size=100, tokenizer=None, data_collator=None, random_state=99):
    train_df, _= train_test_split(train, train_size=train_size, random_state=random_state)
    train_dataset = datasets.Dataset.from_generator(
        preprocess_data,
          gen_kwargs={ "examples": train_df,
                       "dataset_dir": DATASET,
                       "image_dir": DATASET_IMAGES,
                       "alto_dir": DATASET_ALTO,
                       "tokenizer": tokenizer,
                       "image_size": (224, 224)})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutxlm-base",
                                                             id2label=id2ner,
                                                            label2id=ner2id).to(device)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR/run_name,
        report_to="wandb",
        run_name=run_name,
        warmup_ratio=0.1, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-6,
        remove_unused_columns=False,
        load_best_model_at_end=False,
        save_strategy="no",
        evaluation_strategy="steps",
        logging_steps=125,
        save_steps=250,
        max_steps=250
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()


def main():
    os.environ["WANDB_PROJECT"]="knn_kie"
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    data_collator = DataCollatorForTokenClassification(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=None,
        padding="max_length",
        max_length=512,
    )

    labeled_dataset = pd.read_csv(DATASET_CSV)
    labeled_dataset["file"] = labeled_dataset["ocr"].apply(lambda x: x.rsplit("-")[-1])
    train_df, test_df = train_test_split(labeled_dataset, train_size=0.8, random_state=99)
    test_dataset = datasets.Dataset.from_generator(
        preprocess_data,
        gen_kwargs={ "examples": test_df,
                     "dataset_dir": DATASET,
                     "image_dir": DATASET_IMAGES,
                     "alto_dir": DATASET_ALTO,
                     "tokenizer": tokenizer,
                     "image_size": (224, 224)})
    # 20 pseudo random integers in list
    random_states = [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    runs = 5
    for i,tn in enumerate([20, 50, 100, 200]):
        for j in range(runs):
            run_experiment(f"layoutxlm_run_{j}_train_size_{tn}",
                            train_df, test_dataset,
                            train_size=tn,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            random_state=random_states[i*runs+j])

if __name__ == "__main__":
    main()