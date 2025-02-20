"""Main module."""
import argparse
import json
from pathlib import Path

import pandas as pd
import wandb
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, \
    TrainingArguments, Trainer, Wav2Vec2CTCTokenizer
from datasets import Audio, load_dataset, load_metric
import torch
import torchaudio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


prsr = argparse.ArgumentParser(
    description="""This script implements Fine-Tuning of wav2vec2.0 for ASR"""
)

"""
Script args
"""

prsr.add_argument("--data_dir",
                  help="Location of the Data directory",
                  )

prsr.add_argument("--data_name",
                  help="Name of directory containing audio files",
                  default="vercellotti")

prsr.add_argument("--output_dir",
                  help="Location of the output directory",
                 )

prsr.add_argument("--use_cuda",
                  help="Whether to use GPU or not",
                  default="True",
                  type=str2bool)

prsr.add_argument("--batch_size",
                  default=1,
                  type=int)

prsr.add_argument("--finetune",
                  help="Whether to train model on new data or not",
                  default="True",
                  type=str2bool)

"""
Model Args
"""

prsr.add_argument("--num_epochs",
                  default=10,
                  type=int,)

prsr.add_argument("--adam_beta1",
                  type=float,
                  default=0.88)
prsr.add_argument("--adam_beta2",
                  type=float,
                  default=0.98)
prsr.add_argument("--adam_epsilon",
                  type=float,
                  default=1e-8)
prsr.add_argument("--attention_dropout",
                  type=float,
                  default=0.1)
prsr.add_argument("--feat_proj_dropout",
                  type=float,
                  default=0.0)
prsr.add_argument("--gradient_accumulation_steps",
                  type=int,
                  default=2)
prsr.add_argument("--hidden_dropout",
                  type=float,
                  default=0.1)
prsr.add_argument("--layerdrop",
                  type=float,
                  default=0.1)
prsr.add_argument("--warmup_steps",
                  type=int,
                  default=1500)
prsr.add_argument("--weight_decay",
                  type=float,
                  default=0.0)


args = prsr.parse_args()

chars_to_ignore_regex = "[\,\?\.\!\-\;\:\"\']"

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).upper()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = \
        processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    tokenizer = Wav2Vec2CTCTokenizer
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def map_to_result(batch):
    if args.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch


if __name__ == "__main__":
    DATASET_PATH = Path(args.data_dir) / Path(args.data_name)

    dataset = load_dataset("audiofolder", data_dir=str(DATASET_PATH))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    print(dataset)

    dataset = dataset.map(remove_special_characters)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
                                                 padding_value=0.0, do_normalize=True,
                                                 return_attention_mask=False)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h",
                                                  feature_extractor=feature_extractor,
                                                  tokenizer=tokenizer)

    dataset = dataset.map(prepare_dataset, remove_columns=["audio"])

    print(dataset.shape)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h",
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=args.feat_proj_dropout,
        layerdrop=args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.freeze_feature_encoder()

    run = wandb.init(project="huggingface")

    run.config.update(args)

    with open('run_config.json', 'w') as config_file:
        json.dump(dict(run.config), config_file)

    run.save("run_config.json")

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{run.name}",
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=500,
        num_train_epochs=args.num_epochs,
        fp16=args.use_cuda,  # for CUDA, it is True
        gradient_checkpointing=False,
        learning_rate=1e-4,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        save_total_limit=2,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
    )

    if args.finetune:
        trainer.train()
        trainer.save_model(f"{args.output_dir}/{run.name}")

    results = dataset["test"].map(map_to_result,
                                  remove_columns=dataset["test"].column_names)
    results_df = results.to_pandas()
    results_sample = results_df.take(range(50))
    run.log({"results": wandb.Table(dataframe=results_sample)})

    test_wer = wer_metric.compute(predictions=results["pred_str"], references=results["text"])
    run.log({"test_wer": test_wer})
    run.finish()
