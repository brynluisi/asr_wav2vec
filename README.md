# Fine-Tuning Wav2Vec2 for Automatic Speech Recognition (ASR)

This project fine-tunes the Wav2Vec2 model for automatic speech recognition (ASR) using the Vercellotti dataset. The pipeline involves data preprocessing, model training, and evaluation.

## Project Structure

- `asr_wav2vec.py` - Main script for fine-tuning the ASR model.
- `preprocessing.py` - Handles downloading and processing of the dataset.
- `vocab.json` - Vocabulary file for tokenization.
- `run_config.json` - Configuration file saved during training.
- `output/` - Directory where trained models and logs are saved.

## Dataset

The dataset is sourced from [TalkBank](https://slabank.talkbank.org/). The preprocessing script:
- Downloads and extracts transcripts and audio files.
- Cleans transcripts and aligns them with corresponding audio segments.
- Splits data into training, validation, and test sets.

## Usage

### 1. Data Preparation

Run the preprocessing script to download and process the dataset:

```bash
python preprocessing.py
```

### 2. Fine-Tuning Wav2Vec2

Run the main script to fine-tune the model:

```bash
python asr_wav2vec.py --data_dir <path_to_data> --output_dir <path_to_output> --use_cuda True --finetune True
```

#### Key Arguments:
- `--data_dir`: Directory containing the dataset.
- `--output_dir`: Directory to save the trained model.
- `--use_cuda`: Whether to use GPU for training.
- `--finetune`: If `True`, the model is trained on new data.

### 3. Model Evaluation

The script logs results using [Weights & Biases](https://wandb.ai/), including:
- Word Error Rate (WER) computation.
- Sample predictions from the test set.

## Acknowledgments

This implementation uses the Hugging Face Wav2Vec2 model and [Vercelloti](https://media.
talkbank.org/slabank/English/Vercellotti) dataset.

