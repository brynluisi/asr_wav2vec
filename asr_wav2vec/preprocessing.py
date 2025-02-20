import re
import random
import zipfile
from pathlib import Path
from typing import Iterable, Set, Union, Tuple

import requests
import pylangacq
from tqdm import tqdm
from datasets import Audio, load_dataset
from pydub import AudioSegment


# This should be move to YAML config file once we agree how to use it
DATASET_DIR = Path(__file__).parent.parent / "data"
DATASET_NAME = "Vercellotti"
DATASET_PATH = DATASET_DIR / DATASET_NAME
TRANSCRIPTS_URL = "https://slabank.talkbank.org/data/English/Vercellotti.zip"
AUDIO_URL = "https://media.talkbank.org/slabank/English/Vercellotti"

NOTRANS = "notrans"
SAMPLE_MIN_LENGTH = 10
VAL_SIZE = 0.1
TEST_SIZE = 0.2
RANDOM_SEED = 42

# r"\*.*?\%"g

random.seed(RANDOM_SEED)
PathType = Union[Path, str]


class Vercellotti(pylangacq.Reader):

    @staticmethod
    def _get_time_marks(line: str) -> Union[Tuple[int, int], None]:
        time_marks = pylangacq.chat._TIMER_MARKS_REGEX.findall(line)
        if time_marks:
            return int(time_marks[0][0]), int(time_marks[-1][-1])
        else:
            return None

    @staticmethod
    def _mkdir(dir: PathType) -> Path:
        """ Creata the directory including all the parents even when it already exists.

        Parameters
        ----------
        dir : PathType
            Path to directory to be created.

        Returns
        -------
        Path
            Path object to the created directory.
        """
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        return dir
    
    @staticmethod
    def is_transcripted(file_path: PathType) -> bool:
        with open(file_path, "r") as f:
            text = f.read()
        return not re.search(NOTRANS, text)

    @staticmethod
    def download_raw_transcripts(save_path: PathType) -> Path:
        # Download raw transcripts:
        save_path = Vercellotti._mkdir(save_path)
        parent_path = save_path.parent
        zip_path = parent_path / "raw-transcripts.zip"

        with open(zip_path, "wb") as f:
            response = requests.get(TRANSCRIPTS_URL)
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(parent_path)

        (parent_path / DATASET_NAME).rename(save_path)

        return save_path

    @staticmethod
    def download_raw_audios(
        file_ids: Iterable[str], save_path: PathType
    ) -> Path:
        """"""
        save_path = Vercellotti._mkdir(save_path)

        for file_id in tqdm(file_ids):
            audio_url = f"{AUDIO_URL}/{file_id}.mp3"
            file_path = save_path / f"{file_id}.mp3"
            with open(file_path, "wb") as f:   
                response = requests.get(audio_url)
                f.write(response.content)
        
        return save_path

    @staticmethod
    def process(
        file_id: str, raw_path: PathType, save_path: PathType,
    ) -> PathType:
        """ Download transcript into text file. 
            The transcripted will be splitted by utterances.

        Parameters
        ----------
        file_id : str
            File id to be downloaded.
        save_path : PathType
            Path to data saving location.
        transcript_url : str
            URL to download stranscript zip to create pylangacq.Reader.
            See here: https://pylangacq.org/read.html

        Returns
        -------
        PathType
            Path to the downloaded transcript.
        """
        save_path = Vercellotti._mkdir(save_path)
        save_transcripts_path = Vercellotti._mkdir(save_path / "transcripts")
        save_audios_path = Vercellotti._mkdir(save_path / "audios")

        raw_path = Path(raw_path)
        raw_transcript_path = raw_path / "transcripts" / f"{file_id}.cha"
        raw_audio_path = raw_path / "audios" / f"{file_id}.mp3"

        audio = AudioSegment.from_file(raw_audio_path, format="mp3")
        
        with open(raw_transcript_path, "r") as f:
            transcript = f.read()
            transcript = re.sub(
                "\s+|\\x15|\&(?<=\&).+?(?<= )|[\<\[\(].*?[\>\]\)]", 
                " ", 
                transcript
            )
            transcript = re.sub(":", "", transcript)
            transcript = " ".join(re.findall("\*.*?\%", transcript))
            transcript = " ".join(re.findall("[a-zA-Z']+|[0-9]+_[0-9]+", transcript))
            transcript = "0 " + transcript

        segments = transcript.split("_")
        buffer = []
        i = 0
        while i < len(segments):
            if len(buffer[1:-1]) < SAMPLE_MIN_LENGTH:
                tokens = segments[i].split()
                if len(buffer) == 0:
                    buffer += tokens
                else:
                    buffer = buffer[:-1] + tokens[1:]
                i += 1
            else:
                # Save processed transcript:
                processed_transcript_path = save_transcripts_path / f"{file_id}-{i}.txt"
                with open(processed_transcript_path, "w") as f:
                    f.write(" ".join(buffer[1:-1]))

                # Save processed audio:
                audio_segment = audio[int(buffer[0]):int(buffer[-1])]
                processed_audio_path = save_audios_path / f"{file_id}-{i}.mp3"
                audio_segment.export(processed_audio_path, format="mp3")

                # Reset:
                buffer = []

        return processed_transcript_path

    @staticmethod
    def train_test_split(
        file_ids: Iterable[str], val_size: float, test_size: float
    ) -> Set[Iterable[str]]:
        """ Split file id into train, validation, and test sets.

        Parameters
        ----------
        file_ids : Iterable[str]
            List of file ids.
        val_size : float
            Percentage of validation dataset.
        test_size : float
            Percentage of test dataset.

        Returns
        -------
        Set[Iterable[str]]
            Splitted train and test id list.
        """
        assert 0.0 <= val_size <= 1.0, "Validation size must be within [0.0, 1.0]"
        assert 0.0 <= test_size <= 1.0, "Test size must be within [0.0, 1.0]"

        id_mapping = dict(zip(
            file_ids,
            map(lambda file_id: file_id.split("_")[0], file_ids)
        ))

        ids = sorted(set(id_mapping.values()))
        random.shuffle(ids)
        val_split_index = round(len(ids) * (1 - (val_size + test_size)))
        test_split_index = round(len(ids) * (1 - test_size))
        train_ids = ids[:val_split_index]
        val_ids = ids[val_split_index:test_split_index]
        test_ids = ids[test_split_index:]
        train_file_ids = filter(lambda file_id: id_mapping[file_id] in train_ids, file_ids)
        val_file_ids = filter(lambda file_id: id_mapping[file_id] in val_ids, file_ids)
        test_file_ids = filter(lambda file_id: id_mapping[file_id] in test_ids, file_ids)

        return train_file_ids, val_file_ids, test_file_ids

    @staticmethod
    def create_dataset(
        file_ids: Iterable[str], save_path: PathType, raw_path: PathType
    ) -> PathType:
        """ Create transcripts and audios.
            The structure follow this instruction to create HuggingFace AudioFolder:
            https://huggingface.co/docs/datasets/audio_dataset#audiofolder

        Parameters
        ----------
        file_ids : Iterable[str]
            List of file ids.
        save_path : PathType
            Path to data saving location.
        raw_path : PathType
            Path to raw data location.

        Returns
        -------
        PathType
            Path to the directory of created dataset.
        """
        save_path = Vercellotti._mkdir(save_path)
        raw_path = Path(raw_path)

        for file_id in tqdm(file_ids):
            try:
                Vercellotti.process(
                    file_id, raw_path, save_path
                )
            except Exception as e:
                print(file_id, e)
        
        return save_path

    @staticmethod
    def generate_metadata_file(
        trainset_path: PathType, valset_path: PathType, testset_path: PathType, save_path: PathType
    ) -> PathType:
        """ Generate metadata csv file.
            The structure follow this instruction to create HuggingFace AudioFolder:
            https://huggingface.co/docs/datasets/audio_dataset#audiofolder

        Parameters
        ----------
        trainset_path : PathType
            Path to the directory of train dataset.
        valset_path : PathType
            Path to the directory of validation dataset.
        testset_path : PathType
            Path to the directory of test dataset.
        save_path : PathType
            Path to data saving location.

        Returns
        -------
        PathType
            Path to the generated metadata file.
        """
        def write(dataset_path, save_path, metadata_writer):
            for audio_path in dataset_path.rglob("*.mp3"):
                transcript_path = dataset_path / "transcripts" / f"{audio_path.stem}.txt"
                try:
                    with open(transcript_path, "r") as f:
                        transcript = f.read()
                    audio_path = audio_path.relative_to(save_path)
                    metadata_writer.writelines(f"{audio_path},{transcript}\n")
                except Exception as e:
                    print(audio_path)
                    print(transcript_path)

        save_path = Vercellotti._mkdir(save_path)
        file_path = save_path / "metadata.csv"

        with open(file_path, "a") as metadata_writer:
            # Write the columns' names:
            metadata_writer.write("file_name,transcription\n")

            write(trainset_path, save_path, metadata_writer)
            write(valset_path, save_path, metadata_writer)
            write(testset_path, save_path, metadata_writer)

        return file_path


if __name__ == "__main__":
    # Download transcript zip file:
    raw_transcripts_path = Vercellotti.download_raw_transcripts(
        f"{DATASET_PATH}/raw/transcripts"
    )
    
    # Filter only transcripted data:
    # raw_transcripts_path = Path(f"{DATASET_PATH}/raw/transcripts")
    transcripted_files_path = filter(
        Vercellotti.is_transcripted,
        raw_transcripts_path.rglob("*.cha")
    )
    file_ids = list(map(lambda path: path.stem, transcripted_files_path))
    
    # Download raw audios:
    raw_audios_path = Vercellotti.download_raw_audios(
        file_ids,
        f"{DATASET_PATH}/raw/audios",
    )

    # Split the id into train and test sets. Ideally, a participant should be
    # in either train or test sets to avoid overfitting:
    train_file_ids, val_file_ids, test_file_ids = Vercellotti.train_test_split(file_ids, VAL_SIZE, TEST_SIZE) 

    # Create train and test datasets:
    trainset_path = Vercellotti.create_dataset(
        train_file_ids, f"{DATASET_PATH}/train", f"{DATASET_PATH}/raw"
    )
    valset_path = Vercellotti.create_dataset(
        val_file_ids, f"{DATASET_PATH}/val", f"{DATASET_PATH}/raw"
    )
    testset_path = Vercellotti.create_dataset(
        test_file_ids, f"{DATASET_PATH}/test", f"{DATASET_PATH}/raw"
    )

    # The return path point to the directory of generated metadata file:
    metadata_path = Vercellotti.generate_metadata_file(
        trainset_path=DATASET_PATH/"train",
        valset_path=DATASET_PATH/"val",
        testset_path=DATASET_PATH/"test",
        save_path=DATASET_PATH
    )

    # Load HuggingFace dataset. At the moment, all data are in the train
    # dataset. Validation and test datasets need some work to split them
    # with similar distribution:
    dataset = load_dataset("audiofolder", data_dir=str(DATASET_PATH))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))


