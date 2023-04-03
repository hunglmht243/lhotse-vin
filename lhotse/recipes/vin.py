"""
LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus. The main differences from the LibriSpeech corpus are listed below:
The audio files are at 24kHz sampling rate.
The speech is split at sentence breaks.
Both original and normalized texts are included.
Contextual information (e.g., neighbouring sentences) can be extracted.
Utterances with significant background noise are excluded.
For more information, refer to the paper "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, arXiv, 2019. If you use the LibriTTS corpus in your work, please cite this paper where it was introduced.
"""
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import csv
from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, safe_extract, urlretrieve_progress

VIN = (
    "dev",
    "train",
)


def prepare_vin(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "all",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
    link_previous_utt: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :param link_previous_utt: If true adds previous utterance id to supervisions.
        Useful for reconstructing chains of utterances as they were read.
        If previous utterance was skipped from LibriTTS datasets previous_utt label is None.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "all" or dataset_parts[0] == "all":
        dataset_parts = VIN
    elif isinstance(dataset_parts, str):
        assert dataset_parts in VIN
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, prefix="libritts"
        )
    data = {}
    tsvfile='final.tsv'
    with open(corpus_dir/tsvfile, 'r',encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
          speaker_id, wav_path, transcript = row    
          wav_path1=wav_path[:-4]
          data[wav_path1]=[speaker_id,transcript]
    # print(data)
    for part in dataset_parts:

        part_path = corpus_dir / part
        # print(part)
        recordings = RecordingSet.from_dir(part_path, "*.wav", num_jobs=num_jobs)
        supervisions = []
        for rec in recordings:
            rec_id=rec.id
            supervisions.append(
                SupervisionSegment(
                    id=rec_id,
                    recording_id=rec_id,
                    start=0.0,
                    duration=recordings[rec_id].duration,
                    channel=0,
                    language="Vietnamese",
                    text=data[rec_id][1],
                    speaker=data[rec_id][0],
                )
            )

        supervisions = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            supervisions.to_file(output_dir / f"vin_supervisions_{part}.jsonl.gz")
            recordings.to_file(output_dir / f"vin_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recordings, "supervisions": supervisions}

    return manifests

