import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


from lhotse import CutSet, Fbank, RecordingSet
from lhotse.dataset import (
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
    PerturbSpeed,
    PerturbVolume,
    RandomizedSmoothing,
    ReverbWithImpulseResponse,
    SpecAugment,
)
from lhotse import SupervisionSet, SupervisionSegment,RecordingSet
# from lhotse.recipes import (
#     prepare_vin
# )
# root_dir = Path("D:/Download/cv-corpus-13.0-2023-03-09/vin_test")
# # print(root_dir)
# num_jobs = os.cpu_count() - 1
# vin = prepare_vin(
#     root_dir,  output_dir=root_dir
# )
# cuts_train = CutSet.from_manifests(**vin["train"]).trim_to_supervisions()

# print(cuts_train.describe())

# sups2 = SupervisionSet.from_file('D:\\Download\\cv-corpus-13.0-2023-03-09\\vin_test\\vin_supervisions_train.jsonl.gz')
# rec=RecordingSet.from_jsonl('D:\\Download\\cv-corpus-13.0-2023-03-09\\vin_test\\vin_recordings_train.jsonl.gz')
# print(rec[0])