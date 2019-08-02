import os
import shutil
import numpy as np
import audio
# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import cpu_count
from functools import partial
from nnmnkwii.datasets import vctk

# import audio
import Audio
import hparams as hp


def save_mel_spec(wav_file, save_file):
    mel_spec = Audio.tools.get_mel(wav_file)
    save_file = os.path.join("dataset", save_file)
    np.save(save_file, mel_spec)


def save_by_list(wav_list, save_list):
    for ind, wav_file in enumerate(wav_list):
        mel_spec = Audio.tools.get_mel(wav_file)
        save_file = os.path.join("dataset", save_list[ind])
        np.save(save_file, mel_spec)


def preprocess():
    speakers = vctk.available_speakers
    wav_paths = vctk.WavFileDataSource(
        hp.origin_data, speakers=speakers).collect_files()

    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    cnt_speaker = -1
    cnt_num = -1
    dict_speaker = list()
    num_dict = list()

    mel_spec_list = list()

    for wav_file in wav_paths:
        base_name = os.path.basename(wav_file)
        speaker_id = int(base_name[1:4])
        cnt_id = int(base_name[5:8])

        if speaker_id not in dict_speaker:
            dict_speaker.append(speaker_id)
            cnt_speaker = cnt_speaker + 1
            num_dict.clear()
            cnt_num = -1

        if cnt_id not in num_dict:
            num_dict.append(cnt_id)
            cnt_num = cnt_num + 1

        spec_name = str(cnt_speaker) + "_" + str(cnt_num) + ".npy"
        mel_spec_list.append(spec_name)

    # executor = ProcessPoolExecutor(max_workers=cpu_count())
    # futures = list()

    wav_temp_list = list()
    save_temp_list = list()
    total_len = len(wav_paths)

    for ind, wav_file in enumerate(wav_paths):
        wav_temp_list.append(wav_file)
        save_temp_list.append(mel_spec_list[ind])

        if (((ind + 1) % 1000) == 0) or (ind == total_len - 1):
            save_by_list(wav_temp_list.copy(), save_temp_list.copy())
            # futures.append(executor.submit(
            #     partial(save_by_list, wav_temp_list.copy(), save_temp_list.copy())))

            wav_temp_list.clear()
            save_temp_list.clear()
            print("Done", ind+1)

    print("Done")
    # [future.result() for future in futures]


if __name__ == "__main__":
    # Test
    preprocess()

    for i in range(len(vctk.available_speakers)):
        if not os.path.exists(os.path.join("dataset", str(i))):
            os.mkdir(os.path.join("dataset", str(i)))

    for i in range(len(vctk.available_speakers)):
        file_name_list = os.listdir("dataset")
        for file_name in file_name_list:
            if file_name[-3:] == "npy":
                if str(i) == file_name[0:len(str(i))] and file_name[len(str(i))] == "_":
                    shutil.move(os.path.join("dataset", file_name),
                                os.path.join("dataset", str(i)))
