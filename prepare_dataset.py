from dataset.RC_4816.dataset_dict import EVSneg3, test_key, dataset_dict
import os
from tools.file_path_index import parse_path_common

data_paths = parse_path_common()


def split_training_and_evaluation():
    path_dict = {}
    folders = data_paths.keys()
    for folder in folders:
        curpath, ecurpath = data_paths[folder]
        if folder in dataset_dict or folder in EVSneg3:
            files = sorted(curpath)
            startIdx = int(os.path.split(files[0])[-1].split("_")[0])
            # filepath = [os.path.join(curpath, file) for file in files]
            filepath = files
            eventpath = sorted(ecurpath)
            sub_idx = 0
            if folder in dataset_dict:
                data_list = dataset_dict[folder]
                while len(data_list) > 0:
                    start = data_list.pop(0)
                    end = data_list.pop(0)
                    start = start - startIdx
                    end = end - startIdx
                    k = f"{folder}_{sub_idx}" if sub_idx > 0 else folder
                    path_dict.update({
                        k: [filepath[start:end], eventpath[start:end]]
                    })
                    sub_idx += 1
            else:
                k = folder
                path_dict.update({
                    k: [filepath, eventpath]
                })
                sub_idx += 1
    return path_dict

def samples_indexing(path_dict, training_flag):
    for k in path_dict.keys():
        if training_flag:
            if k not in test_key:
                rgb_path, evs_path = path_dict[k]
            else:
                rgb_path, evs_path = [], []
        else:
            if k in test_key:
                rgb_path, evs_path = path_dict[k]
            else:
                rgb_path, evs_path = [], []
        indexes = list(range(0, len(rgb_path)-1, 1))
        # for irl in interp_ratio_list:
        #     for i_ind in range(0, len(indexes) - irl, 1 if self.training_flag else irl):
        #         rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + irl + 1]]
        #         evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + irl]]
        #         rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
        #         self.samples_dict[str(irl)].append([k, rgb_name, rgb_sample, evs_sample])
        for i_ind in range(0, len(indexes)):
            rgb_sample = [rgb_path[sind]]

    return