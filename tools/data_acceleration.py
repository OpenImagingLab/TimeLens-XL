import sys
import os
sys.path.append(os.getcwd())
from params.GOPROv2eIMX636.params_trainGOPROVFI import trainGOPROVFI
from easydict import EasyDict as ED
from multiprocessing import Process, Queue
# for EVDI Net dataloader preparation
import numpy as np
from skimage.io import imread

process = 16
args = ED()
args.model_name = 'EVDI_Net'
args.extension = ''
args.clear_previous = None
args.model_pretrained = None
params = trainGOPROVFI(args)
params.training_config.crop_size = None

train_data_save = '/mnt/workspace/mayongrui/dataset/GOPRO/train_x8interpolated_acc_forx16interp'
test_data_save = '/mnt/workspace/mayongrui/dataset/GOPRO/test_x8interpolated_acc_forx16interp'

os.makedirs(train_data_save, exist_ok=True)
os.makedirs(test_data_save, exist_ok=True)

def indexing_files(data_paths, rgb_sampling_ratio, interp_ratio, training_flag):
    samples_list = []
    for k in data_paths.keys():
        rgb_path, evs_path = data_paths[k]
        indexes = list(range(0, len(rgb_path),
                             rgb_sampling_ratio))
        for i_ind in range(0, len(indexes) - interp_ratio, 1 if training_flag else interp_ratio):
            # print(i_ind, self.interp_ratio, len(indexes), indexes[0], indexes[-1], len(rgb_path), len(evs_path))
            rgb_sample = [rgb_path[sind] for sind in indexes[i_ind:i_ind + interp_ratio + 1]]
            evs_sample = evs_path[indexes[i_ind]:indexes[i_ind + interp_ratio]]
            rgb_name = [os.path.splitext(os.path.split(rs)[-1])[0] for rs in rgb_sample]
            samples_list.append([k, rgb_name, rgb_sample, evs_sample])
    print(f'[Data statistics] Training FLag: {len(samples_list)}')
    return samples_list

#@jit(nopython=True)
def events_dense_to_sparse(events_in, ind_t, events_channel=16):
    previous_events_out = np.zeros(
        (events_channel * 2, events_in.shape[1], events_in.shape[2]), dtype=np.int8)
    post_events_out = np.zeros(
        (events_channel * 2, events_in.shape[1], events_in.shape[2]), dtype=np.int8)
    previous_event = np.copy(events_in[:ind_t, ...])
    previous_event = previous_event[::-1]
    previous_index = np.linspace(0, ind_t, events_channel + 1)[1:]
    itind = 0
    for i in range(ind_t):
        if i > previous_index[itind]:
            itind += 1
        previous_events_out[itind][previous_event[i] > 0] += 1
        previous_events_out[itind + events_channel][previous_event[i] < 0] += 1
    post_event = np.copy(events_in[ind_t:, ...])
    post_index = np.linspace(0, post_event.shape[0], events_channel + 1)[1:]
    itind = 0
    for i in range(post_event.shape[0]):
        if i > post_index[itind]:
            itind += 1
        post_events_out[itind][post_event[i] > 0] += 1
        post_events_out[itind + events_channel][post_event[i] < 0] += 1
    return previous_events_out, post_events_out

#def processing_and_save(samples_list, index, save_path):
def processing_and_save(q, total):
    samples_list, index, save_path = q.get()
    print(f"Current Procesing: [{q.qsize()}/{total}]")
    k, rgb_name, rgb_sample, evs_sample = samples_list[index]
    rgb_data = [imread(rgb_s) for rgb_s in rgb_sample]
    evs_data = [np.load(evs_s, allow_pickle=True)['data'] for evs_s in evs_sample]
    rgb_data = np.stack(rgb_data)
    evs_data = np.stack(evs_data)
    sample_t = list(range(1, 16))
    previous_events = []
    post_events = []

    for t in sample_t:
        pree, pose = events_dense_to_sparse(evs_data, t * 8, 16)
        previous_events.append(pree)
        post_events.append(pose)
    previous_events = np.stack(previous_events).astype(np.int8)
    post_events = np.stack(post_events).astype(np.int8)
    np.savez_compressed(os.path.join(save_path, f"{k}_{os.path.split(os.path.splitext(rgb_sample[0])[0])[-1]}"), folder=k,
                          rgb_name = rgb_name, rgb_data =rgb_data, evs_data = evs_data, previous_events=previous_events,
                        post_events=post_events)


training_files = indexing_files(params.training_config.data_paths, 8, 16, True)
testing_files = indexing_files(params.training_config.data_paths, 8, 16, True)

print('Start Processing Training Data')
queue = Queue()
for files in range(len(training_files)):
    queue.put([training_files, files, train_data_save])
total = queue.qsize()
print("Finish putting commands to queue")
print(f"Queue size: {queue.qsize()}")
while queue.qsize() > 0:
    p_list = []
    for i in range(min(queue.qsize(), process)):
        p = Process(target=processing_and_save, args=(queue, total))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
