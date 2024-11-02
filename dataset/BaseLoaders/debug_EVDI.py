import numpy as np
import torch

def events_dense_to_sparse(events_in, ind_t, events_channel=16):
    previous_events_out = torch.zeros(
        (events_channel * 2, events_in.shape[1], events_in.shape[2])).float()
    post_events_out = torch.zeros(
        (events_channel * 2, events_in.shape[1], events_in.shape[2])).float()
    previous_event = events_in[:ind_t, ...]
    previous_event = previous_event.flip(0)
    previous_index = np.linspace(0, ind_t, events_channel + 1)[1:]
    itind = 0
    for i in range(ind_t):
        if i > previous_index[itind]:
            itind += 1
        previous_events_out[itind][previous_event[i] > 0] += 1
        previous_events_out[itind + events_channel][previous_event[i] < 0] += 1
    post_event = events_in[ind_t:, ...]
    post_index = np.linspace(0, post_event.shape[0], events_channel + 1)[1:]
    itind = 0
    for i in range(post_event.shape[0]):
        if i > post_index[itind]:
            itind += 1
        post_events_out[itind][post_event[i] > 0] += 1
        post_events_out[itind + events_channel][post_event[i] < 0] += 1
    return previous_events_out, post_events_out

fake_events_in = torch.randn(128, 4, 4)
fake_events_in[fake_events_in > 0.5] = 1
fake_events_in[fake_events_in < -0.5] = -1
fake_events_in[(fake_events_in > -0.5) & (fake_events_in < 0.5)] = 0

for it in range(8, 15*8+1, 8):
    print(it)
    events_dense_to_sparse(fake_events_in, it)