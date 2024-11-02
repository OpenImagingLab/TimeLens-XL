import os
from natsort import natsorted
try:
    from dataset.RC_4816.dataset_dict import EVSneg3
except:
    print("EVSneg3 do not load")
    pass


def parse_path(folder, fdepth):
    file_dict = {}
    if fdepth == 1:
        file_dict.update({
            os.path.split(folder)[-1]:[os.path.join(folder, file) for file in natsorted(os.listdir(folder))]
        })
    else:
        for subfolder in os.listdir(folder):
            sfolder = os.path.join(folder, subfolder)
            file_dict.update({
                subfolder:[os.path.join(sfolder, file) for file in natsorted(os.listdir(sfolder))]
            })
    return file_dict


def parse_path_common(folder0, folder1, fdepth=2, cbmnet=False, hsergb=False, bsergb=False, RC=False):
    file_dict = {}
    if cbmnet:
        if folder1 is None:
            folder1 = folder0
        folders = os.listdir(folder0)
        for folder in natsorted(folders):
            file_dict.update({
                folder : [[os.path.join(folder0, folder, 'processed_images', pi) for pi in natsorted(os.listdir(os.path.join(folder0, folder, 'processed_images')))],
                       [os.path.join(folder1, folder, 'processed_events', pe) for pe in natsorted(os.listdir(os.path.join(folder1, folder, 'processed_events')))]]
            })
    elif hsergb:
        if folder1 is None:
            folder1 = folder0
        rootfolders = [os.path.join(folder0, 'close/test'),
                       os.path.join(folder0, 'far/test')]
        folders = []
        for rf in rootfolders:
            for rsubf in os.listdir(rf):
                folders.append(os.path.join(rf, rsubf))
        for folder in natsorted(folders):
            folderkey = os.path.split(folder)[-1]
            file_dict.update({
                folderkey : [[os.path.join(folder0, folder, 'images_corrected', pi) for pi in natsorted(os.listdir(os.path.join(folder0, folder, 'images_corrected')))],
                       [os.path.join(folder1, folder, 'events_aligned', pe) for pe in natsorted(os.listdir(os.path.join(folder1, folder, 'events_aligned')))]]
            })
    elif bsergb:
        if folder1 is None:
            folder1 = folder0
        folders = os.listdir(folder0)
        for folder in natsorted(folders):
            try:
                file_dict.update({
                    folder : [[os.path.join(folder0, folder, 'images', pi) for pi in natsorted(os.listdir(os.path.join(folder0, folder, 'images')))],
                           [os.path.join(folder1, folder, 'events', pe) for pe in natsorted(os.listdir(os.path.join(folder1, folder, 'events')))]]
                })
            except:
                pass
    elif RC:
        subfolders = os.listdir(folder0)
        for sf in natsorted(subfolders):
            evs_keyword = 'RGB-EVS' if sf not in EVSneg3 else 'RGB-EVS_EVSneg3ms'
            offset = 1 if sf in EVSneg3 else 0
            evs_len = len(os.listdir(os.path.join(folder0, sf, evs_keyword)))
            file_dict.update({
                sf: [[os.path.join(folder0, sf, 'visual_RGB', pi) for pi in natsorted(os.listdir(os.path.join(folder0, sf, 'visual_RGB')))[offset:evs_len]],
                     [os.path.join(folder0, sf, evs_keyword, pi) for pi in natsorted(os.listdir(os.path.join(folder0, sf, evs_keyword)))]]
            })
    else:
        if fdepth == 1:
            folder0_item = len(os.listdir(folder0))
            folder1_item = len(os.listdir(folder1))
            if abs(folder1_item-folder0_item) != 1:
                print(f'Folder item not match, skip: {folder0}, {folder1}')
            else:
                file_dict.update({
                    os.path.split(folder0)[-1]:[[os.path.join(folder0, file) for file in natsorted(os.listdir(folder0))],
                                                [os.path.join(folder1, file) for file in natsorted(os.listdir(folder1))]]
                })
        else:
            sf0 = os.listdir(folder0)
            sf1 = os.listdir(folder1)
            common_sf = []
            for sf in sf0:
                if sf in sf1:
                    common_sf.append(sf)
            for subfolder in common_sf:
                sfolder0 = os.path.join(folder0, subfolder)
                sfolder1 = os.path.join(folder1, subfolder)
                folder0_item = len(os.listdir(sfolder0))
                folder1_item = len(os.listdir(sfolder1))
                # print(folder0, folder1, subfolder, folder0_item, folder1_item)
                file_dict.update({
                    subfolder:[[os.path.join(sfolder0, file) for file in natsorted(os.listdir(sfolder0))],
                               [os.path.join(sfolder1, file) for file in natsorted(os.listdir(sfolder1))]]
                })
    return file_dict
