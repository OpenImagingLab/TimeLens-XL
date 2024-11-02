from tools.registery import DATASET_REGISTRY, LOSS_REGISTRY, PARAM_REGISTRY
import params, losses, models, dataset
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tools.interface_deparse import keyword_parse
from tools.model_deparse import *
from tools.initOptimScheduler import init_optimizer
from tqdm import tqdm
import time


def main():
    print('In Network')
    params = keyword_parse()

    model, epoch, metrics = deparse_model(params)

    # define loader
    # define testing dataloader
    test_num_workers = 4 if 'num_workers' not in params.validation_config.keys() else params.validation_config.num_workers
    testDataset = DATASET_REGISTRY.get(params.validation_config.dataloader)(params, training=False)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=test_num_workers)
    print('[Testing Samples Num]', len(testDataset))
    print("Num workers", test_num_workers)
    # define training dataloader
    if params.enable_training:
        trainDataset = DATASET_REGISTRY.get(params.training_config.dataloader)(params)
        trainLoader = DataLoader(trainDataset, batch_size=params.training_config.batch_size, shuffle=True,
                                 num_workers=params.training_config.num_workers, drop_last=True,
                                 pin_memory=True, prefetch_factor=5)
        print('[Training Samples Num]', len(trainDataset))

        # init optimizer and scheduler
        optimizer, scheduler, scheduler_type, epoch = init_optimizer(model, epoch, params, len(trainDataset))
    for cepoch in range(epoch, params.training_config.max_epoch+1):
        model._update_training_time()
        if params.enable_training:
            # st = time.time()
            for step, data in enumerate(trainLoader):
               #print(step, time.time()-st)
               model.net_training(data, optimizer, cepoch, step)
               if scheduler is not None and scheduler_type == 'step':
                   scheduler.step()
               # st = time.time()
            if scheduler is not None and scheduler_type == 'epoch':
                scheduler.step()
            log_cont = model._print_train_log(cepoch)
            save_model(cepoch, metrics, params, model)
        else:
            print('Do not train, start validation')
            log_cont = f'Only for Validation EPOCH: {cepoch} '
            params.validation_config.val_epochs = cepoch
            params.validation_config.val_imsave_epochs = cepoch

        if (cepoch % max(params.validation_config.val_epochs, 1) == 0 and cepoch > 0) or not params.enable_training:
            torch.cuda.empty_cache()
            for _, testdata in tqdm(enumerate(testLoader)):
                model.net_validation(testdata, cepoch)
            log_cont = log_cont.strip('\n')+'\t'
            log_cont += model._print_val_log()
            log_cont = log_cont.strip('\t')+'\n'
        model._init_metrics(None)
        model.write_log(log_cont)
        if not params.enable_training:
            print('Training is not enabled...Validation finished, end the process...')
            exit()

    if not params.enable_training:
        for _, testdata in enumerate(testLoader):
            model.net_validation(testdata, epoch)
        log_cont = f'Only for Validation EPOCH: {epoch} '
        log_cont = log_cont.strip('\n')+'\t'
        log_cont += model._print_val_log()
        log_cont = log_cont.strip('\t')+'\n'
        model._init_metrics(None)
        model.write_log(log_cont)
    return


if __name__ == '__main__':
    main()


