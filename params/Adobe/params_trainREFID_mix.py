from easydict import EasyDict as ED
from params.Paths import GOPRO
from tools import parse_path_common
import os
from params.models import model_arch_config
from tools.registery import PARAM_REGISTRY
from params.Paths.GOPRO import hostname
from params.Paths.Adobe240 import Adobe


mkdir = lambda x:os.makedirs(x, exist_ok=True)

@PARAM_REGISTRY.register()
def testAdobe_REFIDmix(args):
    paths = ED()
    paths.train_rgb = GOPRO.train.dense_rgb
    paths.train_evs = GOPRO.train.evs_refid
    # paths.train_path_dict = parse_path_common(paths.train_rgb, paths.train_evs)
    # Interpolation ratio to generate events
    # paths.train_rgb_skip = 8
    paths.test_rgb = Adobe.test.rgb
    paths.test_evs = Adobe.test.evs
    # paths.test_path_dict = parse_path_common(paths.test_rgb, paths.test_evs)
    # Interpolation ratio to generate events
    # paths.test_rgb_skip = 8

    paths.save = ED()
    paths.save.save_path = './RGB_Adobe240_release'
    paths.save.exp_path = os.path.join(paths.save.save_path, 'Adobe'+f"_{args.model_name}mix" + f"{args.extension}")
    paths.save.record_txt = os.path.join(paths.save.exp_path, 'training_record.txt')
    paths.save.train_im_path = os.path.join(paths.save.exp_path, 'trainining_Visual_Examples')
    paths.save.val_im_path = os.path.join(paths.save.exp_path, 'Validation_Visual_Examples_x16')
    paths.save.weights = os.path.join(paths.save.exp_path, 'weights')


    if args.clear_previous and os.path.isdir(paths.save.exp_path):
        print(f'-- Select args.clear_previous to be True, delete previous results at {paths.save.exp_path}')
        os.system(f"rm -rf {paths.save.exp_path}")

    for k in paths.save.keys():
        if not k.endswith('_txt'):
            mkdir(paths.save[k])

    model_config = ED()
    model_config.name = args.model_name
    model_config.model_pretrained = args.model_pretrained
    cur_model_arch_config = model_arch_config[model_config.name]
    for k in cur_model_arch_config.keys():
        model_config.update({
            k:cur_model_arch_config[k]
        })
    # model_config.define_model = cur_model_arch_config

    training_config = ED()
    training_config.dataloader = 'loader_REFID_mix'
    training_config.crop_size = 224 if hostname == 'server' else 96
    training_config.num_workers = 24 if hostname == 'server' else 4
    training_config.batch_size = 1 if hostname == 'server' else 1
    if not args.calc_flops and not args.skip_training:
        training_config.data_paths = parse_path_common(paths.train_rgb, paths.train_evs)
    training_config.data_index_offset = 1
    training_config.rgb_sampling_ratio = 8
    training_config.interp_ratio = 16
    # training_config.sample_group = 3
    training_config.random_t = False
    training_config.color = 'RGB'
    # training_config.events_channel = 128

    # optimizer and scheduler
    training_config.optim = ED()
    training_config.optim.name = 'AdamW'
    training_config.optim.optim_params = ED()
    training_config.optim.optim_params.lr = 2e-4
    training_config.optim.optim_params.weight_decay = 1e-4
    # training_config.optim.optim_params.betas = [0.9, 0.99]
    # training_config.optim.name = 'Adam'
    training_config.optim.scheduler = 'CosineAnnealingLR'
    # training_config.lr = 2e-4
    training_config.optim.scheduler_params = ED()
    training_config.optim.scheduler_params.T_max = 250000
    training_config.optim.scheduler_params.eta_min = 1e-7
    # training_config.optim.scheduler = 'MultiLR'
    # training_config.lr = 1e-4
    # training_config.optim.scheduler_lr_gamma = 0.5
    # training_config.optim.scheduler_lr_milestone = [25, 50, 75]
    training_config.max_epoch = 10

    training_config.losses = ED()
    training_config.losses.Charbonier = ED()
    training_config.losses.Charbonier.weight = 1.
    training_config.losses.Charbonier.as_loss = True

    training_config.losses.psnr = ED()
    training_config.losses.psnr.weight = 1.
    training_config.losses.psnr.as_loss = False
    training_config.losses.psnr.test_y_channel = False
    
    
    # training_config.losses.lpips = ED()
    # training_config.losses.lpips.weight = .1
    # training_config.losses.lpips.as_loss = True

    # For training loss print
    training_config.train_stats = ED()
    training_config.train_stats.print_freq = 500
    training_config.train_stats.save_im_ep = 2

    validation_config = ED()
    validation_config.dataloader = 'loader_REFID_mix'
    validation_config.val_epochs = 10
    validation_config.val_imsave_epochs = 10
    validation_config.weights_save_freq = 1
    validation_config.crop_size = None if hostname != 'local' else 192
    if not args.calc_flops:
        validation_config.data_paths = parse_path_common(paths.test_rgb, paths.test_evs)
    validation_config.data_index_offset = 1
    validation_config.rgb_sampling_ratio = 8
    validation_config.interp_ratio = 16
    validation_config.random_t = False
    validation_config.color = 'RGB'

    validation_config.losses = ED()
    validation_config.losses.l1_loss = ED()
    validation_config.losses.l1_loss.weight = 1.
    validation_config.losses.l1_loss.as_loss = False

    validation_config.losses.psnr = ED()
    validation_config.losses.psnr.weight = 1.
    validation_config.losses.psnr.as_loss = False
    validation_config.losses.psnr.test_y_channel=False

    validation_config.losses.ssim = ED()
    validation_config.losses.ssim.weight = 1.
    validation_config.losses.ssim.as_loss = False
    validation_config.losses.ssim.test_y_channel = False
    
    validation_config.losses.lpips = ED()
    validation_config.losses.lpips.weight = 1.
    validation_config.losses.lpips.as_loss = False

    validation_config.losses.dists = ED()
    validation_config.losses.dists.weight = 1.
    validation_config.losses.dists.as_loss = False

    params = ED()
    params.paths = paths
    params.training_config = training_config
    params.validation_config = validation_config
    params.model_config = model_config
    params.events_channel = 128
    return params
