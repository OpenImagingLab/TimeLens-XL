from easydict import EasyDict as ED
from tools import parse_path_common
import os
from params.models import model_arch_config
from tools.registery import PARAM_REGISTRY
from params.Paths.RealCaptured import hostname
from params.Paths.RealCaptured import RC
# from params.Paths import GOPRO

mkdir = lambda x:os.makedirs(x, exist_ok=True)

@PARAM_REGISTRY.register()
def timelens_RC_smallmix_lpips(args):
    paths = ED()
    paths.train_rgb = RC.train.rgb
    paths.train_evs = RC.train.evs
    # paths.train_path_dict = parse_path_common(paths.train_rgb, paths.train_evs)
    # Interpolation ratio to generate events
    # paths.train_rgb_skip = 1
    paths.test_rgb = RC.test.rgb
    paths.test_evs = RC.test.evs
    # paths.test_path_dict = parse_path_common(paths.test_rgb, paths.test_evs)
    # Interpolation ratio to generate events
    # paths.test_rgb_skip = 1

    paths.save = ED()
    paths.save.save_path = './RGB_resOut_HQEVFI'
    paths.save.exp_path = os.path.join(paths.save.save_path, f"mix_{args.model_name}x16" + f"_adamLPIPS{args.extension}")
    paths.save.record_txt = os.path.join(paths.save.exp_path, 'training_record.txt')
    paths.save.train_im_path = os.path.join(paths.save.exp_path, 'trainining_Visual_Examples')
    paths.save.val_im_path = os.path.join(paths.save.exp_path, 'Validation_Visual_Examples')
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
    training_config.dataloader = 'loader_RC_timelens_smallmix'
    training_config.crop_size = 256 if hostname == 'server' else 96
    training_config.num_workers = 16 if hostname == 'server' else 1
    training_config.batch_size = 4 if hostname == 'server' else 2
    if not args.calc_flops and not args.skip_training:
        training_config.data_paths = parse_path_common(paths.train_rgb, paths.train_evs, RC=True)
    training_config.data_index_offset = 0
    training_config.rgb_sampling_ratio = 1
    training_config.interp_ratio = 16
    # training_config.sample_group = 3
    training_config.random_t = True
    training_config.color = 'RGB'
    # training_config.events_channel = 128

    # optimizer and scheduler
    training_config.optim = ED()
    training_config.optim.name = 'Adam'
    training_config.optim.optim_params = ED()
    training_config.optim.optim_params.lr = 1e-4
    # training_config.optim.optim_params.weight_decay = 1e-4
    # training_config.optim.optim_params.betas = [0.9, 0.99]
    training_config.optim.scheduler = 'multilr'
    # training_config.lr = 2e-4
    training_config.optim.scheduler_params = ED()
    training_config.optim.scheduler_params.milestones = [12, 24]
    training_config.optim.scheduler_params.gamma = 0.1
    # training_config.optim.scheduler = 'MultiLR'
    # training_config.lr = 1e-4
    # training_config.optim.scheduler_lr_gamma = 0.5
    # training_config.optim.scheduler_lr_milestone = [25, 50, 75]
    training_config.max_epoch = 27

    training_config.losses = ED()
    training_config.losses.Charbonier = ED()
    training_config.losses.Charbonier.weight = 1.
    training_config.losses.Charbonier.as_loss = True

    training_config.losses.lpips = ED()
    training_config.losses.lpips.weight = 0.1
    training_config.losses.lpips.as_loss = True

    training_config.losses.psnr = ED()
    training_config.losses.psnr.weight = 1.
    training_config.losses.psnr.as_loss = False
    training_config.losses.psnr.test_y_channel = False

    # For training loss print
    training_config.train_stats = ED()
    training_config.train_stats.print_freq = 500
    training_config.train_stats.save_im_ep = 5
    # if not args.calc_flops:
    #     training_config.data_paths = parse_path_common(paths.train_rgb, paths.train_evs, RC=True)

    validation_config = ED()
    validation_config.dataloader = 'loader_RC_timelens_smallmix'
    validation_config.val_epochs = 9
    validation_config.val_imsave_epochs = 27
    validation_config.weights_save_freq = 1
    validation_config.crop_size = None
    if not args.calc_flops:
        validation_config.data_paths = parse_path_common(paths.test_rgb, paths.test_evs, RC=True)
    validation_config.data_index_offset = 0
    validation_config.rgb_sampling_ratio = 1
    validation_config.interp_ratio = 16
    validation_config.real_interp = 16
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
    params.training_stage = 'tuning'
    return params
