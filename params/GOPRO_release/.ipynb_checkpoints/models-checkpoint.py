from easydict import EasyDict as ED


model_arch_config = ED()
model_arch_config.EVDI_Net = ED()
model_arch_config.EVDI_Net.events_channel_num = 16
model_arch_config.EVDI_Net.train_config_dataloader = 'loader_EVDI'
model_arch_config.EVDI_Net.val_config_dataloader = 'loader_EVDI'


model_arch_config.E_TRFNetv0 = ED()
model_arch_config.E_TRFNetv0.channel_scalar = 8
model_arch_config.E_TRFNetv0.events_channel_num = 128
model_arch_config.E_TRFNetv0.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.E_TRFNetv0.val_config_dataloader = 'loader_ERFNetv0'

'''
Replace original relu+sigmoid by exp, which fits sensor physics better
'''
model_arch_config.E_TRFNetv0Exp = ED()
model_arch_config.E_TRFNetv0Exp.channel_scalar = 8
model_arch_config.E_TRFNetv0Exp.events_channel_num = 128
model_arch_config.E_TRFNetv0Exp.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.E_TRFNetv0Exp.val_config_dataloader = 'loader_ERFNetv0'
model_arch_config.E_TRFNetv0Exp.coef = 0.25

model_arch_config.E_TRFNetv0ExpSkipFusion = ED()
model_arch_config.E_TRFNetv0ExpSkipFusion.channel_scalar = 8
model_arch_config.E_TRFNetv0ExpSkipFusion.events_channel_num = 128
model_arch_config.E_TRFNetv0ExpSkipFusion.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.E_TRFNetv0ExpSkipFusion.val_config_dataloader = 'loader_ERFNetv0'
model_arch_config.E_TRFNetv0ExpSkipFusion.coef = 0.25

model_arch_config.REFID = ED()
model_arch_config.REFID.define_model = ED()
model_arch_config.REFID.define_model.type = 'FinalBidirectionAttenfusion'  # UNetPSDecoderRecurrent #UNetDecoderRecurrent
model_arch_config.REFID.define_model.img_chn = 6  # 6 for two image, 26 for image and voxel
model_arch_config.REFID.define_model.ev_chn = 2
model_arch_config.REFID.define_model.num_encoders = 3
model_arch_config.REFID.define_model.base_num_channels = 32
model_arch_config.REFID.define_model.out_chn = 1
# recurrent_block_type: 'simpleconvThendown' # 'convlstm' or 'convgru' or 'simpleconv' or 'simpleconvThendown'
model_arch_config.REFID.define_model.num_block = 1  # num_block of resblock in the bottleneck of unet
model_arch_config.REFID.define_model.num_residual_blocks = 2
model_arch_config.REFID.train_config_dataloader = 'loader_REFID'
model_arch_config.REFID.val_config_dataloader = 'loader_REFID'


model_arch_config.Expv2 = ED()
model_arch_config.Expv2.define_model = ED()
model_arch_config.Expv2.define_model.type = 'FinalBidirectionAttenfusion'  # UNetPSDecoderRecurrent #UNetDecoderRecurrent
model_arch_config.Expv2.define_model.img_chn = 6  # 6 for two image, 26 for image and voxel
model_arch_config.Expv2.define_model.ev_chn = 2
model_arch_config.Expv2.define_model.num_encoders = 3
model_arch_config.Expv2.define_model.base_num_channels = 32
model_arch_config.Expv2.define_model.out_chn = 3
# recurrent_block_type: 'simpleconvThendown' # 'convlstm' or 'convgru' or 'simpleconv' or 'simpleconvThendown'
model_arch_config.Expv2.define_model.num_block = 1  # num_block of resblock in the bottleneck of unet
model_arch_config.Expv2.define_model.num_residual_blocks = 2
model_arch_config.Expv2.train_config_dataloader = 'loader_REFID'
model_arch_config.Expv2.val_config_dataloader = 'loader_REFID'