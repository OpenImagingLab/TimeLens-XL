from easydict import EasyDict as ED
from copy import deepcopy as dcopy


model_arch_config = ED()
model_arch_config.EVDI_Net = ED()
model_arch_config.EVDI_Net.events_channel_num = 16
model_arch_config.EVDI_Net.train_config_dataloader = 'loader_EVDI'
model_arch_config.EVDI_Net.val_config_dataloader = 'loader_EVDI'

model_arch_config.SuperSlomo = ED()
model_arch_config.SuperSlomo.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.SuperSlomo.val_config_dataloader = 'loader_ERFNetv0'

model_arch_config.RIFE = dcopy(model_arch_config.SuperSlomo)

model_arch_config.EVDI_Color_Net = ED()
model_arch_config.EVDI_Color_Net.events_channel_num = 16
model_arch_config.EVDI_Color_Net.train_config_dataloader = 'loader_EVDI'
model_arch_config.EVDI_Color_Net.val_config_dataloader = 'loader_EVDI'


model_arch_config.E_TRFNetv0 = ED()
model_arch_config.E_TRFNetv0.channel_scalar = 8
model_arch_config.E_TRFNetv0.events_channel_num = 128
model_arch_config.E_TRFNetv0.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.E_TRFNetv0.val_config_dataloader = 'loader_ERFNetv0'


model_arch_config.REFID = ED()
model_arch_config.REFID.define_model = ED()
model_arch_config.REFID.define_model.type = 'FinalBidirectionAttenfusion'  # UNetPSDecoderRecurrent #UNetDecoderRecurrent
model_arch_config.REFID.define_model.img_chn = 6  # 6 for two image, 26 for image and voxel
model_arch_config.REFID.define_model.ev_chn = 2
model_arch_config.REFID.define_model.num_encoders = 3
model_arch_config.REFID.define_model.base_num_channels = 32

model_arch_config.REFID.define_model.out_chn = 3
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

model_arch_config.Expv2_REFID_decoder = ED()
model_arch_config.Expv2_REFID_decoder.define_model = ED()
model_arch_config.Expv2_REFID_decoder.define_model.type = 'FinalBidirectionAttenfusion'  # UNetPSDecoderRecurrent #UNetDecoderRecurrent
model_arch_config.Expv2_REFID_decoder.define_model.img_chn = 6  # 6 for two image, 26 for image and voxel
model_arch_config.Expv2_REFID_decoder.define_model.ev_chn = 2
model_arch_config.Expv2_REFID_decoder.define_model.num_encoders = 3
model_arch_config.Expv2_REFID_decoder.define_model.base_num_channels = 32
model_arch_config.Expv2_REFID_decoder.define_model.out_chn = 3
# recurrent_block_type: 'simpleconvThendown' # 'convlstm' or 'convgru' or 'simpleconv' or 'simpleconvThendown'
model_arch_config.Expv2_REFID_decoder.define_model.num_block = 1  # num_block of resblock in the bottleneck of unet
model_arch_config.Expv2_REFID_decoder.define_model.num_residual_blocks = 2
model_arch_config.Expv2_REFID_decoder.train_config_dataloader = 'loader_REFID'
model_arch_config.Expv2_REFID_decoder.val_config_dataloader = 'loader_REFID'

model_arch_config.Expv2_withReLUBlocks = dcopy(model_arch_config.Expv2)
model_arch_config.Expv2_REFID_flowdecoder = dcopy(model_arch_config.Expv2)
model_arch_config.Expv2_REFID_flowdecoder.define_model.num_encoders = 2
model_arch_config.Expv2_REFID_decoder_asInit = dcopy(model_arch_config.Expv2)
model_arch_config.Expv2_REFID_decoder_asInit.define_model.type = 'BidirectionalWithFlow'


model_arch_config.Expv2_REFID_FB = dcopy(model_arch_config.Expv2_REFID_flowdecoder)
# model_arch_config.Expv2
model_arch_config.Expv2_REFID_FB_withencoderrefine = dcopy(model_arch_config.Expv2_REFID_flowdecoder)
model_arch_config.Expv2_REFID_flowdecoder_withrefine = dcopy(model_arch_config.Expv2_REFID_flowdecoder)
model_arch_config.Expv2_REFID_FBfwarp = dcopy(model_arch_config.Expv2_REFID_flowdecoder)
model_arch_config.Expv3 = dcopy(model_arch_config.Expv2_REFID_flowdecoder)
model_arch_config.Expv3.define_model.base_channel = 32
model_arch_config.Expv3.define_model.echannel = 128
model_arch_config.Expv3.define_model.interp_ratio = 16
model_arch_config.Expv3.define_model.pos_e = 0.2
model_arch_config.Expv3.define_model.neg_e = 0.2
model_arch_config.Expv3.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.Expv3.val_config_dataloader = 'loader_ERFNetv0'
model_arch_config.Expv2_REFID_FBquick = dcopy(model_arch_config.Expv2_REFID_flowdecoder)
model_arch_config.Expv2_REFID_FB_maskcorrect = dcopy(model_arch_config.Expv2_REFID_flowdecoder)

model_arch_config.Expv3_FBwithrefine = dcopy(model_arch_config.Expv3)
model_arch_config.Expv3_FB = dcopy(model_arch_config.Expv3)
model_arch_config.Expv4_FB = dcopy(model_arch_config.Expv3)
model_arch_config.Expv2_REFID_FB_withencoderrefine_maskcorrect = dcopy(model_arch_config.Expv2_REFID_FB)
model_arch_config.Expv2_REFID_FBquick_newEncoder = dcopy(model_arch_config.Expv2_REFID_FB)
model_arch_config.Expv2_REFID_FBquick_newEncoder.train_config_dataloader = 'loader_ERFNetv0'
model_arch_config.Expv2_REFID_FBquick_newEncoder.val_config_dataloader = 'loader_ERFNetv0'


# model_arch_config.Expv4_FB = dcopy(model_arch_config.Expv3)
model_arch_config.Expv4_FB_direct = dcopy(model_arch_config.Expv3)
model_arch_config.Expv4_withlargeConv = dcopy(model_arch_config.Expv3)
model_arch_config.Expv5 = dcopy(model_arch_config.Expv3)
model_arch_config.Expv4_withlargeConv_nmask = dcopy(model_arch_config.Expv3)
model_arch_config.Expv6 = dcopy(model_arch_config.Expv3)
model_arch_config.Expv7 = dcopy(model_arch_config.Expv3)
model_arch_config.Expv7_dataprop = dcopy(model_arch_config.Expv7)
model_arch_config.Expv7_datafuse = dcopy(model_arch_config.Expv7)
model_arch_config.Expv7_datafuseFeat = dcopy(model_arch_config.Expv7)
model_arch_config.Expv7_datafuseFeat_large = dcopy(model_arch_config.Expv7)
model_arch_config.Expv7_datafuseFeat_onlyEvents = dcopy(model_arch_config.Expv7)
model_arch_config.Expv7_datafuseFeat_direct = dcopy(model_arch_config.Expv7)
model_arch_config.Expv8 = dcopy(model_arch_config.Expv7)
model_arch_config.Expv8_Light = dcopy(model_arch_config.Expv7)
model_arch_config.Expv8_Lights2 = dcopy(model_arch_config.Expv7)
model_arch_config.Expv8_Lights2.train_config_dataloader = 'loader_ERFNetv0Mix'
model_arch_config.Expv8_Lights2.val_config_dataloader = 'loader_ERFNetv0Mix'
model_arch_config.Expv8_Lights2.define_model.num_decoder = 8

model_arch_config.Expv8_large = dcopy(model_arch_config.Expv7)
model_arch_config.Expv8_large.define_model.num_decoder = 8
model_arch_config.Expv8_large.train_config_dataloader = 'loader_ERFNetv0Mix'
model_arch_config.Expv8_large.val_config_dataloader = 'loader_ERFNetv0Mix'

model_arch_config.Expv8_Lights2fieldflow = dcopy(model_arch_config.Expv8_Lights2)




model_arch_config.Expv8_Lights3 = dcopy(model_arch_config.Expv8_Lights2)
model_arch_config.Expv9 = dcopy(model_arch_config.Expv7)
model_arch_config.Expv8_Lights2_onlyFlow = dcopy(model_arch_config.Expv7)

model_arch_config.TimeLens = ED()
model_arch_config.TimeLens.train_config_dataloader = 'loader_timelens_mix'
model_arch_config.TimeLens.val_config_dataloader = 'loader_timelens_mix'
model_arch_config.TimeLens.num_bins = 5
model_arch_config.TimeLens_flow = dcopy(model_arch_config.TimeLens)

model_arch_config.CBMNet = dcopy(model_arch_config.TimeLens)
model_arch_config.CBMNet.num_bins = 16
model_arch_config.myCBMNet = dcopy(model_arch_config.CBMNet)
model_arch_config.CBMNet_large = dcopy(model_arch_config.CBMNet)

model_arch_config.RGBGT = dcopy(model_arch_config.TimeLens)


# Ablation
model_arch_config.Expv8_Lights2fieldflownorefine_direct = dcopy(model_arch_config.Expv8_Lights2)
model_arch_config.Expv8_Lights2fieldflow_direct= dcopy(model_arch_config.Expv8_Lights2)
model_arch_config.Expv8_Lights2fieldflow= dcopy(model_arch_config.Expv8_Lights2)
model_arch_config.Expv8_Lights2norefine = dcopy(model_arch_config.Expv8_Lights2)
model_arch_config.Expv8_Lights3noGClip = dcopy(model_arch_config.Expv8_Lights3)