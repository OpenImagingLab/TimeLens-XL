import torch as th
import torch.nn.functional as F
import sys

from timelens import refine_warp_network, warp_network
from timelens.superslomo import unet
from .masknet import MaskNet

def _pack_input_for_attention_computation(example):
    fusion = example["middle"]["fusion"]
    number_of_examples, _, height, width = fusion.size()
    return th.cat(
        [
            example["after"]["flow"],
            example["middle"]["after_refined_warped"],
            example["before"]["flow"],
            example["middle"]["before_refined_warped"],
            example["middle"]["fusion"],
            th.Tensor(example["middle"]["weight"])
            .view(-1, 1, 1, 1)
            .expand(number_of_examples, 1, height, width)
            .type(fusion.type()),
        ],
        dim=1,
    )


def _compute_weighted_average(attention, before_refined, after_refined, fusion):
    return (
        attention[:, 0, ...].unsqueeze(1) * before_refined
        + attention[:, 1, ...].unsqueeze(1) * after_refined
        + attention[:, 2, ...].unsqueeze(1) * fusion
    )


class AttentionAverage(refine_warp_network.RefineWarp):
    def __init__(self):
        warp_network.Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)
        self.attention_network = unet.UNet(14, 3, False)
        self.masknet = MaskNet(17, 64)
        self.training_stage = None

    def run_fast(self, example):
        example['middle']['before_refined_warped'], \
        example['middle']['after_refined_warped'] = refine_warp_network.RefineWarp.run_fast(self, example)

        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example['middle']['before_refined_warped'],
            example['middle']['after_refined_warped'],
            example['middle']['fusion']
        )
        return average, attention

    def run_attention_averaging(self, example):
        refine_warp_network.RefineWarp.run_and_pack_to_example(self, example)
        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example["middle"]["before_refined_warped"],
            example["middle"]["after_refined_warped"],
            example["middle"]["fusion"],
        )
        return average, attention

    def run_warping(self, example):
        return refine_warp_network.RefineWarp.run_and_pack_to_example(self, example)

    # def run_fusion(self, example):


    def run_and_pack_to_example(self, example):
        (
            example["middle"]["attention_average"],
            example["middle"]["attention"],
        ) = self.run_attention_averaging(example)

    def forward(self, example):
        self.run_warping(example)
        res = self.masknet(
            example['before']['rgb_image_tensor'],
            example['after']['rgb_image_tensor'],
            example['middle']['before_warped'],
            example['middle']['after_warped'],
            example['middle']['weight'],
            example['before']['flow'],
            example['after']['flow']
        )
        example['middle']['flow_res'] = res
        return example

