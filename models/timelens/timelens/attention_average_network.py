import torch as th
import torch.nn.functional as F
import sys
from collections import OrderedDict
from timelens import refine_warp_network, warp_network
from timelens.superslomo import unet

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
        self.training_stage = None
        self.pretrained_weights_dict = {}

    def load_pretrain_network(self):
        print("Current Training Stage: ", self.training_stage)
        # load weights of flow network
        weights = th.load(self.pretrained_weights_dict['flow_network'])['model_state']
        flownet_weights = OrderedDict()
        for k in self.flow_network.state_dict():
            flownet_weights.update({k:weights[f"net.flow_network.{k}"]})
        self.flow_network.load_state_dict(flownet_weights)
        for j, p in enumerate(self.flow_network.parameters()):
            p.requires_grad_(False)
        print("Timelens flow network weights loaded. Paramters are fixed")
        # load weights of fusion network
        weights = th.load(self.pretrained_weights_dict['fusion_network'])['model_state']
        fusionnet_weights = OrderedDict()
        for k in self.fusion_network.state_dict():
            fusionnet_weights.update({k: weights[f'net.fusion_network.{k}']})
        self.fusion_network.load_state_dict(fusionnet_weights)
        for j, p in enumerate(self.fusion_network.parameters()):
            p.requires_grad_(False)
        print("Timelens fusion network weights loaded. Paramters are fixed")
        if self.training_stage == 'attention':
            weights = th.load(self.pretrained_weights_dict['flow_refinement_network'])['model_state']
            flow_refinement_weights = OrderedDict()
            for k in self.flow_refinement_network.state_dict():
                flow_refinement_weights.update({
                    k: weights[f"net.flow_refinement_network.{k}"]
                })
            self.flow_refinement_network.load_state_dict(flow_refinement_weights)
            for j, p in enumerate(self.flow_refinement_network.parameters()):
                p.requires_grad_(False)
            print("---------------------- Timelens fusion network weights loaded. Paramters are fixed")
        return

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
        if self.training_stage == "tuning" or self.training_stage == 'attention':
            return self.run_attention_averaging(example)
        elif self.training_stage == 'warp' or self.training_stage == 'fusion' or self.training_stage == 'warp_refine':
            return self.run_warping(example)

