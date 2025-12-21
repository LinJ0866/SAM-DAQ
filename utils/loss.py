import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask_DC_and_BCE_lossV2(nn.Module):
    def __init__(self, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_lossV2, self).__init__()

        self.ce =  torch.nn.BCELoss()
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        if net_output.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, net_output.shape[-2:], mode='bilinear')
        loss_ce = self.ce(net_output, target)
        # loss_dice = self.dc(net_output, target, sigmoid=False)
        # loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        loss = loss_ce
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss

def get_criterion():
    return Mask_DC_and_BCE_lossV2()