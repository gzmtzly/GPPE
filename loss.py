# @Time    : 12/4/22 10:51 PM
# @Author  : Zhou-Lin-yong
# @File    : loss.py
# @SoftWare: PyCharm
import torch.nn as nn
import torch

class GPPE(nn.Module):
    def __init__(self, alpha_neg=-2, beta_neg=6, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(GPPE, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.alpha_neg = alpha_neg
        self.beta_neg = beta_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, epoch):
        """"
        Parameters
        ----------
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size, 1)
        """
        # Calculating Probabilities
        log_preds = self.logsoftmax(x)
        x_pred = torch.exp(log_preds)

        xs_pos = x_pred[:, 1].view(x.shape[0], 1)
        xs_neg_before = 1 - xs_pos
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))

        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.alpha_neg > 0 or self.beta_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg_before * (1 - y)
            pt = pt0 + pt1

            one_sided_gamma = self.gamma_pos * y + (self.alpha_neg * (1 - pt) + self.beta_neg) * (1 - y)

            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
            loss = loss.sum(dim=-1)

        return -loss.mean()