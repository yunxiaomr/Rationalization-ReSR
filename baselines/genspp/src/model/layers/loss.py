import torch


class CategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_true, y_pred):

        ce = self.ce_loss(y_pred, y_true)
        return torch.mean(ce)


class SparsityLoss(torch.nn.Module):
    def forward(self, original_mask, highlight_mask):

        total_count = torch.sum(original_mask, dim=-1).clamp_min(1.0)

        presence_count = torch.sum(highlight_mask, dim=-1)

        sparsity = presence_count / total_count
        sparsity = torch.mean(sparsity)

        return sparsity


class ContiguityLoss(torch.nn.Module):
    def forward(self, original_mask, highlight_mask):

        left_padding = torch.nn.functional.pad(highlight_mask, pad=(1, 0, 0, 0))
        right_padding = torch.nn.functional.pad(highlight_mask, pad=(0, 1, 0, 0))

        switches = torch.logical_xor(left_padding, right_padding)
        positive_switches = switches * right_padding

        n_switches = torch.sum(positive_switches, dim=-1)
        total_count = torch.sum(original_mask, dim=-1).clamp_min(1.0)

        contiguity = n_switches / total_count
        contiguity = torch.mean(contiguity)

        return contiguity


class ConfidenceLoss(torch.nn.Module):
    def forward(self, original_mask, soft_mask):

        errors = torch.abs(soft_mask - 0.5) * 2
        confidence_sum = torch.sum(errors)

        false_errors = torch.sum(torch.abs(original_mask - 1.0))
        confidence_sum = confidence_sum - false_errors

        total_count = torch.sum(original_mask).clamp_min(1.0)
        confidence = confidence_sum / total_count

        return confidence
