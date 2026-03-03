import torch
import torch.nn as nn

class FeaturesToDepth(torch.nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=80,
        bins_strategy="linear",
        norm_strategy="linear",
    ):
        """
        Module which converts a feature maps into a depth map

        Args:
        min_depth (float): minimum depth, used to calibrate the depth range
        max_depth (float): maximum depth, used to calibrate the depth range
        bins_strategy (str): Choices are 'linear' or 'log', for Uniform or Scale Invariant distributions for depth bins.
                             See AdaBins [1] for more details.
        norm_strategy (str): Choices are 'linear', 'softmax' or 'sigmoid', for the conversion of features to depth logits
        scale_up (bool): If true, and only if regression by classification is not used, the result is multiplied by max_depth


        Example:
        x = depth_model(input_image)  # N C H W
        - If pure regression (C == 1), depth is obtained by scaling and/or shifting x
        - If C > 1, bins are used:
            Depth is obtained as a weighted sum of depth bins, where weights are predicted logits. (see AdaBins [1] for more details)

        [1] AdaBins: https://github.com/shariqfarooq123/AdaBins
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert bins_strategy in ["linear", "log"], "Support bins_strategy: linear, log"
        assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x):
        n_bins = x.shape[1]  # N n_bins H W
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                    device=x.device,
                )
                bins = torch.exp(bins)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(x)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(x, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(x)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        else:
            # standard regression
            output = torch.relu(x) + self.min_depth
        return output