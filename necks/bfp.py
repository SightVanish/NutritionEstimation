import torch.nn as nn
import torch.nn.functional as F
import torch

def calculate_fixed_pool_params(input_size, output_size):
    # Calculate kernel size and stride to approximate the adaptive pooling output
    kernel_size = (input_size[0] // output_size[0], input_size[1] // output_size[1])
    stride = kernel_size
    return kernel_size, stride

class NonLocal2D(nn.Module):
    """ A Gaussian network that finds the correlation between features.
    Think of it as an attention for features.
    """
    def __init__(self, in_channels):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.conv_out = nn.Conv2d(self.in_channels, self.inter_channels, 1)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, HxW]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, 'embedded_gaussian')
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output

class BFP(nn.Module):
    """Balanced Feature Pyramid is a feature pyramid module.
    It differs from approaches like FPN that integrate multi-level features using lateral connections.
    Instead BFP strengthens the multi-level features using the same deeply integrated balanced semantic features.
    Consists 4 steps:
    1. rescaling: rescale images at all levels to an intermediate size (intepolation / max-pooling)
    2. integrating: mean (sum of all rescaled images divided by the number of levels)
    3. refining: A Non local 2D network
    4. strengthening: reverse back to the pyramid
    """
    def __init__(self, in_channels, num_levels, refine_level=1):
        super(BFP, self).__init__()

        self.in_channels = in_channels
        # NUmber of levels for the pyramid
        self.num_levels = num_levels
        self.refine_level = refine_level

        assert 0 < self.refine_level < num_levels

        self.refine = NonLocal2D(self.in_channels)

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: rescaling the images at all levels using interpolation and max-pooling
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                kernel_size, stride = calculate_fixed_pool_params(inputs[i].size()[2:], gather_size)
                # gathered = F.adaptive_max_pool2d(
                    # inputs[i], output_size=gather_size)
                gathered = F.max_pool2d(inputs[i], kernel_size=kernel_size, stride=stride)
                
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        # step 2: integrate, compute the mean
        bsf = sum(feats) / len(feats)

        # step 3: refine gathered features
        bsf = self.refine(bsf)

        # step 4: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                # residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
                kernel_size, stride = calculate_fixed_pool_params(bsf.size()[2:], out_size)
                residual = F.max_pool2d(bsf, kernel_size=kernel_size, stride=stride)

            outs.append(residual + inputs[i])

        return tuple(outs)


# if __name__ == '__main__':
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     p1 = torch.randn((8, 256, 64, 64), device=device)
#     p2 = torch.randn((8, 256, 32, 32), device=device)
#     p3 = torch.randn((8, 256, 16, 16), device=device)
#     p4 = torch.randn((8, 256, 8, 8), device=device)
#     fpn = tuple((p1, p2, p3, p4))
#     bfp = BFP(256, 4)
#     bfp.to(device)
#     results = bfp(fpn)
#     cat1, cat2, cat3, cat4 = results
#     print('debug------------')
