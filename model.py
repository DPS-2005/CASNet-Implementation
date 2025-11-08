import numpy as np
import pdb
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, num_features, bias=True, k_size=3):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, k_size, padding=k_size // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, k_size, padding=k_size // 2, bias=bias),
        )

    def forward(self, x):
        return x + self.body(x)

class SaliencyDetector(nn.Module):
    def __init__(self, img_features, mid_featres=3, num_res_blocks=3, k_size=3, bias=True):
        super(SaliencyDetector, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(img_features, mid_featres, k_size, padding=k_size // 2, bias=bias),
            *[ResidualBlock(mid_featres, bias) for _ in range(num_res_blocks)], 
            nn.Conv2d(mid_featres, 1, k_size, padding=k_size // 2, bias=bias)
        )
        
    def forward(self, x):
        return self.body(x)

def block_measurement_aggregation(saliency_map, block_size, cs_ratio, max_iter=10):
    b, c, h, w = saliency_map.shape
    N = block_size * block_size
    num_blocks = (h // block_size)*(w // block_size)
    avg_measurement = round(cs_ratio * N)
    # pdb.set_trace()
    S = saliency_map.flatten(2).softmax(dim=2).reshape(*saliency_map.shape)
    Q = nn.functional.unfold(S, block_size, stride=block_size).sum(dim=1).unsqueeze(dim=1) # shape b, 1, l
    Q = avg_measurement * num_blocks * Q
    Q = Q.float()
    i = 0
    while True:
        i += 1
        Q = torch.clamp(Q, 0, N).round()
        error = (Q.mean() - avg_measurement).item()
        if error == 0:
            break
        elif i <= max_iter:
            Q -= error
        else:
            if(abs(error) < 0.5):
                break
            rand_distribution = torch.multinomial(torch.ones(num_blocks, device=Q.device) / num_blocks, round(abs(error)), True)
            rand_distribution = torch.bincount(rand_distribution, minlength=num_blocks).float()
            Q -= rand_distribution.reshape(Q.shape) * error/(abs(error))
    return Q.int()

class Extractor(nn.Module):
    def __init__(self, mid_features=8, out_features=3, num_res_blocks=3, bias=True):
        super(Extractor, self).__init__()
        self.out_features = out_features
        self.body = nn.Sequential(
            nn.Conv2d(1, mid_features, 1, bias=bias),
            *[ResidualBlock(mid_features, bias, 1) for _ in range(num_res_blocks)],
            nn.Conv2d(mid_features, out_features, 1, bias=bias )
        )

    def forward(self, x):
        return self.body(x)
    
class UNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(UNet, self).__init__()
        num_features = [in_features, 16, 32, 64, 128]
        k_size, scale, bias, num_res_blocks = 3, 2, False, 2
        def encoder_block(in_features, out_features):
            # default size x -> x//2
            return nn.Sequential(
                nn.Conv2d(in_features, out_features, k_size, padding=k_size // 2, bias=bias),
                *[ResidualBlock(out_features, bias, k_size) for _ in range(num_res_blocks)],
                nn.Conv2d(out_features, out_features, kernel_size=scale, stride=scale, bias=bias) # sconv
            )
        
        def decoder_block(in_features, out_features):
            # default size x -> 2x
            return nn.Sequential(
                nn.ConvTranspose2d(in_features, in_features, kernel_size=scale, stride=scale, bias=bias), # tconv
                *[ResidualBlock(in_features, bias, k_size) for _ in range(num_res_blocks)],
                nn.Conv2d(in_features, out_features, k_size, padding=k_size // 2, bias=bias)
            )
        
        self.encoder = nn.ModuleList([encoder_block(num_features[i], num_features[i+1]) for i in range(len(num_features)-1)])
        num_features[0] = out_features
        self.bottleneck = ResidualBlock(num_features[len(num_features)-1])
        self.decoder = nn.ModuleList([decoder_block(num_features[i], num_features[i-1]) for i in range(len(num_features)-1, 0, -1)])
    
    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for dec, skip in zip(self.decoder, skip_connections):
            x = dec(x+skip)
        return x

class Phase(nn.Module):
    def __init__(self, img_features, block_size):
        super(Phase, self).__init__()
        self.extractor = Extractor()
        self.unet = UNet(self.extractor.out_features+img_features, img_features)
        self.rho = nn.Parameter(torch.Tensor([0.5]))
        self.block_size = block_size
    def forward(self, x, y, cs_ratio_map, phi_T, phi_T_phi, shape):
        # print(f"target: {shape}")
        b, c, h, w = shape
        z_unfold : torch.Tensor = x - self.rho * (phi_T_phi.matmul(x) - phi_T.matmul(y))
        expanded_ratio_map = torch.repeat_interleave(cs_ratio_map, self.block_size, 2).repeat_interleave(self.block_size, 3)
        # pdb.set_trace()
        saliency_feature = self.extractor(expanded_ratio_map)
        # print(f"z_unfold shape: {z_unfold.shape}")
        l = z_unfold.shape[0]//b
        N = self.block_size * self.block_size
        z_unfold = z_unfold.permute(0, 2, 1).reshape(b, l, c, N).permute(0, 2, 3, 1)  # (b, c, l, N)
        z_unfold = z_unfold.reshape(b, c*N, l) # (b, c*N, l)
        z = nn.functional.fold(z_unfold, (h, w), self.block_size, stride=self.block_size)
        z += self.unet(torch.cat((z, saliency_feature), 1))
        return z


class CASNet(nn.Module):
    def __init__(self, block_size, img_features, n_phase = 8, gamma = 0.2882):
        super(CASNet, self).__init__()
        self.block_size = block_size
        self.N = block_size * block_size
        self.n_phase = n_phase
        self.gamma = gamma # sampling proportion
        self.generator = nn.Parameter(torch.randn(self.N, self.N) / self.block_size)
        self.saliency_detector = SaliencyDetector(img_features)
        self.recoverySubnet = Phase(img_features, block_size)

    def _unfold(self, x):
        b, c, h, w = x.shape
        x_unfold = nn.functional.unfold(x, self.block_size, stride=self.block_size) # shape b, c*N, l
        l = x_unfold.shape[2]
        x_unfold = x_unfold.permute(0,2,1).reshape(b*l, self.N, c) # shape b*l, N, c
        return l, x_unfold

    def forward(self, x, cs_ratio, test=False):
        # unfold -> sample -> initialise -> phase by phase recovery -> fold
        # x dimension is b,c,h,w
        # Q dimension is b,1,l
        # generator dimension is N,N
        b, c, h, w = x.shape
        N = self.block_size * self.block_size
        qb = round(cs_ratio * self.gamma * N) # uniform measurement per block
        l, x_unfold = self._unfold(x)
        phi  = self.generator.unsqueeze(0).repeat(b*l, 1, 1)
        row_idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(b*l, 1)
        u_mask  = row_idx < qb
        u_mask = u_mask.unsqueeze(2).repeat(1,1, N).float()
        u_phi = u_mask*phi
        y = u_phi.matmul(x_unfold)
        saliency_map = self.saliency_detector(x)
        Q = block_measurement_aggregation(saliency_map, self.block_size, cs_ratio)
        cs_ratio_map = Q.reshape(b, 1, h // self.block_size, w // self.block_size) / self.N
        Q = Q.permute(0,2,1).reshape(b*l, 1)
        r_mask = (row_idx >= qb) & (row_idx < Q)
        r_mask = r_mask.unsqueeze(2).repeat(1,1, N).float()
        r_phi = r_mask*phi
        phi = u_phi + r_phi
        phi_T = phi.permute(0,2,1)
        phi_T_phi = phi_T.matmul(phi)
        y = phi.matmul(x_unfold)

        if test:
            y = phi_T.matmul(y)
            x0 = phi_T.matmul(y)
            y = y.permute(0, 2, 1).reshape(b, l, c, N).permute(0, 2, 3, 1).reshape(b, c*N, l)
            y = nn.functional.fold(y, (h, w), self.block_size, stride=self.block_size)
            x0 = x0.permute(0, 2, 1).reshape(b, l, c, N).permute(0, 2, 3, 1).reshape(b, c*N, l)
            x0 = nn.functional.fold(x0, (h, w), self.block_size, stride=self.block_size)
            return saliency_map, y, x0
        for _ in range(self.n_phase):
            x = self.recoverySubnet(x_unfold, y, cs_ratio_map, phi_T, phi_T_phi, x.shape)
            _, x_unfold = self._unfold(x)
        return x