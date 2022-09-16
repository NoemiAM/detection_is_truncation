import torch
import swyft
import swyft.lightning as sl
import sims
import numpy as np
from utils import detect_peaks
from icecream import ic


### Detection maps

class RatioEstimatorCube(torch.nn.Module):
    def __init__(self, shapes, ranges, cumsum=False):
        super().__init__()
        self.shapes = shapes
        self.ranges = ranges
        self.cumsum = cumsum
        
    def forward(self, x, z):
        """
        Args:
            x: cube
            z: list
        """
        
        counts = sims.points_to_image(z, None, self.shapes, self.ranges)
        if self.cumsum:
            counts = counts + counts.sum(dim=-1, keepdims = True) - torch.cumsum(counts, dim=-1) # Count sources >= threshold
        mask = (counts>0).long()
        r, mask = sl.equalize_tensors(x, mask)
        r_masked = r*mask
        ratios = sl.LogRatioSamples(params = counts, logratios = r_masked, parnames=np.array(['pdcc']))
        return ratios
    
    
class NetworkMaps(swyft.SwyftModule):
    """
    Network for detection maps
    """
    def __init__(self, lr=1e-3, lrs_factor = 0.1, lrs_patience = 3, early_stopping_patience = 5):
        super().__init__(lr=lr, lrs_factor=lrs_factor, early_stopping_patience=early_stopping_patience, lrs_patience=lrs_patience)

        self.online_z_score = swyft.networks.OnlineStandardizingLayer(shape = (128, 128))
        
        self.CNN1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.LazyConv2d(8, 7, padding=3), torch.nn.LeakyReLU(),
                torch.nn.LazyConv2d(128, 1, padding=0), torch.nn.LeakyReLU(),
                torch.nn.LazyConv2d(1, 1, padding=0)
            ) for _ in range(12)])
        self.re1 = RatioEstimatorCube([128, 128, 12], [[-40, 40], [-40, 40], [-1, 2]], cumsum = True)
    
    def forward(self, x, z):
        
        data = x['data']*0.1
        
        # estimate luminosity ratio
        f2d = [self.CNN1[i](data.unsqueeze(1)).squeeze(1).unsqueeze(-1) for i in range(12)]
        f2dl = torch.cat(f2d, dim = -1)
        v_pos = z['pdcc'][..., :2]
        v_lum = z['pdcc'][..., 2]
        v_pos, v_lum = sl.equalize_tensors(v_pos, v_lum)
        v = torch.cat([v_pos, v_lum.unsqueeze(-1)], dim=-1)
        v[v == -torch.inf] = 0.
        v[:,:,2] = torch.log10(v[:,:,2]+1e-10)
        ratios = self.re1(f2dl, v)

        return dict(psc_lum = ratios, aux_psc = f2dl)
    
    
###### Sensitivity function

class NetworkSensitivity(swyft.SwyftModule):
    """
    Network for sensitivity function
    """
    def __init__(self, lr=1e-3, lrs_factor = 0.1, lrs_patience = 3, early_stopping_patience = 5):
        super().__init__(lr=lr, lrs_factor=lrs_factor, early_stopping_patience=early_stopping_patience, lrs_patience=lrs_patience)
    
        self.re_d = swyft.LogRatioEstimator_1dim(num_features = 3, num_params = 1, 
                                                 hidden_features = 64, num_blocks = 2, 
                                                 dropout=0.2, use_batch_norm=False,
                                                 varnames = 'd')
        
    def forward(self, x, z):
        # ic(x['d_source'].sum(), x['source'])
        re_d = self.re_d(x['source'], z['d_source'])   
        return dict(re_d=re_d)
    
    
###### Network for population parameters given map

class NetworkParamMap(swyft.SwyftModule):

    def __init__(self, lr=1e-3, lrs_factor = 0.1, lrs_patience = 3, early_stopping_patience = 5, network_maps=None):
        super().__init__(lr=lr, lrs_factor=lrs_factor, early_stopping_patience=early_stopping_patience, lrs_patience=lrs_patience)
        
        self.n_maps = 12
        self.maps = network_maps
        self.peaks = detect_peaks
    
        self.re_data = swyft.LogRatioEstimator_1dim(num_features = self.n_maps, num_params = 3, varnames = 'pdp', dropout=0.2)
        marginals = ((0, 1), (0, 2), (1, 2))
        self.re_data2 = swyft.LogRatioEstimator_Ndim(num_features = self.n_maps, marginals = marginals, varnames = 'pdp', dropout=0.2)
   

    def forward(self, x, z):
                
        # estimate exclusion maps
        maps = self.maps(x, x)['aux_psc'].detach().clone()
        maps[maps < 1e-10] = 0
        maps = torch.movedim(maps, -1, -3)
        
        # estimate peaks
        peaks = self.peaks(maps)
        #estimate counts
        counts = peaks.sum((-2, -1))
    
        # ratio estimators
        re_param = self.re_data(counts, z['pdp'])
        re_param2 = self.re_data2(counts, z['pdp'])

        return dict(re_param=re_param, re_param2=re_param2)

    
###### Network for population parameters given list

class NetworkParamList(swyft.SwyftModule):

    def __init__(self, lr=1e-3, lrs_factor = 0.1, lrs_patience = 3, early_stopping_patience = 5):
        super().__init__(lr=lr, lrs_factor=lrs_factor, early_stopping_patience=early_stopping_patience, lrs_patience=lrs_patience)
        
        self.online_z_score_flux = swyft.networks.OnlineStandardizingLayer(shape = (500,))
        self.online_z_score_pos = swyft.networks.OnlineStandardizingLayer(shape = (1000,))
        self.summarize_flux = swyft.networks.channelized.ResidualNetWithChannel(
            channels = 1,
            in_features = 500, 
            out_features = 16,
            hidden_features = 64,
            num_blocks = 2,
        )
        self.summarize_position = swyft.networks.channelized.ResidualNetWithChannel(
            channels = 1,
            in_features = 1000, 
            out_features = 16,
            hidden_features = 64,
            num_blocks = 2,
        )
    
        self.re_data = swyft.LogRatioEstimator_1dim(num_features = 32, num_params = 3, varnames = 'pdp', dropout=0.2)
        marginals = ((0, 1), (0, 2), (1, 2))
        self.re_data2 = swyft.LogRatioEstimator_Ndim(num_features = 32, marginals = marginals, varnames = 'pdp', dropout=0.2)
   

    def forward(self, x, z):
        
        flux_simulated = x['pdcr'][..., 2] # resolved simulated flux list
        flux_simulated[flux_simulated == -torch.inf] = 0
        flux_simulated = (torch.einsum('iik->ik', flux_simulated[:, flux_simulated.argsort()])) # sort based on luminosity
        flux_simulated = flux_simulated[..., 500:]
        flux_simulated = self.online_z_score_flux(flux_simulated)
        f_flux = self.summarize_flux(flux_simulated)
        
        pos_simulated = x['pdcr'][..., 0] # resolved simulated b position list
        pos_simulated[pos_simulated == -torch.inf] = 0
        pos_simulated = (torch.einsum('iik->ik', pos_simulated[:, pos_simulated.argsort()])) # sort
        pos_simulated = self.online_z_score_pos(pos_simulated)
        f_pos = self.summarize_position(pos_simulated)
        
        f = torch.cat((f_flux, f_pos), dim = -1)

        # ratio estimators
        re_param = self.re_data(f, z['pdp'])
        re_param2 = self.re_data2(f, z['pdp'])

        return dict(re_param=re_param, re_param2=re_param2)

