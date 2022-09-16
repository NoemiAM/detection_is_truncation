import swyft.lightning as sl
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import swyft
from dataclasses import dataclass
from utils import *
from icecream import ic


@dataclass
class ToyFermiBound:
    detected: torch.Tensor = torch.zeros((0, 3))
    threshold: float = np.inf
    

class ToyFermi(sl.Simulator):
    def __init__(self, bounds = ToyFermiBound(), seed = None, npix = 256, psc_range = [0, np.inf]):
        super().__init__()
        self.bounds = bounds
        l = np.linspace(-40, 40, npix)
        b = np.linspace(-40, 40, npix)
        self.npix = npix
        self.L, self.B = np.meshgrid(l, b)
        self.seed = seed
        self.psc_range = psc_range
        self.on_after_forward = sl.to_numpy32
        
    def psc_disc_pop(self):
        return np.array([
                np.random.uniform(10, 500.),
                np.random.uniform(1., 3.),
                np.random.uniform(1., 20.),
            ])
        
    def psc_disc_cat(self, N = 100, sigma = 1.0, height = 3):
        DL, DB = 20, height
        N = int(N)
        L = torch.randn(N)*DL
        B = torch.randn(N)*DB
        F = torch.tensor(np.random.lognormal(1.0, sigma, size = N)).float()
        psc_list = torch.stack([B, L, F], 1)
        return var2fix(psc_list, 1000)
    
    def psc_split_disc(self, psc_disc_cat):
        psc_disc_cat = fix2var(psc_disc_cat)
        mask = psc_disc_cat[:,2] < self.bounds.threshold
        A = psc_disc_cat[mask]
        B = psc_disc_cat[~mask]
        return var2fix(A, 1000), var2fix(B, 1000)

    def mu_psc(self, psc_disc_cat):
        psc_disc_cat = fix2var(psc_disc_cat)
        v = psc_disc_cat[:,:2]
        w = psc_disc_cat[:,2]
        img = points_to_image(v, w, [self.npix, self.npix], [[-40, 40], [-40, 40]])
        return img

    def apply_psf(self, mu):
        mu_psf = gaussian_filter(mu, 1.5)*4
        return mu_psf

    def noise(self, mu):
        return np.random.poisson(mu*100)/100
    
    def psc_disc_cat_cnstr(self, pdcu, pdcd):
        pdcu = fix2var(pdcu)
        pdcd = fix2var(pdcd)
        return var2fix(torch.cat([pdcd, pdcu]), 1000)
    
    def psc_disc_cat_det(self):
        det = self.bounds.detected.clone()
        det[:, :2] += (torch.rand_like(det[:, :2])-0.5)*2. # random position around detected one
        f = np.random.lognormal(1.0, 1.5, size = 10000)
        f = np.random.choice(f[f>self.bounds.threshold], size = len(det))
        det[:, 2] = torch.tensor(f).float() # random flux
        return var2fix(det, 1000)
    
    def forward(self, trace):
        # Generate point source population
        pdp = trace.sample('pdp', self.psc_disc_pop)
        pdc = trace.sample('pdc', lambda _: 
                                    self.psc_disc_cat(_[0], _[1], _[2]), pdp)
        
        # Split disc source popluation in resolved and unresolved component
        pdcu, pdcr = trace.sample(['pdcu', 'pdcr'], self.psc_split_disc, pdc)
        
        # Detected population
        pdcd = trace.sample('pdcd', self.psc_disc_cat_det)
        
        # Combine unresolved and detected population
        pdcc = trace.sample('pdcc', self.psc_disc_cat_cnstr, pdcu, pdcd)
        
        mu_psc = trace.sample("mu_psc", self.mu_psc, pdcc)
        mu = trace.sample("mu", self.apply_psf, mu_psc)
        data = trace.sample("data", self.noise, mu)
        

class ToyFermi_d(sl.Simulator):
    def __init__(self, bounds = ToyFermiBound(), seed = None, npix = 256, psc_range = [0, np.inf], network_maps=None):
        super().__init__()
        self.bounds = bounds
        l = np.linspace(-40, 40, npix)
        b = np.linspace(-40, 40, npix)
        self.npix = npix
        self.L, self.B = np.meshgrid(l, b)
        self.seed = seed
        self.psc_range = psc_range
        self.on_after_forward = sl.to_numpy32
        self.network_maps = network_maps
        
    def psc_disc_pop(self):
        return np.array([
                np.random.uniform(10, 500.),
                np.random.uniform(1., 3.),
                np.random.uniform(1., 20.),
            ])
        
    def psc_disc_cat(self, N = 100, sigma = 1.0, height = 3):
        DL, DB = 20, height
        N = int(N)
        L = torch.randn(N)*DL
        B = torch.randn(N)*DB
        F = torch.tensor(np.random.lognormal(1.0, sigma, size = N)).float()
        psc_list = torch.stack([B, L, F], 1)
        return var2fix(psc_list, 1000)
    
    def psc_split_disc(self, psc_disc_cat):
        psc_disc_cat = fix2var(psc_disc_cat)
        mask = psc_disc_cat[:,2] < self.bounds.threshold
        A = psc_disc_cat[mask]
        B = psc_disc_cat[~mask]
        return var2fix(A, 1000), var2fix(B, 1000)

    def mu_psc(self, psc_disc_cat):
        psc_disc_cat = fix2var(psc_disc_cat)
        v = psc_disc_cat[:,:2]
        w = psc_disc_cat[:,2]
        img = points_to_image(v, w, [self.npix, self.npix], [[-40, 40], [-40, 40]])
        return img

    def apply_psf(self, mu):
        mu_psf = gaussian_filter(mu, 1.5)*4
        return mu_psf

    def noise(self, mu):
        return np.random.poisson(mu*100)/100
    
    def psc_disc_cat_cnstr(self, pdcu, pdcd):
        pdcu = fix2var(pdcu)
        pdcd = fix2var(pdcd)
        return var2fix(torch.cat([pdcd, pdcu]), 1000)
    
    def psc_disc_cat_det(self):
        det = self.bounds.detected.clone()
        det[:, :2] += (torch.rand_like(det[:, :2])-0.5)*2. # random position around detected one
        f = np.random.lognormal(1.0, 1.5, size = 10000)
        f = np.random.choice(f[f>self.bounds.threshold], size = len(det))
        det[:, 2] = torch.tensor(f).float() # random flux
        return var2fix(det, 1000)
    
    def detection_d(self, data, pdcc):
        x = swyft.Samples(data=torch.from_numpy(data).float().unsqueeze(0), pdcc=pdcc.float().unsqueeze(0))
        maps = self.network_maps(x, x)['aux_psc'].detach().clone()
        maps = (maps[...] > 5) # r_1 > 5
        maps = maps.sum(-1) 
        maps = maps > 0

        # Get indices from source list to get sensitivity value
        pdcc = fix2var(pdcc) # remove padded values
        v = torch.tensor(pdcc[:, :2])
        w = torch.tensor(pdcc[:, 2])
        d = []

        mu = points_to_image(v.unsqueeze(1), w.unsqueeze(1), [128, 128], [[-40, 40], [-40, 40]])
        mu = mu > 0
        d = []
        # loop over each source to avoid loosing index for sources in the same pixel
        # TO DO: can be smarter and faster
        for i in range(len(v)): 
            if (mu[i] == True).any():
                idxs_mu = (mu[i] == True).nonzero(as_tuple=False)[0]
                d.append(maps[[0, idxs_mu[0], idxs_mu[1]]].int())
            else:
                d.append(torch.tensor(0, dtype=torch.int32))
        d = torch.stack(d)
        return var2fix(d, 1000, fill=0)
        
    def forward(self, trace):
        # Generate point source population
        pdp = trace.sample('pdp', self.psc_disc_pop)
        pdc = trace.sample('pdc', lambda _: 
                                    self.psc_disc_cat(_[0], _[1], _[2]), pdp)
        
        # Split disc source popluation in resolved and unresolved component
        pdcu, pdcr = trace.sample(['pdcu', 'pdcr'], self.psc_split_disc, pdc)
        
        # Detected population
        pdcd = trace.sample('pdcd', self.psc_disc_cat_det)
        
        # Combine unresolved and detected population
        pdcc = trace.sample('pdcc', self.psc_disc_cat_cnstr, pdcu, pdcd)
        
        mu_psc = trace.sample("mu_psc", self.mu_psc, pdcc)
        mu = trace.sample("mu", self.apply_psf, mu_psc)
        data = trace.sample("data", self.noise, mu)
        
        # Compute detection sensitivity
        d = trace.sample('d', self.detection_d, data, pdc)

        
class ToyFermiSensitivity(sl.Simulator):
    def __init__(self, bounds = ToyFermiBound(), seed = None, npix = 256, psc_range = [0, np.inf], sensitivity_function = None):
        super().__init__()
        self.bounds = bounds
        l = np.linspace(-40, 40, npix)
        b = np.linspace(-40, 40, npix)
        self.npix = npix
        self.L, self.B = np.meshgrid(l, b)
        self.seed = seed
        self.psc_range = psc_range
        self.sensitivity_function = sensitivity_function
        self.on_after_forward = sl.to_numpy32
        
    def psc_disc_pop(self):
        return np.array([
                np.random.uniform(10, 500.),
                np.random.uniform(1., 3.),
                np.random.uniform(1., 20.),
            ])
        
    def psc_disc_cat(self, N = 100, sigma = 1.0, height = 3):
        DL, DB = 20, height
        N = int(N)
        L = torch.randn(N)*DL
        B = torch.randn(N)*DB
        F = torch.tensor(np.random.lognormal(1.0, sigma, size = N)).float()
        psc_list = torch.stack([B, L, F], 1)
        return var2fix(psc_list, 1000)
    
    def detection_sensitivity(self, pdc):
        pdc = fix2var(pdc)
        prior_d1 = 0.1573
        prior_d0 = 0.8427
        pdc[..., 2] = np.log10(pdc[..., 2])
        samples = swyft.Samples({'source': pdc, 'd_source': torch.ones((len(pdc),1))*1.})
        r2_1 = self.sensitivity_function(samples, samples)['re_d'].logratios.detach().exp()
        samples = swyft.Samples({'source': pdc, 'd_source': torch.zeros((len(pdc),1))*1.})
        r2_0 = self.sensitivity_function(samples, samples)['re_d'].logratios.detach().exp()
        s = torch.sigmoid(((prior_d1*r2_1)/(prior_d0*r2_0)).log())
        return var2fix(s, 1000)
        
    def detection_d(self, s):
        s = fix2var(s)
        d = torch.bernoulli(s)
        return var2fix(d, 1000)
    
    def psc_split_disc(self, psc_disc_cat, d):
        psc_disc_cat = fix2var(psc_disc_cat)
        d = fix2var(d)
        mask = d.bool().squeeze(1)
        A = psc_disc_cat[mask, ...]
        B = psc_disc_cat[~mask, ...]
        return var2fix(A, 1000), var2fix(B, 1000)
    
    def psc_disc_cat_det(self, pdp):
        det = self.bounds.detected.clone()
        det[:, :2] += (torch.rand_like(det[:, :2])-0.5)*2. # random position around detected one
        f = np.random.lognormal(1.0, pdp[1], size = 10000)
        f = np.random.choice(f[f>self.bounds.threshold], size = len(det))
        det[:, 2] = torch.tensor(f).float() # random flux
        return var2fix(det, 1000)
    
    def psc_disc_cat_cnstr(self, pdcu, pdcd):
        pdcu = fix2var(pdcu)
        pdcd = fix2var(pdcd)
        return var2fix(torch.cat([pdcd, pdcu]), 1000)

    def mu_psc(self, psc_disc_cat):
        psc_disc_cat = fix2var(psc_disc_cat)
        v = psc_disc_cat[:,:2]
        w = psc_disc_cat[:,2]
        img = points_to_image(v, w, [self.npix, self.npix], [[-40, 40], [-40, 40]])
        return img

    def apply_psf(self, mu):
        mu_psf = gaussian_filter(mu, 1.5)*4
        return mu_psf

    def noise(self, mu):
        return np.random.poisson(mu*100)/100
        
    def forward(self, trace):
        # Generate point source population
        pdp = trace.sample('pdp', self.psc_disc_pop)
        pdc = trace.sample('pdc', lambda _: 
                                    self.psc_disc_cat(_[0], _[1], _[2]), pdp)
        
        # Compute detection sensitivity and parameter d
        s = trace.sample('s', self.detection_sensitivity, pdc)
        d = trace.sample('d', self.detection_d, s)
        
        # Split disc source popluation in resolved and unresolved component
        pdcr, pdcu = trace.sample(['pdcr', 'pdcu'], self.psc_split_disc, pdc, d)
        
        # Detected population
        pdcd = trace.sample('pdcd', self.psc_disc_cat_det, pdp)
        
        # Combine unresolved and detected population
        pdcc = trace.sample('pdcc', self.psc_disc_cat_cnstr, pdcu, pdcd)
        
        mu_psc = trace.sample("mu_psc", self.mu_psc, pdcc)
        mu = trace.sample("mu", self.apply_psf, mu_psc)
        data = trace.sample("data", self.noise, mu)