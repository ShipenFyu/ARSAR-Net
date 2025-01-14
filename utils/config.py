import numpy as np
import torch

# Variables of training
device_index = 'cuda:0'

# Configuration of SAR
C = 299792458

Lambda = 0.055517
SampleRate = 66.666667e6
PulseWidth = 45e-6

Kr = -1333333333333.333496
Fdc = 0
prf = 1420
Vs = 7500

Nslow = 512
Nfast = 512
Fc = C / Lambda
dR = C / SampleRate /2
Rmin = 850e3

Rgroup = Rmin + np.arange(Nfast) * dR
Rgroup = np.ones((Nslow, 1)) @ Rgroup.reshape(1, -1)
Rref = Rmin + np.floor(Nfast / 2) * dR

Trl = 2 * Rmin / C + np.arange(Nfast) / SampleRate
Trl = np.ones((Nslow, 1)) @ Trl.reshape(1, -1)
Fal = Fdc + (np.arange(-Nslow // 2, Nslow // 2)) / Nslow * prf
Fal = Fal.reshape(-1, 1) @ np.ones((1, Nfast))
Frl = (np.arange(-Nfast // 2, Nfast // 2)) / Nfast * SampleRate
Frl = np.ones((Nslow, 1)) @ Frl.reshape(1, -1)

D_f_eta = np.sqrt(1 - Lambda**2 * Fal**2 / 4 / Vs**2)
Km = Kr / (1 - Kr * C * Rgroup * Fal**2 / (2 * Vs**2 * Fc**3 * D_f_eta**3))
Dref = np.sqrt(1 - Lambda**2 * Fdc**2 / 4 / Vs**2)

sc = np.exp(1j * np.pi * Km * (Dref / D_f_eta - 1) * (Trl - 2 * Rref / C / D_f_eta)**2)
sc = torch.tensor(sc, dtype=torch.complex64)
sc = torch.div(sc, torch.abs(sc))
rc = np.exp(1j * np.pi * D_f_eta * Frl**2 / Km / Dref + 1j * 4 * np.pi * (1 / D_f_eta - 1 / Dref) * Rref * Frl / C)
rc = torch.tensor(rc, dtype=torch.complex64)
rc = torch.div(rc, torch.abs(rc))
ac = np.exp(1j * 4 * np.pi * Rgroup / Lambda * D_f_eta - 1j * 4 * np.pi * Km / C**2 * (1 - D_f_eta / Dref) * (Rgroup / D_f_eta - Rref / D_f_eta)**2)
ac = torch.tensor(ac, dtype=torch.complex64)
ac = torch.div(ac, torch.abs(ac))

processor = {'sc': sc, 'rc': rc, 'ac': ac}
