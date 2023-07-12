import ROOT
import numpy as np

f = ROOT.TFile.Open("B4_pim.root")
nt = f.Get("B4")

rdf = ROOT.RDataFrame(nt)
dat = rdf.AsNumpy()

de_abs_data = dat['Eabs']
de_gap_data = dat['Egap']
p_data = dat['P']
theta_data = dat['theta']

np.save('pim_mom.npy',p_data)
np.save('pim_theta.npy',theta_data)
np.save('pim_de_abs.npy',de_abs_data)
np.save('pim_de_gap.npy',de_gap_data)




