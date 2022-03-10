#
#	2bodylcfit.py
#
#	Built specifically to fit Haumea and Hiiaka's light curves by fitting sinusoidal shapes
#	to them.
#
#	Benjamin Proudfoot
#	03/07/21
#


import emcee
import numpy as np
import scipy
import corner
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from clustering import *
import corner
from astroquery.jplhorizons import Horizons

def likelihood(params, obsdf, synth = False):
	# unpack params
	a00, a01, phi00, phi01, a10, a11, a12, phi10, phi11, phi12, offset0 = params
	p0 = 3.915341
	p0 = p0/24
	p1 = 9.79736
	p1 = p1/24
	t0 = 2459281

	times = obsdf["JD_UTC"].values.flatten()
	
	residuals = np.zeros(times.size)
	flux_haumea = a00*np.sin(2*np.pi*(phi00 + 2*(times-t0)/p0)) + a01*np.sin(2*np.pi*(phi01 + (times-t0)/p0))
	flux_hiiaka = a10*np.sin(2*np.pi*(phi10 + 2*(times-t0)/p1)) + a11*np.sin(2*np.pi*(phi11 + (times-t0)/p1)) + a12*np.sin(2*np.pi*(phi12 + 3*(times-t0)/p1))
	flux_total = flux_haumea + flux_hiiaka + offset0

	residuals = (obsdf["Source-Sky_T1"] - flux_total)/(obsdf["Source_Error_T1"])

	if synth:
		return flux_total, residuals, flux_haumea, flux_hiiaka

	chisq = np.nansum(residuals**2)

	return chisq


def priors(params):
	# apply flat priors to the data...
	a00, a01, phi00, phi01, a10, a11, a12, phi10, phi11, phi12, offset0 = params
	p1 = 9.8
	#if p0 <= 0 or p1 <= 0:
	#	return -np.inf
	if a00 <= 0 or a01 <= 0 or a10 <= 0 or a11 <= 0 or a12 <= 0:
		return -np.inf
	elif phi00 <= 0 or phi01 <= 0 or phi10 <= 0 or phi11 <= 0 or phi12 <= 0:
		return -np.inf
	elif phi00 >= 1 or phi01 >= 1 or phi10 >= 1 or phi11 >= 1 or phi12 >= 1:
		return -np.inf
	elif offset0 <= 0:
		return -np.inf
	elif a00 <= a01:
		return -np.inf
#	elif a10 <= a11 or a10 <= a12:
#		return -np.inf
	elif 0.5*a00 <= a10:
		return -np.inf
	else:
		return 0

def probability(params, obsdf):
	llhood = -0.5*likelihood(params, obsdf)
	prior = priors(params)
	prob = llhood + prior
	#print(prob)
	return prob

#############################################################################################

# Starting actual code

nwalkers = 500
nburnin = 1000
nsample = 1000

obsdf = pd.read_csv("r_band.csv")
nobs = obsdf["JD_UTC"].values.size

times = obsdf["JD_UTC"].values.flatten()

ourKBO = Horizons(id="Haumea",location=705,epochs = times)

lighttime = ourKBO.vectors()['lighttime']
correctedtime = times-lighttime

obsdf["JD_UTC"] = correctedtime

# draw initial guess
#p0_0 = np.random.normal(loc = 3.915341, scale = 0.0001, size = nwalkers)
a00_0 = np.random.normal(loc = 0.09659, scale = 0.001, size = nwalkers)
a01_0 = np.random.normal(loc = 0.026, scale = 0.001, size = nwalkers)
phi00_0 = np.random.uniform(size = nwalkers)
phi01_0 = np.random.uniform(size = nwalkers)

p1 = np.random.normal(loc = 9.8, scale = 0.0, size = nwalkers)
a10_0 = np.random.normal(loc = 0.0275, scale = 0.001, size = nwalkers)
a11_0 = np.random.normal(loc = 0.0275, scale = 0.001, size = nwalkers)
a12_0 = np.random.normal(loc = 0.01, scale = 0.005, size = nwalkers)
phi10_0 = np.random.uniform(size = nwalkers)
phi11_0 = np.random.uniform(size = nwalkers)
phi12_0 = np.random.uniform(size = nwalkers)
offset0 = np.random.normal(loc = 0.2, scale = 0.1, size = nwalkers)

names = ["a00", "a01", "phi00", "phi01", "a10", "a11", "a12", "phi10", "phi11", "phi12", "offset0"]
p0 = np.array([a00_0, a01_0, phi00_0, phi01_0, a10_0, a11_0, a12_0, phi10_0, phi11_0, phi12_0, offset0]).T
print(p0.shape)
ndim = len(p0[0])

# Go through initial guesses and check that all walkers have finite posterior probability
reset = 0
maxreset = 50000
print('Testing to see if initial params are valid')
for i in tqdm(range(nwalkers)):  
	llhood = probability(p0[i,:], obsdf)
	while (llhood == -np.Inf):
		p = random.random()
		p0[i,:] = (p*p0[random.randrange(nwalkers),:] + (1-p)*p0[random.randrange(nwalkers),:])
		llhood = probability(p0[i,:], obsdf)
		reset += 1
		if reset > maxreset:
			print("ERROR: Maximum number of resets has been reached, aborting run.")
			sys.exit() 
    
# Starting emcee up and doing the burnin
sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, args = [obsdf], live_dangerously = True)

state = sampler.run_mcmc(p0, nburnin, progress = True)
llhoods = sampler.get_log_prob(flat = True)
print(-2*llhoods.max()/(nobs - ndim))

# Implementing the clustering algorithm

sampler, state = clustering(sampler, state, names, probability, obsdf, ndim, nwalkers)
sampler.reset()

# Running the sampler
sampler.run_mcmc(state, nsample, progress = True)

# Making plots
# Likelihood plots    
from matplotlib.backends.backend_pdf import PdfPages
llhoods = sampler.get_log_prob(flat = True)
flatchain = sampler.get_chain(flat = True)

likelihoodspdf = PdfPages("likelihoods.pdf")
ylimmin = np.percentile(llhoods.flatten(), 1)
ylimmax = llhoods.flatten().max() + 1
for i in range(ndim):
	plt.figure(figsize = (9,9))
	plt.subplot(221)
	plt.hist(flatchain[:,i].flatten(), bins = 40, histtype = "step", color = "black")
	plt.subplot(223)
	plt.scatter(flatchain[:,i].flatten(), llhoods.flatten(),
		    c = np.mod(np.linspace(0,llhoods.size - 1, llhoods.size), nwalkers),
		    cmap = "nipy_spectral", edgecolors = "none", rasterized=True)
	plt.xlabel(names[i])
	plt.ylabel("Log(L)")
	plt.ylim(ylimmin, ylimmax)
	plt.subplot(224)
	plt.hist(llhoods.flatten(), bins = 40, orientation = "horizontal", 
		 histtype = "step", color = "black")
	plt.ylim(ylimmin, ylimmax)
	likelihoodspdf.attach_note(names[i])
	likelihoodspdf.savefig()
	#plt.savefig(runprops.get("results_folder")+"/likelihood_" + names[i] + ".png")

likelihoodspdf.close()
plt.close("all")

# Plotting best fit
ind = np.argmax(llhoods)
params = flatchain[ind,:].flatten()
print(params)

model, residuals, flux0, flux1 = likelihood(params, obsdf, synth = True)

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), obsdf["Source-Sky_T1"].values.flatten(), marker = "D", label = "Observed")
plt.plot(obsdf["JD_UTC"].values.flatten(), model, marker = "o", label = "Model")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Flux (arb. units)")
plt.xlim(2459280.8,2459281.1)
plt.savefig("best_model.png")
plt.close()

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), residuals, marker = "o")
plt.xlabel("Time")
plt.ylabel("Residual flux (arb. units)")
plt.xlim(2459280.8,2459281.1)
plt.savefig("best_residuals.png")
plt.close()

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), flux0, marker = "o", label = "Haumea")
plt.plot(obsdf["JD_UTC"].values.flatten(), flux1, marker = "o", label = "Hiiaka")
plt.xlabel("Time")
plt.ylabel("Flux variation (arb. units)")
#plt.xlim(2459280.8,2459281.1)
plt.savefig("variations.png")
plt.show()
plt.close()

# Making corner plot
fig = 0
fig = corner.corner(flatchain, labels = names, bins = 40, show_titles = True, 
		    plot_datapoints = False, color = "blue", fill_contours = True,
		    title_fmt = ".3f", truths = params, label_kwargs=dict(fontsize=20))
fig.tight_layout(pad = 1.08, h_pad = -0.4, w_pad = -0.4)
for ax in fig.get_axes():
	ax.tick_params(axis = "both", labelsize = 12, pad = 0.0)
fname = "corner.pdf"       
fig.savefig(fname, format = 'pdf')
plt.close("all")
