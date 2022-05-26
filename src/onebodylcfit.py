#
#	onebodylcfit.py
#
#	Built specifically to fit Haumea and Hiiaka's light curves by fitting sinusoidal shapes
#	to them.
#
#	Benjamin Proudfoot
#	Seneca Heilesen
#	5/26/2022
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
from astroquery.jplhorizons import Horizons

def likelihood(params, obsdf, synth = False):
	# unpack params
	p0, a00, a01, phi00, phi01, offset0 = params
	p0 = 3.915341 #Haumea rotational period (unit=hours)
	p0 = p0/24 #Convert to days (=0.163139)
	t0 = 2459281 #JD for first BYU observation

	times = obsdf["JD_UTC"].values.flatten()
	
	residuals = np.zeros(times.size)

	flux_haumea = a00*np.sin(2*np.pi*(phi00 + 2*(times-t0)/p0)) + a01*np.sin(2*np.pi*(phi01 + (times-t0)/p0)) + offset0

	#mag_haumea = -2.5*np.log10(flux_haumea)
	#mag_obs = -2.5*np.log10(obsdf["Source-Sky_T1"])
	#mag_err = np.abs(obsdf["Source_Error_T1"]/(obsdf["Source-Sky_T1"]*2.302585093))
	
	#Auto-normalize the data
	source_norm = (obsdf["Source-Sky_T1"] - np.mean(obsdf["Source-Sky_T1"]))/np.std(obsdf["Source-Sky_T1"])
	err_norm = (obsdf["Source_Error_T1"]-np.mean(obsdf["Source_Error_T1"]))/np.std(obsdf["Source-Sky_T1"])

	residuals = (source_norm - flux_haumea)/(err_norm)
	#residuals = (mag_obs - mag_haumea)/(mag_err)

	if synth:
		return flux_haumea, (source_norm - flux_haumea)

	chisq = np.nansum(residuals**2)

	return chisq


def priors(params):
	# apply flat priors to the data...
	#p0, a00, a01, a02, phi00, phi01, phi02, p1, a10, a11, a12, phi10, phi11, phi12, offset0, offset1 = params
	p0, a00, a01, phi00, phi01, offset0 = params
	if p0 <= 0.155:
		return -np.inf
	elif p0 >= 0.17:
		return -np.inf
	elif a00 <= 0 or a01 <= 0:
		return -np.inf
	elif phi00 <= 0:
		phi00=0.9
		return -np.inf
	elif phi01 <= 0:
		phi01 = 0.9
		return -np.inf
	elif phi00 >= 1:
		phi00 = 0.1
		return -np.inf
	elif phi01 >= 1:
		phi01 = 0.1
		return -np.inf
	#elif offset0 <= 0:
	#	return -np.inf
	elif a00 <= a01:
		return -np.inf
	else:
#		print("here")
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
nburnin = 500
nsample = 500

obsdf = pd.read_csv("Measurements.csv")
nobs = obsdf["JD_UTC"].values.size

times = obsdf["JD_UTC"].values.flatten() #unit=days

ourKBO = Horizons(id="Haumea",location=705,epochs = times)

# Correct for light-time variations
lighttime = ourKBO.vectors()['lighttime'] #unit=minutes
lighttime = lighttime*1440 #convert to fraction of a day
correctedtime = times-lighttime

obsdf["JD_UTC"] = correctedtime

# draw initial guess
p0_0 = np.random.normal(loc = 0.163139, scale = 0.0001, size = nwalkers)
a00_0 = np.random.normal(loc = 0.0205, scale = 0.01, size = nwalkers)
a01_0 = np.random.normal(loc = 0.0069, scale = 0.005, size = nwalkers)
phi00_0 = np.random.uniform(size = nwalkers)
phi01_0 = np.random.uniform(size = nwalkers)

offset0 = np.random.normal(loc = 0.2, scale = 0.1, size = nwalkers)
#offset0 = np.random.normal(loc = 1.5, scale = 0.25, size = nwalkers)

names = ["p0", "a00", "a01", "phi00", "phi01", "offset0"]
p0 = np.array([p0_0, a00_0, a01_0, phi00_0, phi01_0, offset0]).T
print(p0.shape)
ndim = len(p0[0])

# Go through initial guesses and check that all walkers have finite posterior probability
reset = 0
maxreset = 5000
print('Testing to see if initial params are valid')
for i in tqdm(range(nwalkers)):  
	llhood = probability(p0[i,:], obsdf)
	#print(llhood)
	while (llhood == -np.Inf):
		p = random.random()
		p0[i,:] = (p*p0[random.randrange(nwalkers),:] + (1-p)*p0[random.randrange(nwalkers),:])
		llhood = probability(p0[i,:], obsdf)
		#print("reset:",llhood)
		reset += 1
		if reset > maxreset:
			print("ERROR: Maximum number of resets has been reached, aborting run.")
			sys.exit() 
    
# start emcee burn in
# start emcee sample
# make plots

#print(obsdf)

sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, args = [obsdf], live_dangerously = True)

state = sampler.run_mcmc(p0, nburnin, progress = True)
llhoods = sampler.get_log_prob(flat = True)
print(-2*llhoods.max()/(nobs - ndim))

#Implement clustering algorithm
sampler, state = clustering(sampler, state, names, probability, obsdf, ndim, nwalkers)
sampler.reset()

# start emcee sample
sampler.run_mcmc(state, nsample, progress = True)
flatchain = sampler.get_chain(flat = True)
ind = np.argmax(llhoods)
params = flatchain[ind,:].flatten()
#print(params)
p0, a00, a01, phi00, phi01, offset0 = params
model, residuals = likelihood(params, obsdf, synth = True)

#make plots
plt.figure()
source_norm = (obsdf["Source-Sky_T1"] - np.mean(obsdf["Source-Sky_T1"]))/np.std(obsdf["Source-Sky_T1"])
plt.plot(obsdf["JD_UTC"].values.flatten(), source_norm.values.flatten(), marker = "D", label = "Observed")
#plt.plot(obsdf["JD_UTC"].values.flatten(), -2.5*np.log10(obsdf["Source-Sky_T1"]), marker = "D")
plt.plot(obsdf["JD_UTC"].values.flatten(), model, marker = "o")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Flux (arb. units)")
plt.savefig("best_model_1body.jpg", bbox_inches='tight')
plt.show()
plt.close()

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), residuals, marker = "o")
plt.xlabel("Time")
plt.ylabel("Residual flux (arb. units)")
plt.savefig("best_residuals_1body.jpg", bbox_inches='tight')
plt.show()
plt.close()


llhoods = sampler.get_log_prob(flat = True)
print(-2*llhoods.max()/(nobs - ndim))

# Likelihood plots    
from matplotlib.backends.backend_pdf import PdfPages
llhoods = sampler.get_log_prob(flat = True)
flatchain = sampler.get_chain(flat = True)

likelihoodspdf = PdfPages("likelihoods_1body.pdf")
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

# Making corner plot
fig = 0
fig = corner.corner(flatchain, labels = names, bins = 40, show_titles = True, 
		    plot_datapoints = False, color = "blue", fill_contours = True,
		    title_fmt = ".3f", truths = params, label_kwargs=dict(fontsize=20))
fig.tight_layout(pad = 1.08, h_pad = -0.4, w_pad = -0.4)
for ax in fig.get_axes():
	ax.tick_params(axis = "both", labelsize = 12, pad = 0.0)
fname = "corner_1body.pdf"       
fig.savefig(fname, format = 'pdf')
plt.close("all")
