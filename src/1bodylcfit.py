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

def likelihood(params, obsdf, synth = False):
	# unpack params
	p0, a00, a01, phi00, phi01, offset0 = params
	p0 = 3.915341
	p0 = p0/24
	t0 = 2459281

	times = obsdf["JD_UTC"].values.flatten()
	
	residuals = np.zeros(times.size)

	flux_haumea = a00*np.sin(2*np.pi*(phi00 + 2*(times-t0)/p0)) + a01*np.sin(2*np.pi*(phi01 + (times-t0)/p0)) + offset0

	#mag_haumea = -2.5*np.log10(flux_haumea)
	#mag_obs = -2.5*np.log10(obsdf["Source-Sky_T1"])
	#mag_err = np.abs(obsdf["Source_Error_T1"]/(obsdf["Source-Sky_T1"]*2.302585093))

	residuals = (obsdf["Source-Sky_T1"] - flux_haumea)/(5.0*obsdf["Source_Error_T1"])
	#residuals = (mag_obs - mag_haumea)/(mag_err)

	if synth:
		return flux_haumea, (obsdf["Source-Sky_T1"] - flux_haumea)

	chisq = np.nansum(residuals**2)

	return chisq


def priors(params):
	# apply flat priors to the data...
	#p0, a00, a01, a02, phi00, phi01, phi02, p1, a10, a11, a12, phi10, phi11, phi12, offset0, offset1 = params
	p0, a00, a01, phi00, phi01, offset0 = params
	if p0 <= 0:
		return -np.inf
	elif a00 <= 0 or a01 <= 0:
		return -np.inf
	elif phi00 <= 0 or phi01 <= 0:
		return -np.inf
	elif phi00 >= 1 or phi01 >= 1:
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
nburnin = 10000
nsample = 10000

obsdf = pd.read_csv("r_band.csv")
nobs = obsdf["JD_UTC"].values.size

# draw initial guess
p0_0 = np.random.normal(loc = 3.915341, scale = 0.0001, size = nwalkers)
a00_0 = np.random.normal(loc = 0.0205, scale = 0.01, size = nwalkers)
a01_0 = np.random.normal(loc = 0.0069, scale = 0.005, size = nwalkers)
phi00_0 = np.random.uniform(size = nwalkers)
phi01_0 = np.random.uniform(size = nwalkers)

offset0 = np.random.normal(loc = 0.2, scale = 0.1, size = nwalkers)
#offset0 = np.random.normal(loc = 1.5, scale = 0.25, size = nwalkers)

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

state = sampler.run_mcmc(p0, 500, progress = True)
llhoods = sampler.get_log_prob(flat = True)
print(-2*llhoods.max()/(nobs - ndim))
flatchain = sampler.get_chain(flat = True)
ind = np.argmax(llhoods)
params = flatchain[ind,:].flatten()
print(params)
print()

model, residuals = likelihood(params, obsdf, synth = True)

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), obsdf["Source-Sky_T1"].values.flatten(), marker = "D")
#plt.plot(obsdf["JD_UTC"].values.flatten(), -2.5*np.log10(obsdf["Source-Sky_T1"]), marker = "D")
plt.plot(obsdf["JD_UTC"].values.flatten(), model, marker = "o")
plt.show()
plt.close()

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), residuals, marker = "o")
plt.show()
plt.close()


sampler.reset

sampler.run_mcmc(state, 20000, progress = True)


llhoods = sampler.get_log_prob(flat = True)
print(-2*llhoods.max()/(nobs - ndim))
flatchain = sampler.get_chain(flat = True)
ind = np.argmax(llhoods)
params = flatchain[ind,:].flatten()
print(params)
print()

model, residuals = likelihood(params, obsdf, synth = True)

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), obsdf["Source-Sky_T1"].values.flatten(), marker = "D")
#plt.plot(obsdf["JD_UTC"].values.flatten(), -2.5*np.log10(obsdf["Source-Sky_T1"]), marker = "D")
plt.plot(obsdf["JD_UTC"].values.flatten(), model, marker = "o")
plt.show()
plt.close()

plt.figure()
plt.plot(obsdf["JD_UTC"].values.flatten(), residuals, marker = "o")
plt.show()
plt.close()

