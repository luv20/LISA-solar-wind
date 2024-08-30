import sys
import os
sys.path.append('../')  # so we can see the packages in the above directory
from model import XYZTDILikelihood, estimate_parameter_uncertainties, set_bounds_from_errors, SNR
from waveform import wrap_BBHx_likelihood, wrap_BBHx_normal
from data import inject_signal
from noise import get_noise_covariance_matrix
import matplotlib.pyplot as py

import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger
from nessai.plot import corner_plot

from bbhx.waveformbuild import BBHWaveformFD


#importing XYZ data 
with open("/mnt/c/Users/User/Desktop/XYZ/no_window.txt", 'r') as f:
    X = f.readline().strip('\n')
    Y = f.readline().strip('\n')
    Z = f.readline().strip('\n')
    freqs = f.readline().strip('\n')
    times = f.readline().strip('\n')

X = np.array(X.split(), dtype=complex)
Y = np.array(Y.split(), dtype=complex)
Z = np.array(Z.split(), dtype=complex)
freqs = np.array(freqs.split(), dtype=float)
freqsnozero = freqs[1:]
times = np.array(times.split(), dtype=float)

windeffect = np.stack([X,Y,Z]).T * 0 #change multiplication factor to change strength of wind
windnozero = windeffect[1:] #without 0 frequency value (average), since that breaks stuff
### DATA PARAMETERS

#duration = 86400 * 0.5  # two weeks -> half a day
duration = times[-1] - times[0] #now determined from dataset
dt = 1.  # sampling cadence (s)
#   df = 1/duration
df = freqs[1]

print(duration)
print(times[0])
print(times[-1])
print(times[-1] - times[0])

#   fmin = df
#   fmax = 1e-1
#   num_freqs = int((fmax - fmin) / df) + 1


# frequency bins of the data
#   freqs = np.arange(num_freqs) * df + fmin  #freqs imported from external file
 
### WAVEFORM WRAPPER

wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), response_kwargs=dict(TDItag="XYZ"))
wave_wrap_like = wrap_BBHx_likelihood(wave_gen, f_ref=0.)
wave_wrap_norm = wrap_BBHx_normal(wave_gen, f_ref=0.)

### SOURCE PARAMETERS

phi_ref = np.pi/4 # phase at f_ref
m1 = 3e6
#m2 = 1e6
m2 = m1/2
a1 = 0.2
a2 = 0.4
dist = 10. # Gpc
inc = np.pi/3.
beta = np.pi/4.  # ecliptic latitude
lam = np.pi/5.  # ecliptic longitude
psi = np.pi/6.  # polarization angle
t_ref = 5*duration / 10  # t_ref (seconds)

Mt = m1 + m2
q = m1/m2

params_in = np.array([
    Mt,
    q,
    a1,
    a2,
    dist,
    phi_ref,
    inc,
    lam,
    beta,
    psi,
    t_ref
])

params_in_dict = {nm: params_in[i] for i, nm in enumerate(["Mt", "q", "a1", "a2", "dist", "phi_ref","iota","lam","beta","psi","t_ref"])}

waveform_kwargs = dict(squeeze=True, freqs=freqsnozero, direct=False, length=1024, fill=True, combine=False)

# inject signal (data = None for noiseless)  ASSUMES DATA IS IN FREQUENCY DOMAIN!

no_wind_signal = inject_signal(wave_wrap_norm, params_in, data=None, waveform_kwargs=waveform_kwargs)
data_with_signal = inject_signal(wave_wrap_norm, params_in, data=windnozero, waveform_kwargs=waveform_kwargs)

chirptimeseries = np.fft.irfft(np.append([[0,0,0]], no_wind_signal , axis=0), axis=0) #inverse FFT of blackhole signal - irfft requires first element to be zero freq., hence the 'append'
windtimeseries = np.fft.irfft(windeffect, axis=0)

#plot the chirp and wind XYZ time series
fig, ax = py.subplots(3, 1, figsize=(10, 15))  # 3 rows, 1 column

ax[0].plot(times, chirptimeseries[:,0], label = 'X')    #note, assumed that chirp average = 0 - may not necessarily be true.
ax[0].plot(times, chirptimeseries[:,1], label = 'Y')
ax[0].plot(times, chirptimeseries[:,2], label = 'Z')
ax[0].set_title('Blackhole Chirp time series')
ax[0].set_xlabel('time (seconds)')
ax[0].legend()

ax[1].plot(times, windtimeseries[:,0], label = 'X')
ax[1].plot(times, windtimeseries[:,1], label = 'Y')
ax[1].plot(times, windtimeseries[:,2], label = 'Z')
ax[1].set_title('Solar wind time series')
ax[1].set_xlabel('time (seconds)')
ax[1].legend()

ax[2].plot(times, chirptimeseries[:,0]+windtimeseries[:,0], label = 'X')
ax[2].plot(times, chirptimeseries[:,1]+windtimeseries[:,1], label = 'Y')
ax[2].plot(times, chirptimeseries[:,2]+windtimeseries[:,2], label = 'Z')
ax[2].set_title('Sum of Blackhole and Solar wind time series')
ax[2].set_xlabel('time (seconds)')
ax[2].legend()


##### configure likelihood

# noise assumptions: covariance matrix (has shape (3,3,N_f))

covariance_matrix = get_noise_covariance_matrix(freqsnozero)

# parameters to sample over
parameters_to_sample = ["Mt", "q", "a1", "a2", "dist", "phi_ref","iota","lam","beta","psi","t_ref"]
#parameters_to_sample = ["Mt", "q"]

# parameters_to_sample = ["Mt", "q", "a1", "a2", "dist"]

# smartly estimate the prior bounds of the analysis from the Fisher information matrix
marginal_widths = estimate_parameter_uncertainties(wave_wrap_norm, params_in, df, covariance_matrix, waveform_kwargs=waveform_kwargs)
prior_bounds = set_bounds_from_errors(params_in, marginal_widths, names=parameters_to_sample, scale=5) #change scale if distribution doesn't fit

# instantiate likelihood object
likelihood_model = XYZTDILikelihood(parameters_to_sample, prior_bounds, wave_wrap_like, data_with_signal, df, covariance_matrix, params_in_dict, waveform_kwargs=waveform_kwargs)

# get the SNR for information
snr_inj = SNR(params_in, wave_wrap_norm, df, covariance_matrix, waveform_kwargs=waveform_kwargs)

outdir = "Outputs/run50_nonoisestandardparams"  # output directory name
ncores = 10  # number of CPU cores to use
logger = setup_logger(output=outdir)

py.savefig('/home/luv/' + outdir + '/timeseriesplots') #save chirp and wind time series

logger.info(f"Optimal SNR of injection is {snr_inj:.2f}")

fs = FlowSampler(
    likelihood_model,
    output=outdir,
    stopping=0.1,
    resume=True,
    seed=42,
    nlive=1000,
    proposal_plots=False,
    plot=True,
    likelihood_chunksize=100,
    n_pool=ncores,
)
fs.run()

# plot the results

true_values = [params_in_dict[nm] for nm in parameters_to_sample]
corner_plot(
    fs.posterior_samples, 
    exclude=["logL","logP","it"], 
    truths=true_values,
    filename=os.path.join(outdir, "corner_plot.pdf"),
)


