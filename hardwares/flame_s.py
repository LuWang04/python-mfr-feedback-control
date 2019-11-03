# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:04:17 2019
******************************************************************************
hardware		: FlameS spectrometer, Ocean Optics
language		: python
requirement		: install python-seabreeze (https://github.com/ap--/python-seabreeze)
author			: Lu Wang, Chemical Engineering, University of Southern California
function		: grab spectrum information from FlameS and process the
				  PL data and UV-vis data in real-time
******************************************************************************
"""

import seabreeze.spectrometers as sb	# Ocean optics
import numpy as np
from scipy.signal import savgol_filter	# Savitzky-Golay filter for smoothing uv-vis specs
import time

# spectrometer initialization
spectrometers = sb.list_devices()			# recognize devices
print(spectrometers)
flames = sb.Spectrometer(spectrometers[0])	# get specification from first device
# start reading
wave = flames.wavelengths()			# return an array containing all wavelengths
# test spectrometer
flames.intensities()

# If there is a blank sample (standard sample) of 16 channels available
# toluene_object = np.load('Toluene.npy', allow_pickle = True)	# a transmittance spec when only pure toluene in tube
# raw_tol = toluene_object.item()




def direct_read():
	flames.integration_time_micros(250000)
	spec = flames.intensities()
	return spec


# Read PL spectrum with Flame-S-UV-Vis
# Return the time, PL spectrum, peak wavelength and FWHM wavelengths
def read_pl(start = 760):
	# define wavelength range of interest (ROI) by index numbers
	# so that we can eliminate the tall peak around 405 nm, which is excitation wavelength
	
	# set integration time in microseconds (us)
	flames.integration_time_micros(500000)
	
	realtime_pl = time.time()
	# read spectrum
	pl_spec = flames.intensities()
	# determine fwhm
	peak, fwhm = FWHM(pl_spec,start)

	return realtime_pl, pl_spec, peak, fwhm


# Determine FWHM
def FWHM(pl_spec,start = 760):
	pl_smooth = savgol_filter(pl_spec, 101, 2) # window size 101, polynomial order 2
	
	peak_intensity = max(pl_smooth[start:])
	baseline = pl_smooth[-750:].mean()
	peak = start + pl_smooth[start:].argmax(axis = 0)
	
	half_max = (peak_intensity-baseline)/2
	spec_cut_half = pl_smooth[start:] - half_max - baseline
	fwhm_index = start + np.where(np.diff(np.sign(spec_cut_half)))[0]
	# according to the index, get wavelengths of these two points
	
	# if it's just a wavy baseline
	if len(fwhm_index) > 3:
		print('wavy baseline')
		return float('nan'), float('nan')
	
	# if the maximum is from 405nm excitation, need to re-define peak
	# and re-define fwhm
	if peak < start+50:
		if len(fwhm_index) == 1:
			# there might be a subtle peak, or no peak, pass
			print('subtle peak')
			return float('nan'), float('nan')
		elif len(fwhm_index) == 2:
			# fwhm might be cutting a local minimum or maximum
			print('fwhm cuts local min or max')
			max_intensity = max(pl_smooth[fwhm_index[0]:fwhm_index[1]])
			if max_intensity > pl_smooth[fwhm_index[0]]:
				# peak is within this range
				peak = pl_smooth[fwhm_index[0]:fwhm_index[1]].argmax(axis = 0)
			else:
				# peak is the right boundary of the fwhm_index
				peak = fwhm_index[1]
		elif len(fwhm_index) == 3:
			# need to re-define the peak location in the range of fwhm
			print('fwhm has three elements')
			left_lim = fwhm_index[-2]
			right_lim = fwhm_index[-1]
			peak = start + left_lim + pl_smooth[left_lim:right_lim].argmax(axis = 0)
				
		peak_intensity = pl_smooth[peak]	
		half_max = (peak_intensity-baseline)/2
		spec_cut_half = pl_smooth[start:] - half_max - baseline
		fwhm_index = start + np.where(np.diff(np.sign(spec_cut_half)))[0]
	
	
	# now determine peak wavelength and fwhm
	peak_wave = wave[peak]
	if len(fwhm_index) > 1:
		print('fwhm_index more than one elements')
		fwhm_2 = wave[fwhm_index[-1]] - wave[fwhm_index[-2]]
	elif len(fwhm_index) == 1:
		print('fwhm_index only one element')
		fwhm_2 = 2 * (wave[fwhm_index[0]] - wave[peak])
	elif len(fwhm_index) == 0:
		print('fwhm_index is empty now, wavy baseline')
		return float('nan'), float('nan')

	return peak_wave, fwhm_2


# Read UV-vis absorption spectrum with Flame-S-UV-vis
# Corrected with control sample of pure toluene, return the time and spectrum
def read_abs(chan, left_standard = 975, right_standard = 1010):
	# define wavelength range of interest (ROI) by index numbers

	flames.integration_time_micros(250000)
	
	realtime_abs = time.time() 
	uv_vis_spec = flames.intensities() # read spectrum

	# If there is a blank sample (standard sample) of 16 channels available
	# uv_vis_spec = process_abs(chan, uv_vis_spec)
	
	return realtime_abs, uv_vis_spec



def process_abs(chan, sample_spec, left_standard = 975,stop = 1010):
	tol = raw_tol['Chan'+str(chan+1)]
	tol_smooth = savgol_filter(tol, 301, 2)
	sample_smooth = savgol_filter(sample_spec, 301, 2)
	# match ranges
	tol_match, sample_match = match_range(tol_smooth,sample_smooth,left_standard, right_standard)
	
	transmittance = sample_match/tol_match
	uv_vis_abs =  -1 * np.log(transmittance)
	return uv_vis_abs	



def match_range(spec1, spec2, left_standard = 700 , right_standard = 980): # spec1 as a standard here
	
	range1 = spec1[right_standard]-spec1[left_standard]
	range2 = spec2[right_standard]-spec2[left_standard]
	ratio = range1/range2
	
	new_spec1 = spec1 - spec1[left_standard]
	new_spec2 = (spec2-spec2[left_standard])*ratio

	# sometimes the negative values appear in specs
	if min(new_spec1)<min(new_spec2):
		neg_baseline = min(new_spec1)
	else:
		neg_baseline = min(new_spec2)
	new_spec1 -= (neg_baseline - 10)
	new_spec2 -= (neg_baseline - 10)
	
	return new_spec1, new_spec2
