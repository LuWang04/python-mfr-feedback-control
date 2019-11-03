# -*- coding: utf-8 -*-
"""
******************************************************************************
hardware		: multiplexer
language		: python
requirement		: --
author			: Lu Wang, Chemical Engineering, University of Southern California
function		: used in main script for reading IR signals. 
				  analogue multiplexer is being switched among 16 channels, 
				  achieved by sending HIGH/LOW status from 4 Arduino pins.
******************************************************************************
"""
import time


muxChannels = {}			# Pre-define a list of S0-S3 status.
for n in range(16):
	Bi = [0,0,0,0]
	i = 0
	a = n
	while a > 0:
		Bi[i] = a%2
		i += 1
		a = a//2
	muxChannels[n] = Bi


def Chan(n):
	return muxChannels[n]


def DefinePin(board, switch_pin_nums, sig_pin_num):
	PinS = []
	for pin in switch_pin_nums:
		PinS.append(board.get_pin('d:'+ str(pin) + ':o'))
	PinSig = board.get_pin('a:'+ str(sig_pin_num) +':i')
	PinSig.enable_reporting()
	time.sleep(0.5)
	return PinS, PinSig


def ReadMUX(PinS, PinSig, n):
	S = Chan(n)
	for i in range(4):
		PinS[i].write(S[i])
		# time.sleep(1)
	time.sleep(1/100)
	# time.sleep(1/45)
	sig = format(PinSig.read()*5, '.4f') #conver to voltage (0-5V) and format digits
	return sig

def SwitchMUX(PinS, n):
	S = Chan(n)
	for i in range(4):
		PinS[i].write(S[i])
		# time.sleep(1)
	time.sleep(1/100)
	

def Read(PinSig):
	return format(PinSig.read()*5, '.4f')





if __name__ == "__main__":
	
	[S0, S1, S2, S3] = Chan(2)
	print(S0, '&', S1, '&', S2, '&', S3)

