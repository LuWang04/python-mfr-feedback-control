"""
******************************************************************************
hardware		: MFCS_EZ_Series
language		: python
date			: 06-03-2019
requirement		: 'mfcs_64.dll' under the same directory as the main script
author			: Lu Wang, Chemical Engineering, University of Southern California
function		: communicate with the pressure regulator, include:
				  obtain regulator status
				  read pressures
				  set pressures
******************************************************************************
"""
from __future__ import print_function	# Used for "print" function compatibility between Python 2.x and 3.x versions
import platform							# Library used for x64 or x86 detection
import time								# Time library, use of sleep function
import ctypes							# Used for variable definition types
from ctypes import *					# Used to load dynamic linked libraries


# This function uses "raw_input" for Python 2.x versions and "input" for Python 3.x versions
def myinput(prompt):
    try:
        return raw_input(prompt)
    except NameError:
        return input(prompt)


# Variable types definition 
mfcsHandle = c_ulong(0)
mySerial = c_ushort(0)
C_error = c_char()
B_OK = bool(False)
purge_status = c_byte(0)
mfcs_status = c_char()
pressure = c_float()
chrono = c_ushort(0)
 
# Load dll into memory
# lib_mfcs = ctypes.WinDLL('mfcs_32.dll')
lib_mfcs = ctypes.WinDLL('mfcs_64.dll')
# Initialize the first MFCS-EZ in Windows enumeration list 
mfcsHandle = lib_mfcs.mfcsez_initialisation(0)
# After initialization a short delay (at least 500ms) are required to make sure that 
#	the USB communication is properly established
time.sleep(1)

def init_mfcs():
	
	# Print status on MFCS initialization
	if (mfcsHandle != 0):
		print ('MFCS-EZ initialized')
	else:
		print ('Error on MFCS-EZ initialisation. Please check that device is plugged in.')

	# Get serial number of the MFCS-EZ associated to the MFCS session
	C_error = lib_mfcs.mfcs_get_serial(mfcsHandle, byref(mySerial))
	# Get status and SN value of called library function
	if (C_error == 0):
		print ('MFCS-EZ SN: ', mySerial.value)
	else:
		print ('Failed to get MFCS-EZ SN.')
	
	C_error=lib_mfcs.mfcs_set_alpha(mfcsHandle,0,5);   # Sends channel configuration
	
	C_error = lib_mfcs.mfcs_get_status(mfcsHandle,byref(mfcs_status))
	# Display mfcs status
	mfcs_status_value = int.from_bytes(mfcs_status.value,byteorder='big')
	if (mfcs_status_value == 0):
		print ('MFCS is reset')
	if (mfcs_status_value == 1):
		print ('MFCS is normal')
		myinput('Please switch on the valve on MFCS-EZ unit.')
	if (mfcs_status_value == 2):
		print ('MFCS is overpressure')
	if (mfcs_status_value == 3):
		print ('MFCS needs to be rearmed')
	
	# Ask user to confirm that chamber is closed
	# myinput("Please confirm that the chamber is closed (press 'ENTER')")


def read_pressure():	# channel number 1-4
	pressure_reading = []
	for channel in range(1,5):
		channel_char = c_char(channel)
		C_error = lib_mfcs.mfcs_read_chan(mfcsHandle,channel_char,
			byref(pressure),byref(chrono))
		print('Channel %i: %f mbar' %(channel, pressure.value))
		pressure_reading.append(pressure.value)
	return pressure_reading

def set_pressure(channel,p_set):
	channel_char = c_char(channel)
	pressure_set = c_float(p_set)
	C_error = lib_mfcs.mfcs_set_auto(mfcsHandle,channel_char,pressure_set)
	# time.sleep(20)
	print('Setting channel %i: %f mbar' %(channel, p_set))

	
def exit_mfcs(lib_mfcs = lib_mfcs):

	print('Setting zero pressure in channels')
	for channel in range(1,5):
		channel_char = c_char(channel) 
		C_error = lib_mfcs.mfcs_set_auto(mfcsHandle,channel_char,0)
		
	# Close communication port 
	B_OK = lib_mfcs.mfcs_close(mfcsHandle);
	if (B_OK == True):
		print ('USB connection closed')
	if (B_OK == False):
		print ('Failed to close USB connection')
	
	# Release the DLL
	ctypes.windll.kernel32.FreeLibrary(lib_mfcs._handle)
	del lib_mfcs
	print ('MFCS library unloaded')
		
	# # Exit application 
	# myinput("Press ENTER to exit application")

# Main function executed when this file is called
if __name__ == "__main__":
	print('mfcs.py run as main script')