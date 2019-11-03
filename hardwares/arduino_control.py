# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:09:48 2019
******************************************************************************
hardware		: Arduino Uno/Mega
language		: python
requirement		: the Arduino boards must be loaded with Firmata file (Arduino IDE)
author			: Lu Wang, Chemical Engineering, University of Southern California
function		: Integrate all the Arduino control, include:
				  Linear stage
				  light sources
				  optical switches
******************************************************************************
"""
import serial
from serial.tools.list_ports import comports

from pyfirmata import ArduinoMega	# for connecting to ArduinoMega
from pyfirmata import Arduino		# for connecting to ArduinoUno


import time

from hardwares import arduino_boards		# import the Arduino board serial numbers



# =============== connect to all the Arduino boards, and ===================
# ============ create a dictionary of board name and board class ==========

Arduino_SN = arduino_boards.Arduino_SN
ArduinoMega_SN = arduino_boards.ArduinoMega_SN

Arduinos = {}			# Initialize the board dictionary
for comport in comports():
	# print(comport.vid)			# 9025 (type: int)
	# print(comport.serial_number)	# "95530343235351605281" (type: str)
	# print(comport.device)			# "COM4" (type: str)
	# print(comport)				# "COM4 - USB Serial Device (COM4)" (type: class)
	SN = comport.serial_number
	
	if SN in ArduinoMega_SN.keys():
		board_name = ArduinoMega_SN[SN]
		print(board_name, ':', comport.device)
		# adding the 'timeout' to prevent raising wirteTimeout Error, it's None by default
		board = ArduinoMega(comport.device, timeout = 0)
		# time.sleep(0.5)		# Wait for communication to get established
		Arduinos[board_name] = board
		
	elif SN in Arduino_SN.keys():
		board_name = Arduino_SN[SN]
		print(board_name, ':',  comport.device)
		board = Arduino(comport.device)
		# time.sleep(0.5)		# Wait for communication to get established
		Arduinos[board_name] = board



# =================== assign pins for each board ====================
# Define each board
# uno_1 = Arduinos['Uno_1']
# mega_1 = Arduinos['Mega_1']
mega_2 = Arduinos['Mega_2']
mega_3 = Arduinos['Mega_3']


# Define pins for stage moving (mega_2)
stage_direct = mega_2.get_pin('d:7:o')
stage_step = mega_2.get_pin('d:8:o')


# Define pins for light sources (mega_3)
uv_vis_pin = mega_3.get_pin('d:13:o')
led_pin = mega_3.get_pin('d:53:o')

# Define a list of pins for optic switches (mega_3)
optic_switch = [mega_3.get_pin('d:41:o'),
				mega_3.get_pin('d:43:o'),
				mega_3.get_pin('d:45:o'),
				mega_3.get_pin('d:47:o')]



# ======================== control functions ======================
# Move the stage
def stage_move(n,direction):
# n: number of blocks the stage need to move
# direction: "towards motor"==1, "away from motor"==0
	pulse_width = 1/1000000000.0 # 1 microseconds
	sec_between_steps = 2/1000000000.0 # larger number results in slower steps
	block_steps = 4060 	# steps motor need to go to travel one block
	if direction == "towards motor":
		stage_direct.write(1)
	elif direction == "away from motor":
		stage_direct.write(0)

	for step in range (n*block_steps):
		stage_step.write(1)
		time.sleep(pulse_width)
		stage_step.write(0)
		time.sleep(sec_between_steps)


# Move stage and choose optical switch according to channel number
def choose_chan(chan): #, init_count):
	
	stage_pos = chan // 4	# stage_pos: i in the flow chart
	optic_pos = chan % 4	# optic_pos: j in the flow chart (remainer)

	if optic_pos == 0:	# this means it's time to start a new stage location
		if stage_pos > 0:	# stage should be at 2/3/4th location
			# Move stage to the next location
			stage_move(1,'away from motor')
		# else:	# stage should stay at the 1st location and not moving
			
	#else: # it's in the process of switching optical swtiches, no need to move stage
	
	# open optical switches accordingly
	optical_switch(chan, 1)


# Move stage back to the first position
def stage_return():
	# Move stage from location 4 to location 1
	stage_move(3,'towards motor')


# Moving stage according to user input
def custom_move_stage():
	# determine moving direction and distance by user input
	is_dir_error = True
	while is_dir_error:
		direct = input('Which direction to move? (format: int number)\n0. Away from the motor\n1. Towards the motor\n')
		if direct.isdigit(): # a number
			if int(direct) > 1 or int(direct) <0:
				is_dir_error = True
				print('Input value not valid, please enter 0 or 1')
			else: 
				if int(direct) == 0:
					answer = input('Moving away from motor, OK? [Y/N]')
				elif int(direct) == 1:
					answer = input('Moving towards motor, OK? [Y/N]')
				
				if answer in ['Y', 'y', '', 'YES', 'yes']:
					is_dir_error = False
				elif answer in ['N', 'n', 'NO', 'No', 'no']:
					is_dir_error = True
				else:
					is_dir_error = True
					print('Not a valid answer...')
		else:
			is_dir_error = True
			print('Input format not valid, please enter an interger')

	is_blc_error = True
	while is_blc_error:
		blocks = input('How many blocks? (formant: int number)\n')
		if blocks.isdigit(): # a number
			if int(blocks) > 3 or int(blocks)<0:
				is_blc_error = True
				print('Value not valid, please enter 1, 2 or 3')
			else: 
				answer = input('Moving '+blocks+' blocks, OK? [Y/N]')
						
				if answer in ['Y', 'y', '', 'YES', 'yes']:
					is_blc_error = False
				elif answer in ['N', 'n', 'NO', 'No', 'no']:
					is_blc_error = True
				else:
					is_dir_error = True
					print('Not a valid answer...')
		else:
			is_blc_error = True
			print('Input format not valid, please enter an interger')

	# start moving the stage
	if int(direct) == 0:
		print('away from motor for '+blocks+' blocks')
		stage_move(blocks,"away from motor")
	else:	
		print('approach to the motor for '+blocks+' blocks')	
		stage_move(blocks,"towards motor")




def light_source(on_light = None):
	if on_light == None:
		#print("Turn off both lights")
		led_pin.write(0)
		uv_vis_pin.write(0)
	elif on_light == "LED":
		#print("Turn on LED light source")
		uv_vis_pin.write(0)
		led_pin.write(1)
	elif on_light == "UV-vis":
		#print("Turn on UV-vis light source")
		led_pin.write(0)
		uv_vis_pin.write(1)


# Switch jth solenoid on/off
def optical_switch(chan,status):
# status: on=1, off=0
	optic_pos = chan % 4	# optic_pos: j in the flow chart (remainer)

	if status == 1:
		optic_switch[optic_pos].write(1)
	elif status == 0:
		optic_switch[optic_pos].write(0)


def exit_boards():
	for board in boards:	# board is key in the dictionary
		boards[board].exit()
		print(board)