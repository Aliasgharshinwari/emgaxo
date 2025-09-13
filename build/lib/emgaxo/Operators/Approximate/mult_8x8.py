from argparse import ArgumentParser
import sys
#from mult_8x8_helper_functions import mult_4x4_approx
from emgaxo.Operators.Approximate.mult_4x4 import mult_4x4_approx

def mult_4x4_acc(multip, multic):
  print('multiplier: ', multip)
  print('multiplicand: ', multic)
  prod = int(multic) * int(multip)
  
  return(prod)

def mult_8x8_approx(multip, mtc, adder_type):
# Represent multiplier and multiplicand in 8-bits
	multip_bin = format(multip, '08b')
	mtc_bin = format(mtc, '08b')

	#Segment multiplier and multiplicand in 4-bit binary numbers.
	#Convert non-string to string and then integer
	multip_bin_lsb = int(str(multip_bin[4:8]), 2) # note the indices
	multip_bin_msb = int(str(multip_bin[0:4]), 2)

	mtc_bin_lsb = int(str(mtc_bin[4:8]), 2)
	mtc_bin_msb = int(str(mtc_bin[0:4]), 2)

	#call multiplier function
	product_LL = mult_4x4_approx(multip_bin_lsb, mtc_bin_lsb)
	product_LH = mult_4x4_approx(multip_bin_lsb, mtc_bin_msb)
	product_HL = mult_4x4_approx(multip_bin_msb, mtc_bin_lsb)
	product_HH = mult_4x4_approx(multip_bin_msb, mtc_bin_msb)

	#Check for final adder type
	if adder_type == 'acc':
		
	#Perfrom padding and shifting before addition
		
		product_LL_shift = int(str(product_LL).zfill(16))
		
		
		product_LH_shift = int(str(product_LH).zfill(12))
		 
		product_LH_shift = product_LH_shift << 4
		 

		product_HL_shift = int(str(product_HL).zfill(12))
		product_HL_shift = product_HL_shift << 4

		product_HH_shift = product_HH << 8

		product = product_LL_shift + product_LH_shift + product_HL_shift +  product_HH_shift
		return (product)
	elif adder_type == 'app':
	# Do not add the N/2 LSBs and MSBs
	# Convert the result of individual 4x4 multipliers into binary

		product_LL_binary = format(product_LL, '08b')
		product_LH_binary = format(product_LH, '08b')
		product_HL_binary = format(product_HL, '08b')
		product_HH_binary = format(product_HH, '08b')
		
	# Separate the LSBs and MSBs before addition
		product_LL_binary_lsb = product_LL_binary[4:8]
		product_LL_binary_msb = product_LL_binary[0:4]
		
		product_LH_binary_lsb = product_LH_binary[4:8]
		product_LH_binary_msb = product_LH_binary[0:4]
		
		product_HL_binary_lsb = product_HL_binary[4:8]
		product_HL_binary_msb = product_HL_binary[0:4]
		
		product_HH_binary_lsb = product_HH_binary[4:8]
		product_HH_binary_msb = product_HH_binary[0:4]

	# add the four MSBs of product_LL with four LSBs of product_LH and product_HL
	# Does not propagate carries. Also discards carries e.g. 1 + 1 + 1 = 1 
	# PP is for storing the result of this addition
		PP = [0] * 8
		for i in range(3, -1, -1):
			PP[i+4] = (int(product_LL_binary_msb[i]) + int(product_LH_binary_lsb[i]) + int(product_HL_binary_lsb[i])) % 2
		
	# Add the four MSbs of product_LH and product_HL with four LSBs of product_HH 
		for j in range(3, -1, -1):
			PP[j] = (int(product_LH_binary_msb[j]) + int(product_HL_binary_msb[j]) + int(product_HH_binary_lsb[j])) % 2

	# PP is a list. Convert list into string
		PP_str = ''.join(str(e) for e in PP)
		
	# Concatenate the four MSBs of product_HH with PP and four LSBs of product_LL
		product = product_HH_binary_msb + PP_str + product_LL_binary_lsb
	# Convert to integer
		product = int(product, 2)

		return (product)



def main():

	parser = ArgumentParser(description='Implements 8x8 multiplier. 1st argument is multiplier and 2nd ardgument is multiplicand.')
	parser.add_argument("multiplier", type = int, help="provide multiplier")
	parser.add_argument("multiplicand", type = int, help="provide multiplicand")
	parser.add_argument("-add", "--adder", default = 'acc', type = str, help='provide adder type: "acc" for accurate or "app" for approximate. Default is accurate')

	args = parser.parse_args()

	#Extract Operands
	multip = args.multiplier
	mtc = args.multiplicand
	adder_type = args.adder

	
	if multip > 255 or mtc > 255:
		print("Inputs out of bound. Exiting in 3 2 1 ...")
		sys.exit()

	product = mult_8x8_approx(multip, mtc, adder_type)
	print ('product: ', product)

if __name__ == '__main__':
	main()