
from argparse import ArgumentParser
import sys
def mult_4x4_approx (multip, multic):
  
  prod = int(multic) * int(multip)
  if prod == 75 or prod == 169:
  	prod = prod - 8
  elif ((multip == 6 and multic == 7) or (multip == 6 and multic == 15) or (multip == 7 and multic == 15)):
    prod = prod - 8
  
  return(prod)


def main():
	parser = ArgumentParser(description='Implements 4x4 multiplier. 1st argument is multiplier and 2nd ardgument is multiplicand.')
	parser.add_argument("multiplier", type = int, help="provide multiplier")
	parser.add_argument("multiplicand", type = int, help="provide multiplicand")

	args = parser.parse_args()

# Extract operands
	multip = args.multiplier
	multic = args.multiplicand
	if multip > 15 or multic > 15:
		print('Operands out of bound.... exiting')
		sys.exit()

	product = mult_4x4_approx(multip, multic)
	print ('product: ', product)
if __name__ == '__main__':
	main()


