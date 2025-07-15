def main():
	try :
		number = float(input("Enter number : "))
	except ValueError:
		print("Invalid number input.")
		return
	try :
		exponent = int(input("Enter exponent : "))
	except ValueError:
		print("Invalid exponent input.")
		return

	if number == 0 and exponent < 0:
		print("Error: Cannot raise 0 to a negative exponent.")
		return

	result = 1.0
	for _ in range(abs(exponent)) :
		result *= number

	if exponent < 0 :
		result = 1/result
	
	print("Result : ", result)

if __name__ == "__main__":
	main()