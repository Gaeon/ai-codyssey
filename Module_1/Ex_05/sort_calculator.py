def main():
	try :
		input_str = input("Enter numbers: ")
		numbers = [float(n) for n in input_str.split()]

		len = n = len(numbers)
	except ValueError:
		print("Invalid input.")

if __name__ == "__main__":
	main()