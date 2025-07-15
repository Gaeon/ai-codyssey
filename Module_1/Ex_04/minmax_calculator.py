def main():
	try:
		input_str = input("Enter numbers: ")
		numbers = [float(n) for n in input_str.split()]
		min_val = numbers[0]
		max_val = numbers[0]

		for num in numbers[1:]:
			if num < min_val:
				min_val = num
			if num > max_val:
				max_val = num

		print(f"Min: {min_val}, Max: {max_val}")
	except ValueError:
		print("Invalid input.")

if __name__ == "__main__":
	main()