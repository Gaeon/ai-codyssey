def main():
	try :
		input_str = input("Enter numbers: ")
		numbers = [float(n) for n in input_str.split()]

		len = len(numbers)
		for i in range(len) :
			min_index = i
			for j in range(i + 1, len):
				if numbers[j] < numbers[min_index]:
					min_index = j
			numbers[i], numbers[min_index] = numbers[min_index], numbers[i]

		print("Sorted:", " ".join(str(float(num)) for num in numbers))

	except ValueError:
		print("Invalid input.")

if __name__ == "__main__":
	main()