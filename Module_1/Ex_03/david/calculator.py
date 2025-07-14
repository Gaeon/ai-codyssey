def add(a, b):
	print("Result : ", a+b)

def substract(a, b):
	print("Result : ", a-b)

def multiply(a, b):
	print("Result : ", a*b)

def divide(a, b):
	if b == 0:
		print("Error: Division by zero.") 
	else :
		print("Result : ", a/b)

def calculator():
	try:
		a = int(float(input("Enter first number: ")))
		b = int(float(input("Enter second number: ")))
	except ValueError:
		print("Invalid number input.")
		return

	operator = input("Input operator (+, -, *, /): ")

	if operator == '+' :
		add(a, b)
	elif operator == '-' :
		substract(a, b)
	elif operator == '*' :
		multiply(a, b)
	elif operator == '/' :
		divide(a, b)
	else :
		print("Invalid operator.")
	
if __name__ == "__main__":
    calculator()