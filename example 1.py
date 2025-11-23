from solvepath import (
    sigmoid, relu, leaky_relu, tanh_activation,
    softplus, swish, gelu, softmax
)

# take user input
x = float(input("Enter a value for x: "))

print("\n--- Activation Function Outputs ---")

# Sigmoid
res, steps = sigmoid(x, show_steps=True)
print(f"\nSigmoid({x}) = {res}")
for s in steps:
    print("  -", s)

# ReLU
res, steps = relu(x, show_steps=True)
print(f"\nReLU({x}) = {res}")
for s in steps:
    print("  -", s)

# Leaky ReLU
res, steps = leaky_relu(x, show_steps=True)
print(f"\nLeaky ReLU({x}) = {res}")
for s in steps:
    print("  -", s)

# Tanh
res, steps = tanh_activation(x, show_steps=True)
print(f"\ntanh({x}) = {res}")
for s in steps:
    print("  -", s)

# Softplus
res, steps = softplus(x, show_steps=True)
print(f"\nSoftplus({x}) = {res}")
for s in steps:
    print("  -", s)

# Swish
res, steps = swish(x, show_steps=True)
print(f"\nSwish({x}) = {res}")
for s in steps:
    print("  -", s)

# GELU
res, steps = gelu(x, show_steps=True)
print(f"\nGELU({x}) = {res}")
for s in steps:
    print("  -", s)

# Softmax
values = [x, x + 1, x + 2]  # example list

res, steps = softmax(values, show_steps=True)

print(f"\nSoftmax({values}) = {res}")
print("\nSoftmax Steps:")
for s in steps:
    print("  -", s)

print("\n-----------------------------------")
