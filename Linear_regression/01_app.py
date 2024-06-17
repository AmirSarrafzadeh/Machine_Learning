"""
This code checks numbers from 0 to 120 and prints the prime numbers.
"""

# check if the number is prime
for i in range(120):
    is_prime = True
    for j in range(2, i): # check if i is prime
        if i % j == 0:
            is_prime = False # i is not prime
            break
    if is_prime:
        print(i)
