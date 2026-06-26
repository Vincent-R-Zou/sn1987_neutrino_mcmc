import sys

index = 0  # Default index value
# If you pass a number from the command line, overwrite index
if len(sys.argv) > 1:
    index = int(sys.argv[1])
    print(f"Running MCMC with dynamic index: {index}")
else:
    print(f"Running MCMC with default index: {index}")

# ... the rest of your mcmc_2d.py code stays exactly the same ...
