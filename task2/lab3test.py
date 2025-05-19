import math

def check_sin(filename, tolerance=0.001):
    print(f"Checking {filename}")
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            id_ = int(parts[0].split(": ")[1])
            arg = float(parts[1].split(": ")[1])
            reported = float(parts[2].split(": ")[1])
            actual = math.sin(arg)
            diff = abs(reported - actual)
            if diff > tolerance:
                print(f"[!] SIN MISMATCH (id={id_}): reported={reported}, actual={actual}, diff={diff}")
    print("Done.\n")

def check_sqrt(filename, tolerance=0.001):
    print(f"Checking {filename}")
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            id_ = int(parts[0].split(": ")[1])
            arg = float(parts[1].split(": ")[1])
            reported = float(parts[2].split(": ")[1])
            actual = math.sqrt(arg)
            diff = abs(reported - actual)
            if diff > tolerance:
                print(f"[!] SQRT MISMATCH (id={id_}): reported={reported}, actual={actual}, diff={diff}")
    print("Done.\n")

def check_pow(filename, tolerance=0.001):
    print(f"Checking {filename}")
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            id_ = int(parts[1])
            base = float(parts[3])
            exponent = float(parts[5])
            reported = float(parts[7])
            actual = math.pow(base, exponent)
            diff = abs(reported - actual)
            if diff > tolerance:
                print(f"[!] POW MISMATCH (id={id_}): reported={reported}, actual={actual}, diff={diff}")
    print("Done.\n")

# Проверяем все три файла
check_sin("/Users/stanislavmalarcuk/Desktop/паралельки/lab3part2/build/sin_results.txt")
check_sqrt("/Users/stanislavmalarcuk/Desktop/паралельки/lab3part2/build/sqrt_results.txt")
check_pow("/Users/stanislavmalarcuk/Desktop/паралельки/lab3part2/build/pow_results.txt")
