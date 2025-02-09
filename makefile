TYPE ?= double  # Значение по умолчанию

all:
	g++ -DTYPE=$(TYPE) laba1.cpp -o lab1
