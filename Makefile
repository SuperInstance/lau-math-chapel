# Makefile for lau-math-chapel

CHPL ?= chpl
CHPL_FLAGS ?= --fast -O2
SRC = LauMatrix.chpl LauLaplacian.chpl LauHeatKernel.chpl LauAgentFleet.chpl LauConservation.chpl LauTopology.chpl
TEST = test_main.chpl
TARGET = test_main

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRC) $(TEST)
	$(CHPL) $(CHPL_FLAGS) $(TEST) -o $(TARGET)

test: $(TARGET)
	./$(TARGET) -nl 1

test-2nl: $(TARGET)
	./$(TARGET) -nl 2

clean:
	rm -f $(TARGET) $(TARGET).real $(TARGET).fabric_util *_test_* *.exec

%.chpl: ; # Chapel source files, no build step needed
