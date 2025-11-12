# Define compiler
NVCC = nvcc

# Define the executable files
TARGET1 := p1
TARGET2 := p2
TARGET3 := p3

# Define the source files
SRC_FILE1 := chileiw_yuchunl8_tingcc3_1.cu
SRC_FILE2 := chileiw_yuchunl8_tingcc3_2.cu
SRC_FILE3 := chileiw_yuchunl8_tingcc3_3.cu

.PHONY: all clean

# Include flag for NVCC
INCLUDE_FLAG := -I.

all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): $(SRC_FILE1)
	$(NVCC) $(SRC_FILE1) -o $(TARGET1) $(INCLUDE_FLAG)

$(TARGET2): $(SRC_FILE2)
	$(NVCC) $(SRC_FILE2) -o $(TARGET2) $(INCLUDE_FLAG)

$(TARGET3): $(SRC_FILE3)
	$(NVCC) $(SRC_FILE3) -o $(TARGET3) $(INCLUDE_FLAG)

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3)
