# Compiler
CXX = g++
Ccc = gcc

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
CXXFLAGS = -fopenmp 
CccFLAGS = -lm

debug: CXXFLAGS += -DDEBUG -g -Wall
debug: CccFLAGS += -DDEBUG
debug: all

# The build target executable:
TARGET = lss

all: $(TARGET)

$(TARGET): $(TARGET).cpp lib/imghandler.o lib/evcontour.o
	$(CXX) $(CXXFLAGS) $(TARGET).cpp lib/imghandler.o lib/evcontour.o -o $(TARGET)

clean:
	$(RM) $(TARGET) *.o lib/*.o *.tar* *.core*
