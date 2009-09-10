CC=gcc
ARCH=-m32
CFLAGS=$(ARCH) -g $(OPT)
LDFLAGS=$(ARCH)
SOURCES=factorial.c 
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=factorial

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *o $(EXECUTABLE)
