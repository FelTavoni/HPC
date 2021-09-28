EXE       = password_bf
CC        = gcc
FLAGS     = -Wall -g -Wextra -O3
INCLUDES  = "-IC:\Program Files\OpenSSL-Win64\include"
LIBDIR    = "-LC:\Program Files\OpenSSL-Win64\lib"
LIBS      = -llibssl -llibcrypto

all: build run

build:
	$(CC) $(FLAGS) $(INCLUDES) password_bf.c $(LIBDIR) $(LIBS) -o $(EXE)

run:
	$(EXE) < password.in

clean:
	DEL password_bf.exe