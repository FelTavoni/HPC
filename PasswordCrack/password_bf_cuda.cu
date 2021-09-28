#include <stdio.h>
#include <stdlib.h>
#include <openssl/md5.h>
#include <string.h>

#define MAX 10

// MD5_DIGEST_LENGTH
// MD5_DIGEST_LENGTH is the length of an MD5 digest, usually 16.

typedef unsigned char byte;

// Uppercase and lower-case caracters, different from the sequential code...
char h_letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";

/*
 * Print a digest of MD5 hash.
*/
void print_digest(byte * hash){
	int x;

	for(x = 0; x < MD5_DIGEST_LENGTH; x++)
        	printf("%02x", hash[x]);
	printf("\n");
}

/*
 * Convert hexadecimal string to hash byte.
*/
void strHex_to_byte(char * str, byte * hash){
	char * pos = str;
	int i;

	for (i = 0; i < MD5_DIGEST_LENGTH/sizeof *hash; i++) {
		sscanf(pos, "%2hhx", &hash[i]);
		pos += 2;
	}
}

__global__ void testSolution(char *d_letters, ) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    char *myAns;
    cudaMalloc((void**)&myAns, idx * sizeof(unsigned char));
    int *aux;
    cudaMalloc((void**)&i, sizeof(int));
    *i = 0;

	str[idx] = letters[c];
	MD5((byte *) str, strlen(str), hash2);
	if(strncmp((char*)hash1, (char*)hash2, MD5_DIGEST_LENGTH) == 0){
		printf("found: %s\n", str);
		//print_digest(hash2);
		*ok = 1;
	}

    cudaFree(myAns);
    cudaFree(pos);
}

/*
 * This procedure generate all combinations of possible letters
*/
void iterate(byte *hash1, byte *hash2, char *str, int idx, int len, int *ok) {
	int c;

	// 'ok' determines when the algorithm matches.
	if(*ok) return;
	if (idx < (len - 1)) {
		// Iterate for all letter combination.
		for (c = 0; c < strlen(letters) && *ok==0; ++c) {
			str[idx] = letters[c];
			// Recursive call
			iterate(hash1, hash2, str, idx + 1, len, ok);
		}
	} else {
		// Include all last letters and compare the hashes.
		testSolution(str, ok);
	}
}

void onDevice(byte *h_hash1, byte *h_hash2, int h_lenMax, int *h_ok, char *h_str) {

    // Defining the grid. It's 63 possible final caracters.
	dim3 threadsPerBlock(64, 1, 1); // 64 threads = 2 warps!
	dim3 blocksPerGrid(1, 1, 1);

    // Generate all possible passwords of different sizes.
    // Words with 1 caracter, words with 2 caracter and so on...

	int h_len = 1;

	while (h_len <= h_lenMax && !(*h_ok))

	for(h_len = 1; h_len <= h_lenMax; h_len++){
		memset(h_str, 0, h_len+1);
		
		int c;

		// 'ok' determines when the algorithm matches.
		if(*ok) return;
		if (idx < (len - 1)) {
			// Iterate for all letter combination.
			for (c = 0; c < strlen(letters) && *ok==0; ++c) {
				str[idx] = letters[c];
				// Recursive call
				iterate(hash1, hash2, str, idx + 1, len, ok);
			}
		} else {
			// Include all last letters and compare the hashes.
			testSolution(str, ok);
		}
        
	}
}

void onHost() {
    char h_str[MAX+1];                        // Return hash string.
	int h_lenMax = MAX;                       // Input maximum size.
	int h_len;                                // _ _ _
	int h_ok = 0, r;                          // Found sequence and return type of scanf.
	char h_hash1_str[2*MD5_DIGEST_LENGTH+1];  // Input hash string. (char/str)
	byte h_hash1[MD5_DIGEST_LENGTH];          // password hash (byte)
	byte h_hash2[MD5_DIGEST_LENGTH];          // string hashes (byte)

	// Input:
	r = scanf("%s", hash1_str);

	// Check input.
	if (r == EOF || r == 0)
	{
		fprintf(stderr, "Error!\n");
		exit(1);
	}

	// Convert hexadecimal string to hash byte.
	strHex_to_byte(h_hash1_str, h_hash1);

	memset(h_hash2, 0, MD5_DIGEST_LENGTH);
	//print_digest(hash1);

	// Calling device to start checking all possibilities.
	onDevice(h_hash1, h_hash2, h_ok, h_str);

	printf("found: %s\n", h_str);
}

int main(int argc, char **argv) {
	onHost();
}
