#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DT 0.0070710676f // delta t
#define DX 15.0f // delta x
#define DY 15.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s
#define HALF_LENGTH 1 // radius of the stencil

__global__ void calcWavelet(int* d_rows, 
                            int* d_cols, 
                            float* d_prev_base, 
                            float* d_next_base, 
                            float* d_vel_base, 
                            float* d_dxSquared,
                            float* d_dySquared,
                            float* d_dtSquared,
                            int* d_half_length) {
    
    // Thread index.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = 0;

    if ((i >= (*d_half_length)) && (j >= (*d_half_length)) && (i < (*d_rows)-(*d_half_length)) && (j < (*d_cols)-(*d_half_length))) {
        
        idx = i * (*d_rows) + j;
    
        // Neighbors in the horizontal direction
        float value = (d_prev_base[idx + 1] - 2.0 * d_prev_base[idx] + d_prev_base[idx - 1]) / (*d_dxSquared);
        
        // Neighbors in the vertical direction
        value += (d_prev_base[idx + (*d_cols)] - 2.0 * d_prev_base[idx] + d_prev_base[idx - (*d_cols)]) / (*d_dySquared);
        
        value *= (*d_dtSquared) * d_vel_base[idx];
        
        d_next_base[idx] = 2.0 * d_prev_base[idx] - d_next_base[idx] + value;

    }

}


/*
 * save the matrix on a file.txt
 */
void save_grid(int rows, int cols, float *matrix){

    system("mkdir -p wavefield");

    char file_name[64];
    sprintf(file_name, "wavefield/wavefield.txt");

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < rows; i++) {

        int offset = i * cols;

        for(int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[offset + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    
    system("python plot.py");
}


void onDevice(int h_rows, int h_cols, int h_iterations, float *h_prev_base, float *h_next_base, float *h_vel_base) {

    int *d_rows, *d_cols;
    float *d_prev_base, *d_next_base, *d_vel_base;

    // cudaError_t err;

    cudaMalloc((void**)&d_rows, sizeof(int));
	cudaMemcpy(d_rows, &h_rows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_cols, sizeof(int));
	cudaMemcpy(d_cols, &h_cols, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_prev_base, h_rows * h_cols * sizeof(float));
	cudaMemcpy(d_prev_base, h_prev_base, h_rows * h_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_next_base, h_rows * h_cols * sizeof(float));
	cudaMemcpy(d_next_base, h_next_base, h_rows * h_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_vel_base, h_rows * h_cols * sizeof(float));
	cudaMemcpy(d_vel_base, h_vel_base, h_rows * h_cols * sizeof(float), cudaMemcpyHostToDevice);

    // err = cudaGetLastError();
    // if (err != cudaSuccess) 
    //     printf("Error: %s\n", cudaGetErrorString(err));

    // Mapping variables in device that will be used in calculus.
    float *d_dxSquared, *d_dySquared, *d_dtSquared;
    float dxSquared = DX * DX;
    float dySquared = DY * DY;
    float dtSquared = DT * DT;
    cudaMalloc((void**)&d_dxSquared, sizeof(float));
	cudaMemcpy(d_dxSquared, &dxSquared, sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_dySquared, sizeof(float));
	cudaMemcpy(d_dySquared, &dySquared, sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_dtSquared, sizeof(float));
	cudaMemcpy(d_dtSquared, &dtSquared, sizeof(float), cudaMemcpyHostToDevice);

    int* d_half_length;
    cudaMalloc((void**)&d_half_length, sizeof(int));
	cudaMemset(d_half_length, HALF_LENGTH, sizeof(int));

    // Number of threads to calculate the wavelet on the the grid.
	dim3 threadsPerBlock(8, 4, 1); // 32 threads < 1 warp!
	dim3 blocksPerGrid(ceil((double)h_rows/8), ceil((double)h_cols/4), 1);

    printf("About to process!\n");

    // wavefield modeling
    for(int n = 0; n < h_iterations; n++) {

        // Launch kernel
        calcWavelet<<<blocksPerGrid, threadsPerBlock>>>(d_rows, d_cols, d_prev_base, d_next_base, d_vel_base, d_dxSquared, d_dySquared, d_dtSquared, d_half_length);
        cudaDeviceSynchronize();
        
        // swap arrays for next iteration
        float* swap = d_next_base;
        d_next_base = d_prev_base;
        d_prev_base = swap;
        
    }

    printf("I'm out!\n");

    cudaMemcpy(h_next_base, d_next_base, h_rows * h_cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_prev_base);
    cudaFree(d_next_base);
    cudaFree(d_vel_base);
    cudaFree(d_dxSquared);
    cudaFree(d_dySquared);
    cudaFree(d_dtSquared);

}

void onHost(int argc, char* argv[]) {

    if(argc != 4){
        printf("Usage: ./stencil N1 N2 TIME\n");
        printf("N1 N2: grid sizes for the stencil\n");
        printf("TIME: propagation time in ms\n");
        exit(-1);
    }

    // number of rows of the grid
    int h_rows = atoi(argv[1]);

    // number of columns of the grid
    int h_cols = atoi(argv[2]);

    // number of timesteps
    int h_time = atoi(argv[3]);
    
    // calc the number of iterations (timesteps)
    int h_iterations = (int)((h_time/1000.0) / DT);

    // Represent the matrix of wavefield as an array
    float *h_prev_base = (float*)malloc(h_rows * h_cols * sizeof(float));
    memset(h_prev_base, 0.0f, h_rows * h_cols * sizeof(float));
    float *h_next_base = (float*)malloc(h_rows * h_cols * sizeof(float));
    memset(h_next_base, 0.0f, h_rows * h_cols * sizeof(float));

    // represent the matrix of velocities as an array
    float *h_vel_base = (float*)malloc(h_rows * h_cols * sizeof(float));
    memset(h_vel_base, V * V, h_rows * h_cols * sizeof(float));

    printf("Grid Sizes: %d x %d\n", h_rows, h_cols);
    printf("Iterations: %d\n", h_iterations);

    // ************* BEGIN INITIALIZATION *************

    printf("Initializing ... \n");

    // Ricker Wavelet.
    // define source wavelet
    float h_wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                           0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                           0.023680172, 0.005611435,  0.001823209,  -0.000720549};

    // add a source to initial wavefield as an initial condition
    for(int s = 11; s >= 0; s--){
        for(int i = h_rows / 2 - s; i < h_rows / 2 + s; i++){

            int offset = i * h_cols;

            for(int j = h_cols / 2 - s; j < h_cols / 2 + s; j++)
                h_prev_base[offset + j] = h_wavelet[s];
        }
    }

    // ************** END INITIALIZATION **************

    printf("Computing wavefield ... \n");

    onDevice(h_rows, h_cols, h_iterations, h_prev_base, h_next_base, h_vel_base);

    save_grid(h_rows, h_cols, h_next_base);

    free(h_prev_base);
    free(h_next_base);
    free(h_vel_base);

}


int main(int argc, char* argv[]) {

    onHost(argc, argv);

    return 0;
}