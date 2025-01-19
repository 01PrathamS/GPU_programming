#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h> 
#include <string.h> 
#include <cuda_runtime.h>


#define INPUT_SIZE 784 
#define HIDDEN_SIZE 32 
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 1000 
#define TEST_SIZE 100
#define BATCH_SIZE 10
#define LEARNING_RATE 0.001 
#define EPOCHS 3 

typedef struct {
    float *weights1; 
    float *weights2; 
    float *bias1; 
    float *bias2;
} NeuralNetwork;

#define CUDA_CHECK(call) \
    do {\
        cudaError_t result = call; \
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \ 
        } \
    } while(0) 

// load batched img data 
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename); 
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading file: %s\n", filename); 
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// kaiming init func for weights 
void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f/ size); 
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockIdx.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m && col < k) {
        float sum = 0.0f; 
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < m && col < k) {
        float sum = 0.0f; 
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = sum;
    }
}

__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col < k) {
        float sum = 0.0f; 
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + row] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }    
}

__global__ void relu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < size) {
        x[idx] = fmax(0.0f, x[idx]);
    }
}

__global__ void bias_add_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int b = idx / size; 
    int i = idx % size; 

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x; 
    if (b < batch_size) {
        float max_val = x[b * size]; 
        for (int i = 1; i < size; ++i) {
            max_val = fmax(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val); 
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmax(x[b * size + i] / sum, 1e-7f);
        }
    }
}

__global__ void clip_gradient_kernel(float *gradients, int size, float max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < size) {
        float grad = gradients[idx]; 
        if (grad > max_norm) {
            gradients[idx] = max_norm;
        } else if (grad < -max_norm) {
            gradients[idx] = -max_norm;
        }
    }
}

void forward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int batch_size) {
    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_input, nn->weights1, d_hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    relu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_hidden, nn->weights2, d_output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    bias_add_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *d_X_train, *d_hidden, *d_output; 
    int *d_y_train; 

    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int))); 

    CUDA_CHECK(cudaMemcpy(d_X_train, X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_SIZE / BATCH_SIZE; 

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = TRAIN_SIZE / BATCH_SIZE; 

        for (epoch = 0; epoch < EPOCHS; epoch++) {
            float total_loss = 0.0f; 
            int correct = 0;

            // Zero out gradients at the beginning of each epoch 
            zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 -1) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
            zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 -1) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
            zero_grad_kernel<<<(HIDDEN_SIZE + 256 -1) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
            zero_grad_kernel<<<(OUTPUT_SIZE + 256 -1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize()); 

            for (int batch = 0; batch < num_batches; batch++) {
                int start_idx = batch * BATCH_SIZE;
                forward(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, BATCH_SIZE);

                float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float)); 
                CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
    }
}


void initialize_neural_network(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    float *h_weights1 = (float *)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *h_weights2 = (float *)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(h_weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(h_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(h_bias1, HIDDEN_SIZE);
    initialize_bias(h_bias2, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);
}

int main() {
    srand(time(NULL)); 

    NeuralNetwork nn; 
    initialize_neural_network(&nn); 

    float *X_train = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float)); 
    int *y_train = (int *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float)); 
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float)); 
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(float)); 

    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE); 
    load_labels("./data/y_train.bin", y_train, TRAIN_SIZE); 
    load_data("./data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE); 
    load_labels("./data/y_test.bin", y_test, TEST_SIZE);

    train(&nn, X_train, y_train);


}
