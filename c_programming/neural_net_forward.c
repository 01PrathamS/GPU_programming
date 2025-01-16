#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h> 
#include <string.h> 

#define INPUT_SIZE 784 
#define HIDDEN_SIZE 1 
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001
#define EPOCHS 10 
#define BATCH_SIZE 4 
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000

typedef struct {
    float *weights1; 
    float *weights2; 
    float *bias1; 
    float *bias2;
} NeuralNetwork;

// load batched img data 
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);   
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n",size, read_size);
        exit(1);
    }
    fclose(file);
}

// void batch labels 
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename); 
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size); 
        exit(1);
    }
    fclose(file);
}

// kaiming init func for weights 
void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size); 
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// initialize bias 
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// softmax to work with batches 
void softmax(float *x, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        float max = x[b * size]; 
        for (int i = 1; i < size; i++) {
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        float sum = 0.0f; 
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max); 
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmax(x[b * size + i] / sum, 1e-7f);
        }
    }
}

void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f; 
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

// Matrix multiplication A @ B.T
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

// Matrix multiplication A.T @ B
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++) {
                C[i * k + j] += A[l * n + i] * B[l * k + j];
            }
        }
    }
}

// ReLU forward
void relu_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

// Add bias
void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

void forward(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size) {
    matmul_a_b(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    bias_forward(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    relu_forward(hidden, batch_size * HIDDEN_SIZE);
    matmul_a_b(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    softmax(output, batch_size, OUTPUT_SIZE);
}



// traning function for forward pass 
void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE; 

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f; 
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE; 
            forward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, BATCH_SIZE);
        }

        // IMPLEMENT BACKPROPAGATION HERE

    
    }
    free(hidden);
    free(output);
}

void initialize_neural_network(NeuralNetwork *nn) {
    nn->weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}

int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = malloc(TEST_SIZE * sizeof(int));

    load_data("../mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("../mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_test.bin", y_test, TEST_SIZE);

    train(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
