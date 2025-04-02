#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <float.h>
#include <string.h>

// Gri tonlara d�n��t�rme fonksiyonu
void convert_to_grayscale(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels) {
    int i;
    for (i = 0; i < width * height; i++) {
        int r = input_image[i * channels];
        int g = input_image[i * channels + 1];
        int b = input_image[i * channels + 2];

        unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        output_image[i] = gray;
    }
}

// Normalize etme fonksiyonu (0-255 aras� piksel de�erlerini 0-1 aras� vekt�re d�n��t�rme)
void normalize_image(unsigned char *input_image, float *output_vector, int width, int height) {
    int i;
    for (i = 0; i < width * height; i++) {
        output_vector[i] = (float)input_image[i] / 255.0f;
    }
}

// Boyutland�rma fonksiyonu (resmi yeniden boyutland�rma)
void resize_image(unsigned char *input_image, unsigned char *output_image, int input_width, int input_height, int output_width, int output_height, int channels) {
    float x_ratio = input_width / (float)output_width;
    float y_ratio = input_height / (float)output_height;
    int i, j;

    for (i = 0; i < output_height; i++) {
        for (j = 0; j < output_width; j++) {
            int x = (int)(j * x_ratio);
            int y = (int)(i * y_ratio);

            int c;
            for (c = 0; c < channels; c++) {
                output_image[(i * output_width + j) * channels + c] = input_image[(y * input_width + x) * channels + c];
            }
        }
    }
}

// Etiketlerin belirlenmesi (resim adlar�na g�re)
void set_labels(const char *filename, int *label) {
    if (strstr(filename, "A") != NULL) {
        *label = 1;  // A s�n�f�
    } else {
        *label = -1; // B s�n�f�
    }
}

// Tanh fonksiyonu
float tanh_activation(float x) {
    return tanh(x);
}

// ��k�� hesaplama fonksiyonu
float compute_output(float *w, float *x, int input_size) {
	int i;
    float dot_product = 0.0f;
    for (i = 0; i < input_size; i++) {
        dot_product += w[i] * x[i];
    }
    dot_product += w[input_size];  // Bias terimi
    return tanh_activation(dot_product);  // ��k��� hesapla
}

// Kay�p fonksiyonu (Mean Squared Error)
float compute_loss(float *predictions, int *labels, int num_samples) {
    float loss = 0.0f;
    int i;
	for (i = 0; i < num_samples; i++) {
        float error = predictions[i] - labels[i];
        loss += error * error;
    }
    return loss / num_samples;
}

// A��rl�k gradyanlar�n� hesaplama
void compute_gradients(float *w, float *x, int *labels, float *gradients, int input_size, float learning_rate, int num_samples) {
    int i,j;
	for (i = 0; i < input_size; i++) {
        gradients[i] = 0.0f;
        for (j = 0; j < num_samples; j++) {
            float prediction = compute_output(w, x + j * input_size, input_size);
            float error = prediction - labels[j];
            gradients[i] += error * x[j * input_size + i];
        }
        gradients[i] /= num_samples;
    }

    // Bias gradyan�
    gradients[input_size] = 0.0f;
	for (j = 0; j < num_samples; j++) {
        float prediction = compute_output(w, x + j * input_size, input_size);
        float error = prediction - labels[j];
        gradients[input_size] += error;
    }
    gradients[input_size] /= num_samples;
}

// Gradyan ini�i (Gradient Descent)
void gradient_descent(float *w, float *x, int *labels, int input_size, int num_samples, float learning_rate, int epochs) {
    float *gradients = (float *)malloc((input_size + 1) * sizeof(float));
	int epoch, i;
    for (epoch = 0; epoch < epochs; epoch++) {
        compute_gradients(w, x, labels, gradients, input_size, learning_rate, num_samples);

        // A��rl�klar� g�ncelle
        for (i = 0; i < input_size + 1; i++) {
            w[i] -= learning_rate * gradients[i];
        }

        // Kay�p fonksiyonunu hesapla
        if (epoch % 10 == 0) {
            float *predictions = (float *)malloc(num_samples * sizeof(float));
            for (i = 0; i < num_samples; i++) {
                predictions[i] = compute_output(w, x + i * input_size, input_size);
            }
            float loss = compute_loss(predictions, labels, num_samples);
            printf("Epoch %d: Loss = %f\n", epoch, loss);
            free(predictions);
        }
    }

    free(gradients);
}

// Resimleri i�leme ve etiketleri ayarlama
void process_images_in_directory(const char *input_dir, const char *output_dir) {
	int i;
    struct dirent *entry;
    DIR *dp = opendir(input_dir);
    
    if (dp == NULL) {
        perror("Dizin a��lamad�");
        return;
    }

    // Dizin i�inde gezin
    while ((entry = readdir(dp)) != NULL) {
        const char *filename = entry->d_name;
        if (filename[0] == '.') continue;

        // Resmin tam yolunu olu�tur
        char input_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_dir, filename);

        // Resmi y�kle
        int width, height, channels;
        unsigned char *input_image = stbi_load(input_path, &width, &height, &channels, 0);
        if (!input_image) {
            printf("Resim y�klenemedi: %s\n", input_path);
            continue;
        }

        // Boyutland�rma ve gri tonlara d�n��t�rme
        int output_width = 30, output_height = 30;
        unsigned char *output_image = (unsigned char *)malloc(output_width * output_height);
        if (!output_image) {
            printf("Bellek ayr�lmad�.\n");
            stbi_image_free(input_image);
            continue;
        }

        unsigned char *resized_image = (unsigned char *)malloc(width * height * channels);
        resize_image(input_image, resized_image, width, height, output_width, output_height, channels);
        convert_to_grayscale(resized_image, output_image, output_width, output_height, channels);

        // Normalize edilmi� vekt�r
        float *output_vector = (float *)malloc(output_width * output_height * sizeof(float));
        normalize_image(output_image, output_vector, output_width, output_height);

        // Etiketi belirle
        int label;
        set_labels(filename, &label);

        // Resim ve vekt�r� kaydet
        char output_vector_path[512];
        snprintf(output_vector_path, sizeof(output_vector_path), "%s/gray_%s_vector.txt", output_dir, filename);
        FILE *f = fopen(output_vector_path, "w");
        if (f) {
            for (i = 0; i < output_width * output_height; i++) {
                fprintf(f, "%f ", output_vector[i]);
                if ((i + 1) % output_width == 0) {
                    fprintf(f, "\n");
                }
            }
            fclose(f);
        } else {
            printf("Vekt�r kaydedilemedi: %s\n", output_vector_path);
        }

        // E�itim i�in haz�rl�k
        int num_samples = 1;  // Bu �rnekte sadece bir �rnek var
        int input_size = output_width * output_height;
        float *w = (float *)malloc((input_size + 1) * sizeof(float));  // A��rl�klar
        for (i = 0; i < input_size + 1; i++) {
            w[i] = ((float)rand() / RAND_MAX) * 0.1f;  // Rastgele a��rl�klar
        }

        // Gradyan ini�i ba�lat
        gradient_descent(w, output_vector, &label, input_size, num_samples, 0.01f, 100);

        // Temizlik
        free(output_image);
        free(resized_image);
        free(output_vector);
        stbi_image_free(input_image);
    }

    closedir(dp);
}

int main() {
    const char *input_dir = "./images";  // Resimlerin oldu�u dizin
    const char *output_dir = "./output"; // ��kt�lar�n kaydedilece�i dizin

    process_images_in_directory(input_dir, output_dir);

    return 0;
}

