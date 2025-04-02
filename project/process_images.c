#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#define N 28

int data_size = 0;

void convert_to_grayscale(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels) {
    int i;
    for (i = 0; i < width * height; i++) {
        int r = input_image[i * channels];
        int g = input_image[i * channels + 1];
        int b = input_image[i * channels + 2];

        // Gri tonu hesapla (Y = 0.299R + 0.587G + 0.114B form�l�yle)
        unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        output_image[i] = gray;
    }
}

void normalize_image(unsigned char *input_image, float *output_vector, int width, int height) {
    int i;
    for (i = 0; i < width * height; i++) {
        output_vector[i] = (float)input_image[i] / 255.0f;  // 0-255 aras� de�eri 0-1 aras�na d�n��t�r
    }
}

// Boyutland�rma i�lemi
void resize_image(unsigned char *input_image, unsigned char *output_image, int input_width, int input_height, int output_width, int output_height, int channels) {
    float x_ratio = input_width / (float)output_width;
    float y_ratio = input_height / (float)output_height;
    int i, j;

    for (i = 0; i < output_height; i++) {
        for (j = 0; j < output_width; j++) {
            int x = (int)(j * x_ratio);
            int y = (int)(i * y_ratio);

            // Resmin renk kanal�n� al
            int c;
			for (c = 0; c < channels; c++) {
                output_image[(i * output_width + j) * channels + c] = input_image[(y * input_width + x) * channels + c];
            }
        }
    }
}

void process_images_in_directory(const char *input_dir, const char *output_dir) {
    struct dirent *entry;
    DIR *dp = opendir(input_dir);
    
    if (dp == NULL) {
        perror("Dizin a��lamad�");
        return;
    }

    // Dizin i�inde gezmek i�in
    while ((entry = readdir(dp)) != NULL) {
        // Dosya ad�n� al
        const char *filename = entry->d_name;

        // "." ve ".." dizinlerini atla
        if (filename[0] == '.') {
            continue;
        }
    	data_size ++;
    	printf("%d",data_size);

        // Tam giri� dosya yolunu olu�tur
        char input_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_dir, filename);

        // Resmi y�kle
        int width, height, channels;
        unsigned char *input_image = stbi_load(input_path, &width, &height, &channels, 0);
        if (!input_image) {
            printf("Resim y�klenemedi: %s\n", input_path);
            continue;
        }

        printf("Isleniyor: %s (%dx%d, Kanallar: %d)\n", input_path, width, height, channels);

        // Yeni NxN'luk boyutta gri tonlu resim i�in bellek ay�r
        int output_width = N, output_height = N;
        unsigned char *output_image = (unsigned char *)malloc(output_width * output_height);
        if (!output_image) {
            printf("Bellek ayr�lmad�.\n");
            stbi_image_free(input_image);
            continue;
        }

        // Resmi NxN'a boyutland�r
        unsigned char *resized_image = (unsigned char *)malloc(width * height * channels);
        if (!resized_image) {
            printf("Boyutland�r�lm�� resim i�in bellek ayr�lmad�.\n");
            free(output_image);
            stbi_image_free(input_image);
            continue;
        }

        resize_image(input_image, resized_image, width, height, output_width, output_height, channels);

        // Gri tonlara d�n��t�r
        convert_to_grayscale(resized_image, output_image, output_width, output_height, channels);

        // Normalize edilmi� vekt�r i�in bellek ay�r
        float *output_vector = (float *)malloc(output_width * output_height * sizeof(float));  // Orijinal boyutlarda vekt�r
        if (!output_vector) {
            printf("Vekt�r i�in bellek ayr�lmad�.\n");
            free(output_image);
            free(resized_image);
            stbi_image_free(input_image);
            continue;
        }

        // Normalize et
        normalize_image(output_image, output_vector, output_width, output_height);

        // ��k�� dosya ad�n� olu�tur
        // Dosyan�n bulundu�u klas�r�n ismini al
        char directory_name[512];
        char *dir_name = strrchr(input_path, '/');  // Son '/' karakterini bul

        if (dir_name != NULL) {
            // Klas�r ismini al (dir_name bir pointer oldu�undan, ba�lang�� noktas�na kadar kopyalayaca��z)
            strncpy(directory_name, input_path, dir_name - input_path);
            directory_name[dir_name - input_path] = '\0';  // Klas�r ad�n� sonland�r
        } else {
            // E�er '/' bulunmazsa, directory_name'i bo� b�rak�yoruz (bu durumda dosya, k�k dizinde olur)
            directory_name[0] = '\0';
        }

        // Klas�r ad� ile i�lemi yap
        char new_filename[512];
        if (strncmp(directory_name, "zero", 4) == 0) {  // E�er klas�r ad� "zero" i�eriyorsa
            snprintf(new_filename, sizeof(new_filename), "0");
        } else if (strncmp(directory_name, "one", 3) == 0) {  // E�er klas�r ad� "one" i�eriyorsa
            snprintf(new_filename, sizeof(new_filename), "1");
        } else {
            snprintf(new_filename, sizeof(new_filename), "%s", filename);  // Klas�r ad� ba�ka bir �eyse oldu�u gibi b�rak
        }
        
        char output_path[512];
		snprintf(output_path, sizeof(output_path), "%s/%s_%d.txt", output_dir, new_filename, data_size);

        // Vekt�r� dosyaya kaydet
        FILE *f = fopen(output_path, "w");
        if (f) {
        	int i;
            for (i = 0; i < output_width * output_height; i++) {
                fprintf(f, "%f ", output_vector[i]);  // Vekt�r� yaz
            }
            fclose(f);
            printf("Vekt�r kaydedildi: %s\n", output_path);
        } else {
            printf("Vekt�r kaydedilemedi: %s\n", output_path);
        }

        /*// Gri tonlu resmi kaydet
        char gray_image_path[512];
        snprintf(gray_image_path, sizeof(gray_image_path), "%s/gray_%s", output_dir, filename);
        if (!stbi_write_jpg(gray_image_path, output_width, output_height, 1, output_image, 100)) {
            printf("Gri tonlu resim kaydedilemedi: %s\n", gray_image_path);
        } else {
            printf("Gri tonlu resim kaydedildi: %s\n", gray_image_path);
        }*/

        // Belle�i serbest b�rak
        free(output_image);
        free(resized_image);
        free(output_vector);
        stbi_image_free(input_image);
    }

    closedir(dp);
}


int main() {
    const char *input1_directory = "zero_train";
    const char *input2_directory = "one_train";
    const char *output1_directory = "train_set";
    const char *input3_directory = "zero_test";
    const char *input4_directory = "one_test";
    const char *output2_directory = "test_set";

    process_images_in_directory(input1_directory, output1_directory);
    process_images_in_directory(input2_directory, output1_directory);
    printf("train set: %d\n\n\n",data_size);
    
    process_images_in_directory(input3_directory, output2_directory);
    process_images_in_directory(input4_directory, output2_directory);
    printf("total set: %d",data_size);
    return 0;
}

