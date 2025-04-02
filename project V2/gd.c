#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <time.h>

#define N 28
#define EPOCHS 100
#define NUM_CLASSES 4

int softmax(float *z, int num_classes, float *output) {
    float max_z = z[0];
    int i,max=0;
    for (i = 1; i < num_classes; i++) {
        if (z[i] > max_z) max_z = z[i];
    }
    
    float sum = 0.0f;
    for (i = 0; i < num_classes; i++) {
        output[i] = exp(z[i] - max_z);
        sum += output[i];
    }
    for (i = 0; i < num_classes; i++) {
        output[i] /= sum;
        if(output[i]>output[max]){
        	max=i;
		}
    }
    return max;
}

void gradient_descent(const char *dir_path, const char *dir2_path) {
    struct dirent *entry;
    DIR *dp = opendir(dir_path);
    int epoch,i,c,current_index;
    float train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    clock_t start, end;
	float epoch_time,total_time=0;
	
	FILE *w_record = fopen("results/w_gd.txt", "w");
    if (w_record == NULL) {
        printf("Dosya açýlamadý.\n");
        return;
    }

    float w[N * N + 1][NUM_CLASSES], learning_rate = 0.001f;
    for (i = 0; i < N * N + 1; i++) {
        for (c = 0; c < NUM_CLASSES; c++) {
            w[i][c] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            fprintf(w_record, "%f ", w[i][c]);
        }
    }
	fprintf(w_record, "\n");
    
    if (dp == NULL) {
        perror("Dizin açýlamadý");
        return;
    }
    
    for (epoch = 0; epoch < EPOCHS; epoch++) {
    	start = clock();
		float mse = 0.0f,True=0,False=0;
		current_index = 0;
		
		float gradients[N * N + 1][NUM_CLASSES] = {{0.0f}};
		while ((entry = readdir(dp)) != NULL) {
		    const char *filename = entry->d_name;
		
		    if (filename[0] == '.') {
		        continue;
		    }
		
		    // Dosya yolunu oluþtur
		    char file_path[512];
		    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, filename);
		
		    // Dosyayý aç
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("Dosya açýlamadý: %s\n", file_path);
		        continue;
		    }
		
		    float x_train[N*N] = {0.00f};
		    
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_train[i]) != 1) {
			        printf("Dosyadan deðer okunamadý: %s\n", file_path);
			        fclose(input_file);
			        exit(-1);
			    }
			}
		    fclose(input_file);
		
		    int label = filename[0] - '0';
		    
		    float z[NUM_CLASSES] = {0.0f};
            for (c = 0; c < NUM_CLASSES; c++) {
                for (i = 0; i < N * N; i++) {
                    z[c] += w[i][c] * x_train[i];
                }
                z[c] += w[N * N][c]; // Bias ekleme
            }	
			
			float softmax_output[NUM_CLASSES];
			int prediction = softmax(z, NUM_CLASSES, softmax_output);
				
			
			if(prediction == label){
				True++;
			}else{
				False++;
			}
			
			mse -= log(softmax_output[label]);  // eðer 1 olursa hata 0 olur
			
			for (c = 0; c < NUM_CLASSES; c++) {
                float error = softmax_output[c] - (c == label ? 1.0f : 0.0f);
                for (i = 0; i < N * N; i++) {
                    gradients[i][c] += error * x_train[i];
                }
                gradients[N * N][c] += error; // Bias için gradyan
            }
		    
		    current_index++;
		}
		
		for (c = 0; c < NUM_CLASSES; c++) {
            for (i = 0; i < N * N + 1; i++) {
                //w[i][c] -= learning_rate * gradients[i][c] / current_index;
                w[i][c] -= learning_rate * gradients[i][c];
                fprintf(w_record, "%f ", w[i][c]);
            }
        }
		fprintf(w_record, "\n");
		
		mse /= current_index;
		train_losses[epoch] = mse;
		train_accuracies[epoch] = (True / (True + False)) * 100;
		
		end = clock();
		epoch_time = ((float) (end - start)) / CLOCKS_PER_SEC;
		total_time += epoch_time;
		train_times[epoch] = total_time;
		
		
		if(epoch%1==0){
			printf("Epoch %d: MSE = %f, Accuricy: %f, Time:%f \n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch]);
		}

        rewinddir(dp);
	}
	fclose(w_record);
	closedir(dp); 
	
	FILE *file = fopen("results/results_gd.txt", "w");
	for (epoch = 0; epoch < EPOCHS; epoch++) {
    	fprintf(file, "%d %f %f %f\n", epoch,train_losses[epoch],train_accuracies[epoch], train_times[epoch]);
	}
	fclose(file);
	
	// test
	DIR *dp2 = opendir(dir2_path);
	if (dp2 == NULL) {
        perror("Dizin açýlamadý");
        return;
    }
    float True=0,False=0;
    while ((entry = readdir(dp2)) != NULL) {
	    const char *filename = entry->d_name;
	
	    if (filename[0] == '.') {
	        continue;
	    }
	
	    // Dosya yolunu oluþtur
	    char file_path[512];
	    snprintf(file_path, sizeof(file_path), "%s/%s", dir2_path, filename);
	
	    // Dosyayý aç
	    FILE *input_file = fopen(file_path, "r");
	    if (!input_file) {
	        printf("Dosya açýlamadý: %s\n", file_path);
	        continue;
	    }
	
	    float x_test[N*N] = {0.00f};
	    
	    for (i = 0; i < N * N; i++) {
		    if (fscanf(input_file, "%f", &x_test[i]) != 1) {
		        printf("Dosyadan deðer okunamadý: %s\n", file_path);
		        fclose(input_file);
		        exit(-1);
		    }
		}
	    fclose(input_file);
	
	    int label = filename[0] - '0';
	    
	    float logits[NUM_CLASSES] = {0.0f};
	    for (c = 0; c < NUM_CLASSES; c++) {
	        for (i = 0; i < N * N; i++) {
	            logits[c] += w[i][c] * x_test[i];
	        }
	        logits[c] += w[N * N][c];
	    }
	    
	    /*float exp_sum = 0.0f;
	    float probabilities[NUM_CLASSES];
	    for (c = 0; c < NUM_CLASSES; c++) {
	        probabilities[c] = exp(logits[c]);
	        exp_sum += probabilities[c];
	    }
	    for (c = 0; c < NUM_CLASSES; c++) {
	        probabilities[c] /= exp_sum; // Normalize et
	    }
	    
	    // En yüksek olasýlýða sahip sýnýfý seç
	    prediction = 0;
	    float max_prob = probabilities[0];
	    for (c = 1; c < NUM_CLASSES; c++) {
	        if (probabilities[c] > max_prob) {
	            max_prob = probabilities[c];
	            prediction = c;
	        }
	    }*/
	    
	    float probabilities[NUM_CLASSES];
		int prediction = softmax(logits, NUM_CLASSES, probabilities);
	    
	    if (label == prediction) {
	        True++;
	    } else {
	        False++;
	    }
			    
	    printf("dogruluk: %f (%.f/%.f)\n", True/(False+True)*100,True,True+False);
	    
	    current_index++;
	}
}

int main() {
	srand(time(NULL));
    const char *output1_directory = "train_set";
    const char *output2_directory = "test_set";
    
    gradient_descent(output1_directory, output2_directory);
    return 0;
}

