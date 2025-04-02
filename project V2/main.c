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
#define EPOCHS 200
#define NUM_CLASSES 4

float w_first[N * N + 1][NUM_CLASSES];

void initialize_weights() {
    int i,c;
    for (i = 0; i < N * N + 1; i++) {
        for (c = 0; c < NUM_CLASSES; c++) {
            w_first[i][c] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

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
    float test_losses[EPOCHS], test_accuracies[EPOCHS];
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
            w[i][c] = w_first[i][c];
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

        rewinddir(dp);
        
        
        
        // test
		DIR *dp2 = opendir(dir2_path);
		if (dp2 == NULL) {
	        perror("Dizin açýlamadý");
	        return;
	    }
	    True=0,False=0;
		float mse_test=0, test_size=0;
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
		    
		    float z[NUM_CLASSES] = {0.0f};
		    for (c = 0; c < NUM_CLASSES; c++) {
		        for (i = 0; i < N * N; i++) {
		            z[c] += w[i][c] * x_test[i];
		        }
		        z[c] += w[N * N][c];
		    }
		    
		    float probabilities[NUM_CLASSES];
			int prediction = softmax(z, NUM_CLASSES, probabilities);
		    
		    if (label == prediction) {
		        True++;
		    } else {
		        False++;
		    }
		    
		    mse_test -= log(probabilities[label]);
		    //printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		    test_size++;
		}

		mse_test /=  test_size;
		test_losses[epoch] = mse_test;
		test_accuracies[epoch] =  True/(False+True)*100;
		if(epoch%5==0){
			printf("Epoch %d: MSE = %f, Accuricy: %f, Time:%f, test_mse: %f, test_Accuricy: %f\n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch],mse_test,test_accuracies[epoch]);
		}
		// end of test
		
	}
	fclose(w_record);
	closedir(dp); 
	
	FILE *file = fopen("results/results_gd.txt", "w");
	for (epoch = 0; epoch < EPOCHS; epoch++) {
    	fprintf(file, "%d %f %f %f %f %f\n", epoch,train_losses[epoch],train_accuracies[epoch], train_times[epoch],test_losses[epoch], test_accuracies[epoch]);
	}
	fclose(file);	
}


void sgd(const char *dir_path, const char *dir2_path) {
    struct dirent *entry;
    DIR *dp = opendir(dir_path);
    int epoch,i,c,current_index,batch=0;
    clock_t start, end;
    float epoch_time,total_time=0;
    float train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    float test_losses[EPOCHS], test_accuracies[EPOCHS];
    
    FILE *w_record = fopen("results/w_sgd.txt", "w");
    if (w_record == NULL) {
        printf("Dosya açýlamadý.\n");
        return;
    }
    
    float w[N * N + 1][NUM_CLASSES], learning_rate = 0.01f;
    for (i = 0; i < N * N + 1; i++) {
        for (c = 0; c < NUM_CLASSES; c++) {
            w[i][c] = w_first[i][c];
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
    	if (epoch % 30 == 0 && epoch > 0) {
		    learning_rate *= 0.80;
		    printf("%f \n",learning_rate);
		}
		float mse = 0.0f, True=0, False=0;
		current_index = 0;
		
		float gradients[N * N + 1][NUM_CLASSES] = {{0.0f}};	
			
		for(batch=0;batch<64;batch++){
			// Dosya yolunu oluþtur
			int label;
		    char file_path[512];
		    if (batch % 4 == 0) {
		        snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "0_", rand() % 500 + 1, ".txt");
		        label = 0; // Sýnýf 0
		    } else if (batch % 4 == 1) {
		        snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "1_", rand() % 500 + 501, ".txt");
		        label = 1; // Sýnýf 1
		    } else if (batch % 4 == 2) {
		        snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "2_", rand() % 500 + 1001, ".txt");
		        label = 2; // Sýnýf 2
		    } else {
		        snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "3_", rand() % 500 + 1501, ".txt");
		        label = 3; // Sýnýf 3
		    }
		
		    // Dosyayý aç
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("\n\nDosya açýlamadý: %s\n", file_path);
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
			
			mse -= log(softmax_output[label]);  // e?er 1 olursa hata 0 olur
			
			for (c = 0; c < NUM_CLASSES; c++) {
                float error = softmax_output[c] - (c == label ? 1.0f : 0.0f);
                for (i = 0; i < N * N; i++) {
                    gradients[i][c] += error * x_train[i];
                }
                gradients[N * N][c] += error; // Bias i?in gradyan
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
		
		mse /=  current_index;
		train_losses[epoch] = mse;
		train_accuracies[epoch] = (True / (True + False)) * 100;
		
		end = clock();
		epoch_time = ((float) (end - start)) / CLOCKS_PER_SEC;
		total_time += epoch_time;
		train_times[epoch] = total_time;
		
		 // test
		DIR *dp2 = opendir(dir2_path);
		if (dp2 == NULL) {
	        perror("Dizin açýlamadý");
	        return;
	    }
	    True=0,False=0;
		float mse_test=0, test_size=0;
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
		    
		    float z[NUM_CLASSES] = {0.0f};
		    for (c = 0; c < NUM_CLASSES; c++) {
		        for (i = 0; i < N * N; i++) {
		            z[c] += w[i][c] * x_test[i];
		        }
		        z[c] += w[N * N][c];
		    }
		    
		    float probabilities[NUM_CLASSES];
			int prediction = softmax(z, NUM_CLASSES, probabilities);
		    
		    if (label == prediction) {
		        True++;
		    } else {
		        False++;
		    }
		    
		    mse_test -= log(probabilities[label]);
		    //printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		    test_size++;
		}

		mse_test /=  test_size;
		test_losses[epoch] = mse_test;
		test_accuracies[epoch] =  True/(False+True)*100;
		if(epoch%5==0){
			printf("Epoch %d: MSE = %f, Accuricy: %f, Time:%f, test_mse: %f, test_Accuricy: %f\n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch],mse_test,test_accuracies[epoch]);
		}
		// end of test
	}
	
	fclose(w_record);

	FILE *file = fopen("results/results_sgd.txt", "w");
	for (epoch = 0; epoch < EPOCHS; epoch++) {
    	fprintf(file, "%d %f %f %f %f %f\n", epoch,train_losses[epoch],train_accuracies[epoch], train_times[epoch],test_losses[epoch], test_accuracies[epoch]);
	}
	fclose(file);
    
}


void adam(const char *dir_path, const char *dir2_path) {
    struct dirent *entry;
    DIR *dp = opendir(dir_path);
    int epoch,i,c,current_index;
    clock_t start, end;
    float m[N*N+1][NUM_CLASSES] = {{0.0f}}, v[N * N + 1][NUM_CLASSES] = {{0.0f}}, epoch_time,total_time=0;
    float beta1 = 0.8f, beta2 = 0.8f, epsilon = 1e-8f, learning_rate = 0.01f;
    float train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    float test_losses[EPOCHS], test_accuracies[EPOCHS];
    
    FILE *w_record = fopen("results/w_adam.txt", "w");
    if (w_record == NULL) {
        printf("Dosya açýlamadý.\n");
        return;
    }
    
    float w[N * N + 1][NUM_CLASSES];
    for (i = 0; i < N * N + 1; i++) {
        for (c = 0; c < NUM_CLASSES; c++) {
            w[i][c] = w_first[i][c];
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
		float mse = 0.0f, True=0, False=0;;
		current_index = 0;
		float gradients[N * N + 1][NUM_CLASSES] = {{0.0f}};
		
		while ((entry = readdir(dp)) != NULL) {
		    const char *filename = entry->d_name;
		
		    if (filename[0] == '.') {
		        continue;
		    }
		
		    char file_path[512];
		    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, filename);

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
			
			mse -= log(softmax_output[label]);
			
			for (c = 0; c < NUM_CLASSES; c++) {
                float error = softmax_output[c] - (c == label ? 1.0f : 0.0f);
                for (i = 0; i < N * N; i++) {
                    gradients[i][c] += error * x_train[i];
                }
                gradients[N * N][c] += error; // Bias icin gradyan
            }
		    
		    current_index++;
		}
		
		for (c = 0; c < NUM_CLASSES; c++) {
		    for (i = 0; i < N * N + 1; i++) {
		        m[i][c] = beta1 * m[i][c] + (1 - beta1) * gradients[i][c];
		        v[i][c] = beta2 * v[i][c] + (1 - beta2) * (gradients[i][c] * gradients[i][c]);
	        
		        float m_hat = m[i][c] / (1 - powf(beta1, epoch + 1));
		        float v_hat = v[i][c] / (1 - powf(beta2, epoch + 1));
		
		        w[i][c] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
		
		        fprintf(w_record, "%f ", w[i][c]);
		    }
		}
        fprintf(w_record, "\n");
        
        mse /=  current_index;
        train_losses[epoch] = mse;
		train_accuracies[epoch] = (True / (True + False)) * 100;
		
		end = clock();
		epoch_time = ((float) (end - start)) / CLOCKS_PER_SEC;
		total_time += epoch_time;
		train_times[epoch] = total_time;
        rewinddir(dp);
        
        // test
		DIR *dp2 = opendir(dir2_path);
		if (dp2 == NULL) {
	        perror("Dizin açýlamadý");
	        return;
	    }
	    True=0,False=0;
		float mse_test=0, test_size=0;
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
		    
		    float z[NUM_CLASSES] = {0.0f};
		    for (c = 0; c < NUM_CLASSES; c++) {
		        for (i = 0; i < N * N; i++) {
		            z[c] += w[i][c] * x_test[i];
		        }
		        z[c] += w[N * N][c];
		    }
		    
		    float probabilities[NUM_CLASSES];
			int prediction = softmax(z, NUM_CLASSES, probabilities);
		    
		    if (label == prediction) {
		        True++;
		    } else {
		        False++;
		    }
		    
		    mse_test -= log(probabilities[label]);
		    //printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		    test_size++;
		}

		mse_test /=  test_size;
		test_losses[epoch] = mse_test;
		test_accuracies[epoch] =  True/(False+True)*100;
		if(epoch%5==0){
			printf("3- Epoch %d: MSE = %f, Accuricy: %f, Time:%f, test_mse: %f, test_Accuricy: %f\n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch],mse_test,test_accuracies[epoch]);
		}
		// end of test
	}
	fclose(w_record);
	closedir(dp); 
	
	FILE *file = fopen("results/results_adam.txt", "w");
	for (epoch = 0; epoch < EPOCHS; epoch++) {
    	fprintf(file, "%d %f %f %f %f %f\n", epoch,train_losses[epoch],train_accuracies[epoch], train_times[epoch],test_losses[epoch], test_accuracies[epoch]);
	}
	fclose(file);
	
}


int main() {
	srand(time(NULL));
    const char *output1_directory = "train_set";
    const char *output2_directory = "test_set";
    
    initialize_weights();
    
    //gradient_descent(output1_directory, output2_directory);
    
    //sgd(output1_directory, output2_directory);
    
    adam(output1_directory, output2_directory);
    return 0;
}

