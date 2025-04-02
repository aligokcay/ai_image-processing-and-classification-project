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

void adam(const char *dir_path, const char *dir2_path) {
    struct dirent *entry;
    DIR *dp = opendir(dir_path);
    int epoch,i,current_index;
    clock_t start, end;
    float w[N*N+1],m[N*N+1] = {0.0f}, v[N * N + 1] = {0.0f}, epoch_time,total_time=0;
    float beta1 = 0.8f, beta2 = 0.8f, epsilon = 1e-8f, learning_rate = 0.01f;
    float train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    
    FILE *w_record = fopen("results/w_adam_0.txt", "w");
    if (w_record == NULL) {
        printf("Dosya açýlamadý.\n");
        return;
    }
    
    for (i = 0; i < N * N + 1; i++) {
    	//w[i]=0;
    	w[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    	fprintf(w_record, "%f ", w[i]);
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
		float gradients[N * N + 1] = {0.0f};
		
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
		
		    int label;
			if(filename[0] == '0'){
				label=-1;
			}else{
				label=1;
			}
		    
		    
		    float dot_product = 0.0f;
		    for(i=0; i<N*N;i++){
		    	dot_product += w[i] * x_train[i];
			}
			dot_product += w[N*N];
			float z = tanh(dot_product);		
			float error = z - label;
			
			if(z*label>0){
				True++;
			}else{
				False++;
			}
			
			mse += error * error;
			
			for (i = 0; i < N * N; i++) {
		        gradients[i] += (2.0f / N) * error * (1 - z * z) * x_train[i];
		    }
		    gradients[N * N] += (2.0f / N) * error * (1 - z * z); 
		    
		    current_index++;
		}
		if (current_index > 0) {
			//mse = sqrt(mse);
		    mse /=  current_index;
		}
		
		for (i = 0; i < N * N + 1; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] * gradients[i]);

            float m_hat = m[i] / (1 - powf(beta1, epoch + 1));
            float v_hat = v[i] / (1 - powf(beta2, epoch + 1));

            w[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
            fprintf(w_record, "%f ", w[i]);
        }
        fprintf(w_record, "\n");
        
        train_losses[epoch] = mse;
		train_accuracies[epoch] = (True / (True + False)) * 100;
		
		end = clock();
		epoch_time = ((float) (end - start)) / CLOCKS_PER_SEC;
		total_time += epoch_time;
		train_times[epoch] = total_time;
        
		if(epoch%10==0){
			printf("Epoch %d: MSE = %f, Accuricy: %f, Time:%f \n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch]);
		}

        rewinddir(dp);
	}
	fclose(w_record);
	closedir(dp); 
	
	FILE *file = fopen("results/results_adam.txt", "w");
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
		    int prediction;
		
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
		
		    int label;
			if(filename[0] == '0'){
				label=-1;
			}else{
				label=1;
			}
		    
		    float dot_product = 0.0f;
		    for(i=0; i<N*N;i++){
		    	dot_product += w[i] * x_test[i];
			}
			dot_product += w[N*N];
			float z = tanh(dot_product);
			
			printf("\n %f ",z);
			
			if(z>0){
				prediction = 1;
			}else{
				prediction = -1;
			}
			
			if(label==prediction){
				True++;
			}else{
				False++;
			}
		    
		    printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		}
}

int main() {
	srand(time(NULL));
    const char *output1_directory = "train_set";
    const char *output2_directory = "test_set";
    
    adam(output1_directory, output2_directory);
    return 0;
}

