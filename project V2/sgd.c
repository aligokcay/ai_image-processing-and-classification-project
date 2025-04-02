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

void sgd(const char *dir_path, const char *dir2_path) {
    struct dirent *entry;
    DIR *dp = opendir(dir_path);
    int epoch,i,current_index,batch=0;
    clock_t start, end;
    float epoch_time,total_time=0;
    float w[N*N+1], learning_rate = 0.01f, train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    
    FILE *w_record = fopen("results/w_sgd.txt", "w");
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
    	if (epoch % 30 == 0 && epoch > 0) {
		    learning_rate *= 0.80;
		    printf("%f \n",learning_rate);
		}
		float mse = 0.0f, True=0, False=0;
		current_index = 0;
		
		float gradients[N * N + 1] = {0.0f};
		
		for(batch=0;batch<32;batch++){
			// Dosya yolunu oluþtur
			int label;
		    char file_path[512];
		    if(batch%2==0){
		    	snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "0_",rand() % 500+1,".txt");
		    	label = -1;
			}else{
				snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "1_",rand() % 500+501,".txt");
				label = 1;
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
		    
		    fclose(input_file);
		}
		
		if (current_index > 0) {
			//mse = sqrt(mse);
		    mse /=  current_index;
		}
		    
		for (i = 0; i < N * N + 1; i++) {
		    w[i] -= learning_rate * gradients[i];
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
	}
	
	fclose(w_record);

	FILE *file = fopen("results/results_sgd.txt", "w");
	for (epoch = 0; epoch < EPOCHS; epoch++) {
    	fprintf(file, "%d %f %f %f\n", epoch,train_losses[epoch],train_accuracies[epoch], train_times[epoch]);
	}
	fclose(file);
	
	//test
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
	//srand(time(NULL));
    const char *output1_directory = "train_set";
    const char *output2_directory = "test_set";
    
    sgd(output1_directory, output2_directory);
    return 0;
}

