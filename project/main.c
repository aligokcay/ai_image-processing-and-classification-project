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

float w_first[N * N + 1];

void initialize_weights() {
    int i;
    for (i = 0; i < N * N + 1; i++) {
        //w_first[i] = 0;
		w_first[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void gradient_descent(const char *dir_path, const char *dir2_path) {
    struct dirent *entry;
    DIR *dp = opendir(dir_path);
    int epoch,i,current_index;
    float train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    float test_losses[EPOCHS], test_accuracies[EPOCHS];
    clock_t start, end;
	float epoch_time,total_time=0;
	
	FILE *w_record = fopen("results/w_gd.txt", "w");
    if (w_record == NULL) {
        printf("Dosya a��lamad�.\n");
        return;
    }

    float w[N*N+1], learning_rate = 0.01f;
	for (i = 0; i < N * N + 1; i++) {
    	//w[i]=0;
    	w[i] = w_first[i];
    	fprintf(w_record, "%f ", w[i]);
	}
	fprintf(w_record, "\n");
    

    if (dp == NULL) {
        perror("Dizin a��lamad�");
        return;
    }
    
    for (epoch = 0; epoch < EPOCHS; epoch++) {
    	start = clock();
		float mse = 0.0f,True=0,False=0;
		current_index = 0;
		
		float gradients[N * N + 1] = {0.0f};
		while ((entry = readdir(dp)) != NULL) {
		    const char *filename = entry->d_name;
		
		    if (filename[0] == '.') {
		        continue;
		    }
		
		    // Dosya yolunu olu�tur
		    char file_path[512];
		    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, filename);
		
		    // Dosyay� a�
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("Dosya a��lamad�: %s\n", file_path);
		        continue;
		    }
		
		    float x_train[N*N] = {0.00f};
		    
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_train[i]) != 1) {
			        printf("Dosyadan de�er okunamad�: %s\n", file_path);
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
	
		//mse = sqrt(mse);
		mse /=  current_index;
		
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

        rewinddir(dp);
        
        
        
        // test
		DIR *dp2 = opendir(dir2_path);
		if (dp2 == NULL) {
	        perror("Dizin a��lamad�");
	        return;
	    }
	    True=0,False=0;
		float mse_test=0, test_size=0;
	    while ((entry = readdir(dp2)) != NULL) {
		    const char *filename = entry->d_name;
		    int prediction;
		
		    if (filename[0] == '.') {
		        continue;
		    }
		
		    // Dosya yolunu olu�tur
		    char file_path[512];
		    snprintf(file_path, sizeof(file_path), "%s/%s", dir2_path, filename);
		
		    // Dosyay� a�
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("Dosya a��lamad�: %s\n", file_path);
		        continue;
		    }
		    float x_test[N*N] = {0.00f};
		    
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_test[i]) != 1) {
			        printf("Dosyadan de�er okunamad�: %s\n", file_path);
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
			
			float error = z - label;
			mse_test += error * error;
			
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
		    
		    //printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		    test_size++;
		}
		if (current_index > 0) {
			//mse_test = sqrt(mse);
		    mse_test /=  test_size;
		}
		test_losses[epoch] = mse_test;
		test_accuracies[epoch] =  True/(False+True)*100;
		if(epoch%10==0){
			printf("1- Epoch %d: MSE = %f, Accuricy: %f, Time:%f, test_mse: %f, test_Accuricy: %f\n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch],mse_test,test_accuracies[epoch]);
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
    int epoch,i,current_index,batch=0;
    clock_t start, end;
    float epoch_time,total_time=0;
    float w[N*N+1], learning_rate = 0.01f, train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    float test_losses[EPOCHS], test_accuracies[EPOCHS];
    
    FILE *w_record = fopen("results/w_sgd.txt", "w");
    if (w_record == NULL) {
        printf("Dosya a��lamad�.\n");
        return;
    }
    
    for (i = 0; i < N * N + 1; i++) {
    	//w[i]=0;
    	w[i] =  w_first[i];
    	fprintf(w_record, "%f ", w[i]);
	}
	fprintf(w_record, "\n");
    

    if (dp == NULL) {
        perror("Dizin a��lamad�");
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
			// Dosya yolunu olu�tur
			int label;
		    char file_path[512];
		    if(batch%2==0){
		    	snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "0_",rand() % 500+1,".txt");
		    	label = -1;
			}else{
				snprintf(file_path, sizeof(file_path), "%s/%s%d%s", dir_path, "1_",rand() % 500+501,".txt");
				label = 1;
			}
		
		    // Dosyay� a�
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("\n\nDosya a��lamad�: %s\n", file_path);
		        continue;
		    }
		    float x_train[N*N] = {0.00f};
		    
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_train[i]) != 1) {
			        printf("Dosyadan de�er okunamad�: %s\n", file_path);
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
		
		// test
		DIR *dp2 = opendir(dir2_path);
		if (dp2 == NULL) {
	        perror("Dizin a��lamad�");
	        return;
	    }
	    True=0,False=0;
		float mse_test=0, test_size=0;
	    while ((entry = readdir(dp2)) != NULL) {
		    const char *filename = entry->d_name;
		    int prediction;
		
		    if (filename[0] == '.') {
		        continue;
		    }
		
		    // Dosya yolunu olu�tur
		    char file_path[512];
		    snprintf(file_path, sizeof(file_path), "%s/%s", dir2_path, filename);
		
		    // Dosyay� a�
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("Dosya a��lamad�: %s\n", file_path);
		        continue;
		    }
		
		    float x_test[N*N] = {0.00f};
		    
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_test[i]) != 1) {
			        printf("Dosyadan de�er okunamad�: %s\n", file_path);
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
			
			float error = z - label;
			mse_test += error * error;
			
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
		    
		    //printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		    test_size++;
		}
		if (current_index > 0) {
			//mse_test = sqrt(mse);
		    mse_test /=  test_size;
		}
		test_losses[epoch] = mse_test;
		test_accuracies[epoch] =  True/(False+True)*100;
		if(epoch%5==0){
			printf("2- Epoch %d: MSE = %f, Accuricy: %f, Time:%f, test_mse: %f, test_Accuricy: %f\n", epoch + 1, mse,train_accuracies[epoch], train_times[epoch],mse_test,test_accuracies[epoch]);
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
    int epoch,i,current_index;
    clock_t start, end;
    float w[N*N+1],m[N*N+1] = {0.0f}, v[N * N + 1] = {0.0f}, epoch_time,total_time=0;;
    float beta1 = 0.8f, beta2 = 0.8f, epsilon = 1e-8f, learning_rate = 0.01f;
    float train_losses[EPOCHS], train_accuracies[EPOCHS],train_times[EPOCHS];
    float test_losses[EPOCHS], test_accuracies[EPOCHS];
    
    FILE *w_record = fopen("results/w_adam.txt", "w");
    if (w_record == NULL) {
        printf("Dosya a��lamad�.\n");
        return;
    }
    
    for (i = 0; i < N * N + 1; i++) {
    	//w[i]=0;
    	w[i] =  w_first[i];
    	fprintf(w_record, "%f ", w[i]);
	}
	fprintf(w_record, "\n");
    
    if (dp == NULL) {
        perror("Dizin a��lamad�");
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
		        printf("Dosya a��lamad�: %s\n", file_path);
		        continue;
		    }
		
		    float x_train[N*N] = {0.00f};
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_train[i]) != 1) {
			        printf("Dosyadan de�er okunamad�: %s\n", file_path);
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
        rewinddir(dp);
        
        // test
		DIR *dp2 = opendir(dir2_path);
		if (dp2 == NULL) {
	        perror("Dizin a��lamad�");
	        return;
	    }
	    True=0,False=0;
		float mse_test=0, test_size=0;
	    while ((entry = readdir(dp2)) != NULL) {
		    const char *filename = entry->d_name;
		    int prediction;
		
		    if (filename[0] == '.') {
		        continue;
		    }
		
		    // Dosya yolunu olu�tur
		    char file_path[512];
		    snprintf(file_path, sizeof(file_path), "%s/%s", dir2_path, filename);
		
		    // Dosyay� a�
		    FILE *input_file = fopen(file_path, "r");
		    if (!input_file) {
		        printf("Dosya a��lamad�: %s\n", file_path);
		        continue;
		    }
		
		    float x_test[N*N] = {0.00f};
		    
		    for (i = 0; i < N * N; i++) {
			    if (fscanf(input_file, "%f", &x_test[i]) != 1) {
			        printf("Dosyadan de�er okunamad�: %s\n", file_path);
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
			
			float error = z - label;
			mse_test += error * error;
			
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
		    
		    //printf("dogruluk: %f (%.f/%.f)", True/(False+True)*100,True,True+False);
		    
		    current_index++;
		    test_size++;
		}
		if (current_index > 0) {
			//mse_test = sqrt(mse);
		    mse_test /=  test_size;
		}
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
    //sgd(output1_directory, output2_directory);
    
    //gradient_descent(output1_directory, output2_directory);
    
    adam(output1_directory, output2_directory);
    return 0;
}

