#include "knn.h"
#include <omp.h>
#include <string.h>


typedef struct {
    float *distances;
    char *labels;
} TopKResult;

void merge_topk(TopKResult *dest, TopKResult *src, int k) {
    float *new_dist = malloc(sizeof(float) * k);
    char *new_labels = malloc(sizeof(char) * k);
    
    int i = 0, j = 0, pos = 0;
    
    while (pos < k && (i < k || j < k)) {
        float d1 = (i < k && dest->distances[i] != -1) ? dest->distances[i] : 1e9;
        float d2 = (j < k && src->distances[j] != -1) ? src->distances[j] : 1e9;
        
        if (d1 < d2 && i < k) {
            new_dist[pos] = dest->distances[i];
            new_labels[pos] = dest->labels[i];
            i++;
        } else if (j < k) {
            new_dist[pos] = src->distances[j];
            new_labels[pos] = src->labels[j];
            j++;
        }
        pos++;
    }
    
    free(dest->distances);
    free(dest->labels);
    
    dest->distances = new_dist;
    dest->labels = new_labels;
}

char knn(int n_groups, Group * groups, int k, Point to_evaluate) {
    int num_threads = omp_get_max_threads();
    TopKResult *partial_results = malloc(sizeof(TopKResult) * num_threads);
    
    for (int t = 0; t < num_threads; t++) {
        partial_results[t].distances = malloc(sizeof(float) * k);
        partial_results[t].labels = malloc(sizeof(char) * k);
        for (int i = 0; i < k; i++) {
            partial_results[t].distances[i] = -1.0f;
            partial_results[t].labels[i] = -1;
        }
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float *local_dist = partial_results[tid].distances;
        char *local_labels = partial_results[tid].labels;
        
        #pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < n_groups; i++) {
            Group g = groups[i];
            
            for (int j = 0; j < g.length; j++) {
                float d = euclidean_distance_no_sqrt(to_evaluate, g.points[j]);
                
                for (int x = 0; x < k; x++) {
                    if (d < local_dist[x] || local_dist[x] == -1.0f) {
                        for (int y = k - 1; y > x; y--) {
                            local_dist[y] = local_dist[y - 1];
                            local_labels[y] = local_labels[y - 1];
                        }
                        local_dist[x] = d;
                        local_labels[x] = g.label;
                        break;
                    }
                }
            }
        }
    }
    
    TopKResult global;
    global.distances = malloc(sizeof(float) * k);
    global.labels = malloc(sizeof(char) * k);
    for (int i = 0; i < k; i++) {
        global.distances[i] = -1.0f;
        global.labels[i] = -1;
    }
    
    for (int t = 0; t < num_threads; t++) {
        merge_topk(&global, &partial_results[t], k);
        free(partial_results[t].distances);
        free(partial_results[t].labels);
    }
    free(partial_results);
    
    qsort(global.labels, k, sizeof(char), compare_for_sort);
    
    char most_frequent = global.labels[0];
    int most_frequent_count = 1;
    int current_frequency = 1;
    
    for (int i = 1; i < k; i++) {
        if (global.labels[i] != global.labels[i - 1]) {
            if (current_frequency > most_frequent_count) {
                most_frequent = global.labels[i - 1];
                most_frequent_count = current_frequency;
            }
            current_frequency = 1;
        } else {
            current_frequency++;
        }
        
        if (i == k - 1 && current_frequency > most_frequent_count) {
            most_frequent = global.labels[i - 1];
            most_frequent_count = current_frequency;
        }
    }
    
    free(global.distances);
    free(global.labels);
    
    return most_frequent;
}

char knn_sequencial(int n_groups, Group * groups, int k, Point to_evaluate) {
    char * labels = (char *) malloc(sizeof(char) * k);
    float * distances = (float *) malloc(sizeof(float) * k);
    
    int i, j, x, y;
    
    for (i = 0; i < k; i++) {
        labels[i] = -1;
        distances[i] = -1.0f;
    }
    
    for (i = 0; i < n_groups; i++) {
        Group g = groups[i];
        
        for (j = 0; j < g.length; j++) {
            float d = euclidean_distance_no_sqrt(to_evaluate, g.points[j]);
            
            for (x = 0; x < k; x++) {
                if (d < distances[x] || distances[x] == -1.0f) {
                    for (y = k - 1; y > x; y--) {
                        distances[y] = distances[y - 1];
                        labels[y] = labels[y - 1];
                    }
                    distances[x] = d;
                    labels[x] = g.label;
                    break;
                }
            }
        }
    }
    
    qsort(labels, k, sizeof(char), compare_for_sort);
    
    char most_frequent = labels[0];
    int most_frequent_count = 1;
    int current_frequency = 1;
    
    for (i = 1; i < k; i++) {
        if (labels[i] != labels[i - 1]) {
            if (current_frequency > most_frequent_count) {
                most_frequent = labels[i - 1];
                most_frequent_count = current_frequency;
            }
            current_frequency = 1;
        } else {
            current_frequency++;
        }
        
        if (i == k - 1 && current_frequency > most_frequent_count) {
            most_frequent = labels[i - 1];
            most_frequent_count = current_frequency;
        }
    }
    
    free(labels);
    free(distances);
    
    return most_frequent;
}

int main() {
    int n_groups = parse_number_of_groups();
    Group * groups = (Group *) malloc(sizeof(Group) * n_groups);
    
    for (int i = 0; i < n_groups; i++) { 
        groups[i] = parse_next_group();
    }
    
    int k = parse_k();
    Point to_evaluate = parse_point();
    
    printf("=== Teste de Performance ===\n");
    printf("Grupos: %d | K: %d\n\n", n_groups, k);
    
    double start_seq = omp_get_wtime();
    char resultado_seq = knn_sequencial(n_groups, groups, k, to_evaluate);
    double end_seq = omp_get_wtime();
    double tempo_seq = end_seq - start_seq;
    
    printf("Versão Sequencial:\n");
    printf("  Resultado: %c\n", resultado_seq);
    printf("  Tempo: %.4f segundos\n\n", tempo_seq);
    
    int threads[] = {2, 4, 8};
    
    printf("Versão Paralela:\n");
    printf("%-8s %-12s %-10s %-10s\n", "Threads", "Tempo (s)", "Speedup", "Eficiência");
    printf("----------------------------------------\n");
    
    for (int t = 0; t < 3; t++) {
        int num_threads = threads[t];
        omp_set_num_threads(num_threads);
        
        double start_par = omp_get_wtime();
        char resultado_par = knn(n_groups, groups, k, to_evaluate);
        double end_par = omp_get_wtime();
        double tempo_par = end_par - start_par;
        
        double speedup = tempo_seq / tempo_par;
        double eficiencia = speedup / num_threads;
        
        printf("%-8d %-12.4f %-10.2f %-10.2f %c\n", 
               num_threads, tempo_par, speedup, eficiencia, resultado_par);
    }
    
    for (int i = 0; i < n_groups; i++) {
        free(groups[i].points);
    }
    free(groups);
    
    return 0;
}