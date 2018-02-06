#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void die(const char *message, const int line, const char *file);

void initialise(float **a, float **b, float **c, const int N);
void finalise(float **a, float **b, float **c);

int main(int argc, char const *argv[]) {
  int N = 1024; /* vector size */
  int num_iterations = 100000;
  float *a = NULL;
  float *b = NULL;
  float *c = NULL;

  initialise(&a, &b, &c, N);

  // Set values of a and b on the host
  for (int i = 0; i < N; i++) {
    a[i] = 1.f;
    b[i] = 2.f;
  }

// Set up the mapping of arrays between the host and the device
#pragma omp target data map(to : a [0:N], b [0:N]) map(from : c [0:N])
  for (int itr = 0; itr < num_iterations; itr++) {
// Execute vecadd on the target device
#pragma omp target
#pragma omp teams distribute parallel for simd
    for (int i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
    }
  }

  // Verify the results
  int correct_results = 1;
  for (int i = 0; i < N; i++) {
    if (fabs(c[i] - 3.f) > 0.00001f) {
      printf("Incorrect answer at index %d\n", i);
      correct_results = 0;
    }
  }

  if (correct_results) {
    printf("Success!\n");
  }

  finalise(&a, &b, &c);
  return 0;
}

void initialise(float **a, float **b, float **c, const int N) {
  // Initialise the arrays on the host
  *a = malloc(sizeof(float) * N);
  if (*a == NULL)
    die("cannot allocate memory for a", __LINE__, __FILE__);
  *b = malloc(sizeof(float) * N);
  if (*b == NULL)
    die("cannot allocate memory for b", __LINE__, __FILE__);
  *c = malloc(sizeof(float) * N);
  if (*c == NULL)
    die("cannot allocate memory for c", __LINE__, __FILE__);
}

void finalise(float **a, float **b, float **c) {
  free(*a);
  *a = NULL;

  free(*b);
  *b = NULL;

  free(*c);
  *c = NULL;
}

void die(const char *message, const int line, const char *file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}
