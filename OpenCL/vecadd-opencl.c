#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define OCL_KERNELS_FILE  "vecadd.cl"

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  vecadd;

  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;

  int wgsize;
} t_ocl;

void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
cl_device_id selectOpenCLDevice();

void initialise(t_ocl *ocl, float ** h_a, float ** h_b, float ** h_c, const int N);
void finalise(t_ocl ocl, float ** h_a, float ** h_b, float ** h_c);

int main(int argc, char const *argv[])
{
  int N = 1024;               /* vector size */
  int num_iterations = 100000;
  t_ocl ocl;
  float *h_a = NULL;
  float *h_b = NULL;
  float *h_c = NULL;
  cl_int err;

  initialise(&ocl, &h_a, &h_b, &h_c, N);

  // Set values of a and b on the host
  for(int i = 0; i < N; i++)
  {
    h_a[i] = 1.f;
    h_b[i] = 2.f;
  }

  // Write h_a to OpenCL buffer on the device
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.d_a, CL_TRUE, 0,
    sizeof(float) * N, h_a, 0, NULL, NULL);
  checkError(err, "writing h_a data", __LINE__);

  // Write h_b to OpenCL buffer on the device
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.d_b, CL_TRUE, 0,
    sizeof(float) * N, h_b, 0, NULL, NULL);
  checkError(err, "writing h_b data", __LINE__);

  for(int itr = 0; itr < num_iterations; itr++)
  {
    // Set kernel arguments
    err = clSetKernelArg(ocl.vecadd, 0, sizeof(cl_mem), &ocl.d_a);
    checkError(err, "setting vecadd arg 0", __LINE__);
    err = clSetKernelArg(ocl.vecadd, 1, sizeof(cl_mem), &ocl.d_b);
    checkError(err, "setting vecadd arg 1", __LINE__);
    err = clSetKernelArg(ocl.vecadd, 2, sizeof(cl_mem), &ocl.d_c);
    checkError(err, "setting vecadd arg 2", __LINE__);

    // Enqueue kernel
    size_t global[1] = {N};
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.vecadd,
                                 1, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "enqueueing vecadd kernel", __LINE__);

    // Wait for kernel to finish
    err = clFinish(ocl.queue);
    checkError(err, "waiting for vecadd kernel", __LINE__);
  }
  // Read d_c from OpenCL buffer on the device
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.d_c, CL_TRUE, 0,
    sizeof(float) * N, h_c, 0, NULL, NULL);
  checkError(err, "reading h_c data", __LINE__);

  // Verify the results
  int correct_results = 1;
  for(int i = 0; i < N; i++)
  {
    if(fabs(h_c[i] - 3.f) > 0.00001f)
    {
      printf("Incorrect answer at index %d\n", i);
      correct_results = 0;
    }
  }

  if(correct_results)
  {
    printf("Success!\n");
  }

  finalise(ocl, &h_a, &h_b, &h_c);
  return 0;
}

void initialise(t_ocl *ocl, float ** h_a, float ** h_b, float ** h_c, const int N)
{

  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */
  cl_int err;

  // Initialise the arrays on the host
  *h_a = malloc(sizeof(float) * N);
  if (*h_a == NULL) die("cannot allocate memory for h_a", __LINE__, __FILE__);
  *h_b = malloc(sizeof(float) * N);
  if (*h_b == NULL) die("cannot allocate memory for h_b", __LINE__, __FILE__);
  *h_c = malloc(sizeof(float) * N);
  if (*h_c == NULL) die("cannot allocate memory for h_c", __LINE__, __FILE__);

  // Initialise OpenCL
  // Get an OpenCL device
  ocl->device = selectOpenCLDevice();

  ocl->wgsize = 64;

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCL_KERNELS_FILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCL_KERNELS_FILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->vecadd = clCreateKernel(ocl->program, "vecadd", &err);
  checkError(err, "creating vecadd kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->d_a = clCreateBuffer(
    ocl->context, CL_MEM_READ_ONLY,
    sizeof(cl_float) * N, NULL, &err);
  checkError(err, "creating buffer a", __LINE__);
  ocl->d_b = clCreateBuffer(
    ocl->context, CL_MEM_READ_ONLY,
    sizeof(cl_float) * N, NULL, &err);
  checkError(err, "creating buffer b", __LINE__);
  ocl->d_c = clCreateBuffer(
    ocl->context, CL_MEM_WRITE_ONLY,
    sizeof(cl_float) * N, NULL, &err);
  checkError(err, "creating buffer c", __LINE__);
}

void finalise(t_ocl ocl, float ** h_a, float ** h_b, float ** h_c)
{
  free(*h_a);
  *h_a = NULL;

  free(*h_b);
  *h_b = NULL;

  free(*h_c);
  *h_c = NULL;

  clReleaseMemObject(ocl.d_a);
  clReleaseMemObject(ocl.d_b);
  clReleaseMemObject(ocl.d_c);
  clReleaseKernel(ocl.vecadd);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}


void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
