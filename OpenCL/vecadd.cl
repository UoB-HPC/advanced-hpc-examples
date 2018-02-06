#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void vecadd(global const float * a,
                   global const float * b,
                   global float * c)
{
  const size_t i = get_global_id(0);
  c[i] = a[i] + b[i];
}
