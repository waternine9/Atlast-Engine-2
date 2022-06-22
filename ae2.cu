#include "ae2.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include "SDL.h"

#define tid threadIdx.x
#define bid blockIdx.x

static float _aspect = 1.0f;
static float _fov = 1.0f;

static curandState* randAr_d = nullptr;

static aeBufferUC _frameBuffer;
static aeBuffer _depthBuffer;
static aeBuffer _protoBuffer;

static aeBuffer _vertexBuffer;
static aeBufferI _faceBuffer;
static aeBuffer _attrBuffer;
static aeBuffer _bvhBuffer;

static int2 _screenRes;

static SDL_Window* _window = NULL;
static SDL_Surface* _windowSurface = NULL;
static SDL_Renderer* _windowRenderer = NULL;

static float3 _camPos;
static float3 _camRot;

struct octreeNode
{
	struct octreeNode* children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
	struct octreeNode* b = nullptr;
	float3 min;
	float3 max;
	int *faceData = nullptr;
	int stride = 0;
};

static octreeNode *_octree = nullptr;

// Math Utils
__host__ __device__ float _dot_d(float3 x, float3 y)
{
	return x.x * y.x + x.y * y.y + x.z * y.z;
}

__host__ __device__ float3 _cross_d(float3 x, float3 y)
{
	return {
		x.y * y.z - x.z * y.y,
		-(x.x * y.z - x.z * y.x),
		x.x * y.y - x.y * y.x
	};
}

__host__ __device__ float3 _sub_d(float3 x, float3 y)
{
	return { x.x - y.x, x.y - y.y, x.z - y.z };
}

__host__ __device__ float3 _add_d(float3 x, float3 y)
{
	return { x.x + y.x, x.y + y.y, x.z + y.z };
}

__host__ __device__ float3 _mul_d(float3 x, float3 y)
{
	return { x.x * y.x, x.y * y.y, x.z * y.z };
}
__host__ __device__ float3 _div_d(float3 x, float3 y)
{
	return { x.x / y.x, x.y / y.y, x.z / y.z };
}
__device__ float3 _norm_d(float3 x)
{
	float magn = 1.0f / sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
	return { x.x * magn, x.y * magn, x.z * magn };
}

// GENERAL FUNCTIONS

void aeGenBuffer(aeBuffer* buffer, size_t size)
{
	cudaMalloc(&buffer->data, sizeof(float) * size);
	buffer->size = size;
}
void aeGenBuffer(aeBufferI* buffer, size_t size)
{
	cudaMalloc(&buffer->data, sizeof(int) * size);
	buffer->size = size;
}
void aeGenBuffer(aeBufferUC* buffer, size_t size)
{
	cudaMalloc(&buffer->data, sizeof(unsigned char) * size);
	buffer->size = size;
}
void aeBufferData(aeBuffer* buffer, float* data, size_t size)
{
	cudaMemcpy(buffer->data, data, sizeof(float) * size, cudaMemcpyHostToDevice);
}
void aeBufferData(aeBufferI* buffer, int* data, size_t size)
{
	cudaMemcpy(buffer->data, data, sizeof(int) * size, cudaMemcpyHostToDevice);
}
void aeBufferData(aeBufferUC* buffer, unsigned char* data, size_t size)
{
	cudaMemcpy(buffer->data, data, sizeof(unsigned char) * size, cudaMemcpyHostToDevice);
}
void aeAddVBO(aeBuffer* buffer)
{
	cudaMalloc(&_vertexBuffer.data, sizeof(float) * buffer->size);
	cudaMemcpy(_vertexBuffer.data, buffer->data, sizeof(float) * buffer->size, cudaMemcpyDeviceToDevice);
	_vertexBuffer.size = buffer->size;
}
void aeAddFBO(aeBufferI* buffer)
{
	cudaMalloc(&_faceBuffer.data, sizeof(int) * buffer->size);
	cudaMemcpy(_faceBuffer.data, buffer->data, sizeof(int) * buffer->size, cudaMemcpyDeviceToDevice);
	_faceBuffer.size = buffer->size;
}
void aeAddABO(aeBuffer* buffer)
{
	cudaMalloc(&_attrBuffer.data, sizeof(float) * buffer->size);
	cudaMemcpy(_attrBuffer.data, buffer->data, sizeof(float) * buffer->size, cudaMemcpyDeviceToDevice);
	_attrBuffer.size = buffer->size;
}
void aeAddBVH(aeBuffer* buffer)
{
	cudaMalloc(&_bvhBuffer.data, sizeof(float) * buffer->size);
	cudaMemcpy(_bvhBuffer.data, buffer->data, sizeof(float) * buffer->size, cudaMemcpyDeviceToDevice);
	_bvhBuffer.size = buffer->size;
}
void aeAddFrame(aeBufferUC* buffer)
{
	cudaMalloc(&_frameBuffer.data, sizeof(unsigned char) * buffer->size);
	cudaMemcpy(_frameBuffer.data, buffer->data, sizeof(unsigned char) * buffer->size, cudaMemcpyDeviceToDevice);
	_frameBuffer.size = buffer->size;
}
void aeFreeFrame()
{
	cudaFree(_frameBuffer.data);
}
void aeAddProto(aeBuffer* buffer)
{
	cudaMalloc(&_protoBuffer.data, sizeof(float) * buffer->size);
	cudaMemcpy(_protoBuffer.data, buffer->data, sizeof(float) * buffer->size, cudaMemcpyDeviceToDevice);
	_protoBuffer.size = buffer->size;
}
void aeFreeProto()
{
	cudaFree(_protoBuffer.data);
}
__global__ void _init_rand_d(curandState* state, size_t offset)
{
	curand_init(clock() + bid + offset, bid + offset, 0, &state[bid]);
}
void aeRandInit()
{
	cudaMalloc(&randAr_d, sizeof(curandState) * 128 * 128 * 64); 
	_init_rand_d<<<128 * 128 * 64, 1>>>(randAr_d, clock());
	cudaDeviceSynchronize();
}
void aeFreeBuffer(aeBuffer* buffer)
{
	cudaFree(buffer->data);
}
void aeFreeBuffer(aeBufferI* buffer)
{
	cudaFree(buffer->data);
}
void aeSetFlag(aeBuffer* buffer, char flag)
{
	buffer->flags |= flag;
}
void aeViewport(int res_x, int res_y, float fov)
{
	_screenRes = { res_x / 2, res_y / 2 };
	_aspect = (float)res_y / (float)res_x;
	_fov = fov;
}
__host__ __device__ float3 _rotate(float3 v, float3 rotate_by, float3 center)
{
	v = _sub_d(v, center);
	float input1[1][3];
	float input2[3][3];
	if (rotate_by.z != 0.0f)
	{
		float SINF = sinf(rotate_by.z);
		float COSF = cosf(rotate_by.z);
		float output[1][3] = { { 0, 0, 0 } };

		input2[0][0] = COSF;
		input2[0][1] = -SINF;
		input2[0][2] = 0;
		input2[1][0] = SINF;
		input2[1][1] = COSF;
		input2[1][2] = 0;
		input2[2][0] = 0;
		input2[2][1] = 0;
		input2[2][2] = 1;
		input1[0][0] = v.x;
		input1[0][1] = v.y;
		input1[0][2] = v.z;
		for (int _ = 0;_ < 1;_++)
			for (int Y = 0;Y < 3;Y++)
				for (int k = 0;k < 3;k++)
				{
					output[_][Y] += input1[_][k] * input2[k][Y];
				}
		v = { (float)output[0][0], (float)output[0][1], (float)output[0][2] };
	}
	if (rotate_by.y != 0.0f)
	{
		float SINF = sinf(rotate_by.y);
		float COSF = cosf(rotate_by.y);
		float output[1][3] = { { 0, 0, 0 } };
		input2[0][0] = COSF;
		input2[0][1] = 0;
		input2[0][2] = SINF;
		input2[1][0] = 0;
		input2[1][1] = 1;
		input2[1][2] = 0;
		input2[2][0] = -SINF;
		input2[2][1] = 0;
		input2[2][2] = COSF;
		input1[0][0] = v.x;
		input1[0][1] = v.y;
		input1[0][2] = v.z;
		for (int _ = 0;_ < 1;_++)
			for (int Y = 0;Y < 3;Y++)
				for (int k = 0;k < 3;k++)
				{
					output[_][Y] += input1[_][k] * input2[k][Y];
				}
		v = { (float)output[0][0], (float)output[0][1], (float)output[0][2] };
	}
	if (rotate_by.x != 0.0f)
	{
		float SINF = sinf(rotate_by.x);
		float COSF = cosf(rotate_by.x);
		float output[1][3] = { { 0, 0, 0 } };
		input2[0][0] = 1;
		input2[0][1] = 0;
		input2[0][2] = 0;
		input2[1][0] = 0;
		input2[1][1] = COSF;
		input2[1][2] = -SINF;
		input2[2][0] = 0;
		input2[2][1] = SINF;
		input2[2][2] = COSF;
		input1[0][0] = v.x;
		input1[0][1] = v.y;
		input1[0][2] = v.z;
		for (int _ = 0;_ < 1;_++)
			for (int Y = 0;Y < 3;Y++)
				for (int k = 0;k < 3;k++)
				{
					output[_][Y] += input1[_][k] * input2[k][Y];
				}
		v = { (float)output[0][0], (float)output[0][1], (float)output[0][2] };
	}
	v = _add_d(v, center);
	return v;
}

__global__ void aeRotate_d(size_t _start, size_t _end, float* _vertexBufferData, int *_faceBufferData, float3 axis, float3 center, float *bvh)
{
	float minX = 1e+6, maxX = -1e+6, minY = 1e+6, maxY = -1e+6, minZ = 1e+6, maxZ = -1e+6;
	bool alreadyChosen[4096];
	memset(alreadyChosen, 0, sizeof(bool) * 4096);
	for (int trid = _start / 3;trid < _end / 3;trid++)
	{
		for (int vid = 0;vid < 2;vid++)
		{
			int vert_i = _faceBufferData[trid * 3 + vid] * 3;
			if (!alreadyChosen[vert_i])
			{
				
				float3 v0{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				v0 = _rotate(v0, axis, center);

				minX = min(v0.x, minX);
				minY = min(v0.y, minY);
				minZ = min(v0.z, minZ);
				maxX = max(v0.x, maxX);
				maxY = max(v0.y, maxY);
				maxZ = max(v0.z, maxZ);

				_vertexBufferData[vert_i] = v0.x;
				_vertexBufferData[vert_i + 1] = v0.y;
				_vertexBufferData[vert_i + 2] = v0.z;
				alreadyChosen[vert_i] = true;
			}
		}
	}
	bvh[0] = minX;
	bvh[2] = minY;
	bvh[4] = minZ;
	bvh[1] = maxX;
	bvh[3] = maxY;
	bvh[5] = maxZ;
}
void aeRotate(size_t idx, float ax, float ay, float az, float cx, float cy, float cz)
{
	
	float *h_bvh = (float*)malloc(sizeof(float) * _bvhBuffer.size);
	cudaMemcpy(h_bvh, _bvhBuffer.data, sizeof(float) * _bvhBuffer.size, cudaMemcpyDeviceToHost);
	aeRotate_d<<<1, 1>>>((size_t)h_bvh[idx * 8 + 6], (size_t)h_bvh[idx * 8 + 7], _vertexBuffer.data, _faceBuffer.data, { ax, ay, az }, { cx, cy, cz }, _bvhBuffer.data + idx * 8);
	cudaDeviceSynchronize();
	free(h_bvh);
}

void aeCameraTransform(float rx, float ry, float rz, float cx, float cy, float cz)
{
	_camPos = { cx, cy, cz };
	_camRot = { rx, ry, rz };
}

void _VertexShader()
{
	// Will implement once Ellipse gets integrated.
	return;
}
void _GeometryShader()
{
	// Will implement once Ellipse gets integrated.
	return;
}
__device__ bool _rayTriangleIntersect(const float3 orig, const float3 dir, const float3 v0, const float3 v1, const float3 v2, float* t, float* u, float* v)
{
	float3 v0v1 = _sub_d(v1, v0);
	float3 v0v2 = _sub_d(v2, v0);
	float3 pvec = _cross_d(dir, v0v2);
	float det = _dot_d(v0v1, pvec);
	if (fabs(det) < 0.0001F) return false;
	float invDet = 1 / det;
	float3 tvec = _sub_d(orig, v0);
	*u = _dot_d(tvec, pvec) * invDet;
	if (*u < 0 || *u > 1) return false;
	float3 qvec = _cross_d(tvec, v0v1);
	*v = _dot_d(dir, qvec) * invDet;
	if (*v < 0 || *u + *v > 1) return false;
	*t = _dot_d(v0v2, qvec) * invDet;
	if (*t < 0.0f) return false;
	return true;
}
__device__ bool _rayBboxIntersect(float3 ro, float3 rd, float3 bboxMin, float3 bboxMax, float &near) 
{
	float tmin = (bboxMin.x - ro.x) / rd.x;
	float tmax = (bboxMax.x - ro.x) / rd.x;

	if (tmin > tmax)
	{
		float temp = tmin;
		tmin = tmax;
		tmax = temp;
	}

	float tymin = (bboxMin.y - ro.y) / rd.y;
	float tymax = (bboxMax.y - ro.y) / rd.y;

	if (tymin > tymax)
	{
		float temp = tymin;
		tymin = tymax;
		tymax = temp;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (bboxMin.z - ro.z) / rd.z;
	float tzmax = (bboxMax.z - ro.z) / rd.z;

	if (tmin > tmax)
	{
		float temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	near = tmin;

	if (near < 0) {
		near = tmax;
		if (near < 0) return false;
	}

	return true;
};
__global__ void _PathTrace_d(int* _faceBufferData, int _triangleCount, float* _vertexBufferData, float* _bvhBufferData, int _bvhCount, float* _attrBufferData, float* _protoBufferData, curandState* _randAr, int2 _screenRes, int2 _topLeftCoord, float3 _camPos_d, float3 _camRot_d)
{
	float2 localUV = { ((float)blockIdx.y + _topLeftCoord.x) / _screenRes.x, ((float)bid + _topLeftCoord.y) / _screenRes.y };
	float3 ro = _camPos_d;
	float3 rd = _rotate(_norm_d({ localUV.x - 0.5f, localUV.y - 0.5f, 1.0f }), _camRot_d, { 0.0f, 0.0f, 0.0f });
	float3 rc{ 0.04f, 0.04f, 0.04f };
	float eta = 1.45f;
	__shared__ float3 finalCol[64];
	for (int rayStep = 0;rayStep < 5;rayStep++)
	{
		int minTrid = -1;
		float minT = 1e+10, minU = 0.0f, minV = 0.0f;
		for (int bvhid = 0;bvhid < _bvhCount;bvhid++)
		{
			float curDist;
			if (!_rayBboxIntersect(ro, rd, { _bvhBufferData[bvhid * 8], _bvhBufferData[bvhid * 8 + 2], _bvhBufferData[bvhid * 8 + 4] }, { _bvhBufferData[bvhid * 8 + 1], _bvhBufferData[bvhid * 8 + 3], _bvhBufferData[bvhid * 8 + 5] }, curDist)) continue;
			if (curDist > minT) continue;
			for (int trid = _bvhBufferData[bvhid * 8 + 6] / 3;trid < _bvhBufferData[bvhid * 8 + 7] / 3;trid++)
			{
				int vert_i = _faceBufferData[trid * 3] * 3;
				float3 v0{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				vert_i = _faceBufferData[trid * 3 + 1] * 3;
				float3 v1{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				vert_i = _faceBufferData[trid * 3 + 2] * 3;
				float3 v2{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				float t, u, v;
				if (_rayTriangleIntersect(ro, rd, v0, v1, v2, &t, &u, &v))
				{
					if (t < minT)
					{
						minTrid = trid;
						minT = t;
						minU = u;
						minV = v;
					}
				}
			}
		}
		if (minTrid > -1)
		{
			ro = _add_d(ro, _mul_d(rd, { minT, minT, minT }));

			int vert_i = _faceBufferData[minTrid * 3] * 3;
			float3 v0{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
			vert_i = _faceBufferData[minTrid * 3 + 1] * 3;
			float3 v1{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
			vert_i = _faceBufferData[minTrid * 3 + 2] * 3;
			float3 v2{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
			float3 norm = _mul_d(_norm_d(_cross_d(_sub_d(v2, v0), _sub_d(v1, v0))), { 1.0f, 1.0f, 1.0f });
			
			float dNI = _dot_d(rd, norm);
			if (dNI > 0.0f)
			{
				norm = _mul_d(norm, { -1.0f, -1.0f, -1.0f });
				dNI = _dot_d(rd, norm);
			}

			int triangleAttr_i = minTrid * 5;
			float3 color = {
				_attrBufferData[triangleAttr_i],
				_attrBufferData[triangleAttr_i + 1],
				_attrBufferData[triangleAttr_i + 2],
			};

			float Kr = _attrBufferData[triangleAttr_i + 3];
			bool isRefract = _attrBufferData[triangleAttr_i + 4] > 0.0f;

			if (!isRefract)
			{
				rd = _sub_d(rd, _mul_d({ 2.0f, 2.0f, 2.0f }, _mul_d({ dNI, dNI, dNI }, norm)));
			}
			else
			{
				eta = 1.0f / eta;
				
				float k = 1.0f - eta * eta * (1.0f - dNI * dNI);
				if (k < 0.0f)
				{
					rd = _sub_d(rd, _mul_d({ 2.0f, 2.0f, 2.0f }, _mul_d({ dNI, dNI, dNI }, norm)));
				}
				else
				{
					float n = eta * dNI + sqrtf(k);
					rd = _sub_d(_mul_d({ eta, eta, eta }, rd), _mul_d({ n, n, n }, norm));
				}
			}
			
			rd.x += curand_uniform(&_randAr[(bid + blockIdx.y * 128) * 32 + tid]) * Kr - 0.5f * Kr;
			rd.y += curand_uniform(&_randAr[(bid + blockIdx.y * 128) * 32 + tid]) * Kr - 0.5f * Kr;
			rd.z += curand_uniform(&_randAr[(bid + blockIdx.y * 128) * 32 + tid]) * Kr - 0.5f * Kr;
			rd = _norm_d(rd);

			ro = _add_d(ro, _mul_d(rd, { 0.001f, 0.001f, 0.001f }));

			rc.x *= color.x;
			rc.y *= color.y;
			rc.z *= color.z;
			if (color.x + color.y + color.z > 10.0f) break;
		}
		else
		{
			rc = { 0.0f, 0.0f, 0.0f };
			break;
		}
	}
	finalCol[tid].x = min(rc.x, 1.0f);
	finalCol[tid].y = min(rc.y, 1.0f);
	finalCol[tid].z = min(rc.z, 1.0f);
	__syncthreads();
	if (tid == 0)
	{
		int basePid = ((blockIdx.y + _topLeftCoord.x) + (bid + _topLeftCoord.y) * (_screenRes.x)) * 3;
		for (int i = 0;i < 64;i++)
		{
			
			_protoBufferData[basePid] += finalCol[i].x / 64.0f;
			_protoBufferData[basePid + 1] += finalCol[i].y / 64.0f;
			_protoBufferData[basePid + 2] += finalCol[i].z / 64.0f;
		}
		_protoBufferData[basePid] /= 2.0f;
		_protoBufferData[basePid + 1] /= 2.0f;
		_protoBufferData[basePid + 2] /= 2.0f;
	}
}

__global__ void _OctreePathTrace_d(octreeNode* octree, int maxDepth, float* _protoBufferData, float *_vertexBufferData, float *_attrBufferData, curandState* _randAr, int2 _screenRes, int2 _topLeftCoord, float3 _camPos_d, float3 _camRot_d)
{
	float2 localUV = { ((float)tid + _topLeftCoord.x) / _screenRes.x, ((float)bid + _topLeftCoord.y) / _screenRes.y };
	float3 ro = _camPos_d;
	float3 rd = _rotate(_norm_d({ localUV.x * 2.0f - 1.0f, localUV.y * 2.0f - 1.0f, 1.0f }), _camRot_d, { 0.0f, 0.0f, 0.0f });

	ro = _add_d(ro, _mul_d(rd, { 0.01, 0.01, 0.01 }));

	float3 rc{ 0.04f, 0.04f, 0.04f };
	float eta = 1.45f;
	
	int depth = 0;
	for (int j = 0;j < 200;j++)
	{
		
		octreeNode* curNode = octree;
		
		depth = 0;
		for (int i = 0;i < maxDepth;i++)
		{
			depth = i;
			if (curNode->faceData != NULL) break;
			float3 center = _div_d(_add_d(curNode->min, curNode->max), { 2.0f, 2.0f, 2.0f });
			int bx = ro.x < center.x;
			int by = ro.y < center.y;
			int bz = ro.z < center.z;
			bool moved = false;
			int idx = bx + by * 2 + bz * 4;
			if (curNode->children[idx] != NULL)
			{
				curNode = curNode->children[idx];
			}
			else
			{
				break;
			}
		}
		
		if (curNode->faceData != nullptr)
		{
			
			int minTrid = -1;
			float minT = 1e+6, minU, minV;
			for (int trid = 0;trid < curNode->stride / 3;trid++)
			{
				
				int vert_i = curNode->faceData[trid * 3] * 3;
				float3 v0{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				vert_i = curNode->faceData[trid * 3 + 1] * 3;
				float3 v1{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				vert_i = curNode->faceData[trid * 3 + 2] * 3;
				float3 v2{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				float t, u, v;
				if (_rayTriangleIntersect(_camPos_d, rd, v0, v1, v2, &t, &u, &v))
				{
					if (t < minT)
					{
						minTrid = trid;
						minT = t;
						minU = u;
						minV = v;
					}
				}
			}
			if (minTrid > -1)
			{

				ro = _camPos_d;
				
				ro = _add_d(ro, _mul_d(rd, { minT, minT, minT }));

				

				int vert_i = curNode->faceData[minTrid * 3] * 3;
				float3 v0{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				vert_i = curNode->faceData[minTrid * 3 + 1] * 3;
				float3 v1{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				vert_i = curNode->faceData[minTrid * 3 + 2] * 3;
				float3 v2{ _vertexBufferData[vert_i], _vertexBufferData[vert_i + 1], _vertexBufferData[vert_i + 2] };
				float3 norm = _norm_d(_cross_d(_sub_d(v2, v0), _sub_d(v1, v0)));

				float dNI = _dot_d(rd, norm);
				if (dNI > 0.0f)
				{
					norm = _mul_d(norm, { -1.0f, -1.0f, -1.0f });
					dNI = _dot_d(rd, norm);
				}

				int triangleAttr_i = minTrid * 5;
				float3 color = {
					_attrBufferData[triangleAttr_i],
					_attrBufferData[triangleAttr_i + 1],
					_attrBufferData[triangleAttr_i + 2],
				};

				float Kr = _attrBufferData[triangleAttr_i + 3];
				bool isRefract = _attrBufferData[triangleAttr_i + 4] > 0.0f;

				if (!isRefract)
				{
					rd = _sub_d(rd, _mul_d({ 2.0f, 2.0f, 2.0f }, _mul_d({ dNI, dNI, dNI }, norm)));
				}
				else
				{
					eta = 1.0f / eta;

					float k = 1.0f - eta * eta * (1.0f - dNI * dNI);
					if (k < 0.0f)
					{
						rd = _sub_d(rd, _mul_d({ 2.0f, 2.0f, 2.0f }, _mul_d({ dNI, dNI, dNI }, norm)));
					}
					else
					{
						float n = eta * dNI + sqrtf(k);
						rd = _sub_d(_mul_d({ eta, eta, eta }, rd), _mul_d({ n, n, n }, norm));
					}
				}

				rd.x += curand_uniform(&_randAr[(bid + blockIdx.y * 128) * 32 + tid]) * Kr - 0.5f * Kr;
				rd.y += curand_uniform(&_randAr[(bid + blockIdx.y * 128) * 32 + tid]) * Kr - 0.5f * Kr;
				rd.z += curand_uniform(&_randAr[(bid + blockIdx.y * 128) * 32 + tid]) * Kr - 0.5f * Kr;
				rd = _norm_d(rd);

				ro = _add_d(ro, _mul_d(rd, { 0.001f, 0.001f, 0.001f }));

				rc.x *= color.x;
				rc.y *= color.y;
				rc.z *= color.z;
				if (color.x + color.y + color.z > 10.0f) break;
			}
			else
			{
				rc = { 0.0f, 0.0f, 0.0f };
			}
		}
		ro = _add_d(ro, _mul_d(rd, { 1.0f / (float)((depth + 1) * (depth + 1)), 1.0f / (float)((depth + 1) * (depth + 1)), 1.0f / (float)((depth + 1) * (depth + 1)) }));
	}

	int basePid = ((tid + _topLeftCoord.x) + (bid + _topLeftCoord.y) * (_screenRes.x)) * 3;
	_protoBufferData[basePid] = _protoBufferData[basePid] * 0.9f + min(rc.x, 1.0f) * 0.1f;
	_protoBufferData[basePid + 1] = _protoBufferData[basePid + 1] * 0.9f + min(rc.y, 1.0f) * 0.1f;
	_protoBufferData[basePid + 2] = _protoBufferData[basePid + 2] * 0.9f + min(rc.z, 1.0f) * 0.1f;
}

__global__ void _FragmentShader_d(float* _protoBufferData, unsigned char* _frameBufferData, int2 _screenRes_d)
{
	int basePid = bid * (_screenRes_d.x << 1) + tid;
	int Pid_ = basePid * 4;
	int _Pid = basePid * 3;
	_frameBufferData[Pid_] = (int)((_protoBufferData[_Pid]) * 400);
	_frameBufferData[Pid_ + 1] = (int)((_protoBufferData[_Pid + 1]) * 400);
	_frameBufferData[Pid_ + 2] = (int)((_protoBufferData[_Pid + 2]) * 400);
	_frameBufferData[Pid_ + 3] = 255;
}
void _FragmentShader()
{
	_FragmentShader_d<<<_screenRes.y << 1, _screenRes.x << 1>>>(_protoBuffer.data, _frameBuffer.data, _screenRes);
}
void _PathTrace()
{
	for (int y = 0;y < _screenRes.y << 1;y += 128)
	{
		for (int x = 0;x < _screenRes.x << 1;x += 128)
		{
			_PathTrace_d<<<dim3 { (unsigned int)128, (unsigned int)128, (unsigned int)1 }, 64>>>(_faceBuffer.data, _faceBuffer.size / 3, _vertexBuffer.data, _bvhBuffer.data, _bvhBuffer.size / 8, _attrBuffer.data, _protoBuffer.data, randAr_d, { _screenRes.x << 1, _screenRes.y << 1 }, { x, y }, _camPos, _camRot);
			
		}
		_FragmentShader();
		aeWindowUpdate();
	}
}

void _OctreePathTrace()
{
	for (int y = 0;y < _screenRes.y << 1;y += 512)
	{
		for (int x = 0;x < _screenRes.x << 1;x += 512)
		{
			_OctreePathTrace_d<<<512, 512>>>(_octree, 10, _protoBuffer.data, _vertexBuffer.data, _attrBuffer.data, randAr_d, { _screenRes.x << 1, _screenRes.y << 1 }, { x, y }, _camPos, _camRot);
			_FragmentShader();
			aeWindowUpdate();
		}
	}
}

__global__ void _OctreeGen(octreeNode* octree, float* vertices, int* faces, int fid, curandState* _randAr)
{

	octree->min = { -10.0f, -10.0f, -10.0f };
	octree->max = { 10.0f, 10.0f, 10.0f };
	octreeNode* curNode = octree;

	int vert_i = faces[bid * 3] * 3;
	float3 v0{ vertices[vert_i], vertices[vert_i + 1], vertices[vert_i + 2] };
	vert_i = faces[bid * 3 + 1] * 3;
	float3 v1{ vertices[vert_i], vertices[vert_i + 1], vertices[vert_i + 2] };
	vert_i = faces[bid * 3 + 2] * 3;
	float3 v2{ vertices[vert_i], vertices[vert_i + 1], vertices[vert_i + 2] };

	int stack[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int depth = 0;

	while (true)
	{
		if (curNode == NULL)
		{
			curNode = octree;
			depth = 0;
			stack[depth]++;
			if (stack[depth] > 7)
			{
				break;
			}
		}

		float sF = (float)((depth + 1) * (depth + 1));
		bool bx = (stack[depth] & 0b1) == 1;
		bool by = (stack[depth] & 0b10) == 2;
		bool bz = (stack[depth] & 0b100) == 4;
		
		float3 minC = curNode->min, maxC = curNode->max;

		if (bx) minC.x += 5.0f / sF;
		else maxC.x -= 5.0f / sF;
		if (by) minC.y += 5.0f / sF;
		else maxC.y -= 5.0f / sF;
		if (bz) minC.z += 5.0f / sF;
		else maxC.z -= 5.0f / sF;

		float3 center = _div_d(_add_d(minC, maxC), { 2.0f, 2.0f, 2.0f });

		float3 ro = center;
		float3 rd = _norm_d(_sub_d(_div_d(_add_d(_add_d(v0, v1), v2), { 3.0f, 3.0f, 3.0f }), ro));




		float t1, t2, u, v;
		_rayBboxIntersect(
			ro,
			rd,
			minC,
			maxC,
			t1
		);
		_rayTriangleIntersect(ro, rd, v0, v1, v2, &t2, &u, &v);
		bool moved = t2 > t1;

		if (!moved || curNode->faceData != NULL)
		{
			if (curNode->faceData != NULL)
			{
				if (curNode->stride < 50 * 3 - 3)
				{
					curNode->faceData[curNode->stride] = faces[bid * 3];
					curNode->faceData[curNode->stride + 1] = faces[bid * 3 + 1];
					curNode->faceData[curNode->stride + 2] = faces[bid * 3 + 2];
					curNode->stride += 3;
				}
			}
			stack[depth]++;
		}
		else
		{
			if (curNode->children[stack[depth]] == NULL)
			{
				octreeNode* node = new octreeNode;
				if (node == NULL) break;
				node->min = minC;
				node->max = maxC;
				node->b = curNode;
				curNode->children[stack[depth]] = node;

				depth++;
				if (depth >= 9)
				{
					node->faceData = (int*)malloc(sizeof(int) * 50 * 3);
					if (node->faceData == NULL)
					{
						printf("Bruh\n");
						break;
					}
					else
					{
						printf("Dick\n");
					}
					node->faceData[0] = faces[bid * 3];
					node->faceData[1] = faces[bid * 3 + 1];
					node->faceData[2] = faces[bid * 3 + 2];
					node->stride += 3;
				}
				curNode = node;
				stack[depth] = 0;
			}
			else
			{
				depth++;
				curNode = curNode->children[stack[depth]];
			}
		}

		bool isBreak = false;
		for (int n = 0;n < 10;n++)
		{
			if (stack[depth] > 7)
			{
				if (depth <= 0)
				{
					isBreak = true;
					break;
				}
				stack[depth] = 0;
				depth--;
				curNode = curNode->b;
				stack[depth]++;
			}
		}
		if (isBreak) break;
	}
	
}

void aeOctree()
{
	cudaMalloc(&_octree, sizeof(octreeNode));
	_OctreeGen << <_faceBuffer.size / 3, 1 >> > (_octree, _vertexBuffer.data, _faceBuffer.data, 0, randAr_d);
	aeErrorCheck();
}

void aeRenderBuffer()
{
	_VertexShader();
	_GeometryShader();
	_OctreePathTrace();
	_FragmentShader();
	std::cout << "[AE2-DEBUG] RENDER_LAST_ERROR " << cudaGetErrorString(cudaGetLastError()) << std::endl;
}

void aeErrorCheck()
{
	std::cout << "[AE2-DEBUG] LAST_ERROR " << cudaGetErrorString(cudaGetLastError()) << std::endl;
}

void aeCreateWindow(const char* title, bool fullscreen)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		SDL_Quit();
		exit(0);
	}
	_window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _screenRes.x << 1, _screenRes.y << 1, NULL);
	_windowSurface = SDL_GetWindowSurface(_window);
	_windowRenderer = SDL_CreateRenderer(_window, -1, 0);
}
void aePollEvents()
{
	SDL_Event e;
	while (SDL_PollEvent(&e)) {
		if (e.type == SDL_QUIT) {
			SDL_DestroyWindow(_window);
		}
	}
}
void aeWindowUpdate()
{
	unsigned char* _frameBuffer_h = (unsigned char*)malloc(sizeof(unsigned char) * (_screenRes.x << 1) * (_screenRes.y << 1) * 4);
	cudaMemcpy(_frameBuffer_h, _frameBuffer.data, sizeof(unsigned char) * (_screenRes.x << 1) * (_screenRes.y << 1) * 4, cudaMemcpyDeviceToHost);
	SDL_Surface* surf = SDL_CreateRGBSurfaceFrom((void*)_frameBuffer_h, _screenRes.x << 1, _screenRes.y << 1, 32, (_screenRes.x << 1) * 4, 0xFF0000, 0x00FF00, 0x0000FF, 0x000000);
	SDL_UpperBlit(surf, NULL, _windowSurface, NULL);
	SDL_UpdateWindowSurface(_window);
	free(_frameBuffer_h);
}
void aeSaveWindow(const char* name)
{
	SDL_SaveBMP(_windowSurface, name);
}
