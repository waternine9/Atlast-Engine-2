#pragma once

#define AE_BACKFACE_CULLING 1
#define AE_FRONTFACE_CULLING 1 << 1
#define AE_VERTEX_BUFFER 1 << 2
#define AE_FACE_BUFFER 1 << 3
#define AE_GEOMETRY_BUFFER 1 << 4
#define AE_ATTR_BUFFER 1 << 5
#define AE_TEXTURE 1 << 6
#define AE_FRAME_BUFFER 1 << 7
#define AE_VERTEX_COLORS 1 << 8

__declspec(dllexport) struct aeBuffer
{
	float* data = 0;
	size_t size = 0;
	short flags = 0;
};
__declspec(dllexport) struct aeBufferI
{
	int* data = 0;
	size_t size = 0;
	short flags = 0;
};
__declspec(dllexport) struct aeBufferUC
{
	unsigned char* data = 0;
	size_t size = 0;
	short flags = 0;
};
__declspec(dllexport) void aeGenBuffer(aeBuffer* buffer, size_t size);
__declspec(dllexport) void aeGenBuffer(aeBufferI* buffer, size_t size);
__declspec(dllexport) void aeGenBuffer(aeBufferUC* buffer, size_t size);
__declspec(dllexport) void aeAddVBO(aeBuffer* buffer);
__declspec(dllexport) void aeAddFBO(aeBufferI* buffer);
__declspec(dllexport) void aeAddBVH(aeBuffer* buffer);
__declspec(dllexport) void aeAddABO(aeBuffer* buffer);
__declspec(dllexport) void aeBufferData(aeBuffer* buffer, float* data, size_t size);
__declspec(dllexport) void aeBufferData(aeBufferI* buffer, int* data, size_t size);
__declspec(dllexport) void aeBufferData(aeBufferUC* buffer, unsigned char* data, size_t size);
__declspec(dllexport) void aeAddFrame(aeBufferUC* buffer);
__declspec(dllexport) void aeFreeFrame();
__declspec(dllexport) void aeAddProto(aeBuffer* buffer);
__declspec(dllexport) void aeFreeProto();
__declspec(dllexport) void aeRotate(size_t idx, float ax, float ay, float az, float cx, float cy, float cz);
__declspec(dllexport) void aeCameraTransform(float rx, float ry, float rz, float cx, float cy, float cz);
__declspec(dllexport) void aeOctree();
__declspec(dllexport) void aeRandInit();
__declspec(dllexport) void aeFreeBuffer(aeBuffer* buffer);
__declspec(dllexport) void aeFreeBuffer(aeBufferI* buffer);
__declspec(dllexport) void aeSetFlag(aeBuffer* buffer, char flag);
__declspec(dllexport) void aeErrorCheck();
__declspec(dllexport) void aeViewport(int res_x, int res_y, float fov);
__declspec(dllexport) void aeRenderBuffer();
__declspec(dllexport) void aeCreateWindow(const char* title, bool fullscreen);
__declspec(dllexport) void aePollEvents();
__declspec(dllexport) void aeWindowUpdate();
__declspec(dllexport) void aeSaveWindow(const char* name);