#pragma once
#include <stdio.h>
#include <vector>
#include "linear_math.h"
#include "GReadFileToMemory.h"


using namespace std;


enum IMAGETYPE
{ 
	BITMAP24 = 0,
	TGA,
};

// 纹理类
class Texture
{
public:
	Vec4f *m_Bitmap;
	int m_Width, m_Height;
	IMAGETYPE m_ImageType;
	int m_index;
	int m_Textype;            // textype:0(普通贴图),1(法线贴图)
	char szTexPath[128];      // 纹理的路径
public:
	Texture(char* a_File, IMAGETYPE type, int textype);            // textype:0(普通贴图),1(法线贴图)
	~Texture();
	void LoadTexture(char* a_File, IMAGETYPE type);
	Vec4f* GetBitmap() {return m_Bitmap;}
	void GetTexel(float a_U, float a_V, Vec4f *cr);
	int GetWidth() {return m_Width;}
	int GetHeight() {return m_Height;}
	int GetTexType();
	void SetTexPath(char *pPath);
	bool CmpTexPath(char *pPath);
};

struct TextureContainer
{
	vector<Texture*> arrayTexture;
	TextureContainer();
	~TextureContainer();
	void AddTexture(Texture *pt);
	Texture* FindTexByName(char *szPath);
};

struct TextureCUDA
{
	int height;
	int width;
	Vec4f* texels;

	TextureCUDA()
	{
	}

	//void InitTextureFromCpu(Texture *t)
	//{
	//	cudaMalloc(&gpuTexels, height * width * sizeof(float4));
	//	cudaMemcpy(gpuTexels, t->GetBitmap(), height * width * sizeof(float4), cudaMemcpyHostToDevice);

	//	diffuse_tex.filterMode = cudaFilterModeLinear;

	//	cudaChannelFormatDesc channeldesc = cudaCreateChannelDesc<float4>(); 
	//	cudaBindTexture(NULL, &diffuse_tex, gpuTexels, &channeldesc, height * width * sizeof(float4));  // 2k map:
	//}

	~TextureCUDA()
	{
	}
};

extern TextureContainer g_TextureContainer;