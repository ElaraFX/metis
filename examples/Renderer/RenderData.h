#pragma once
#include "linear_math.h"
#include <cuda.h>
#include "stdio.h"
#include "Camera.h"
#include "MMaterial.h"

//#define scrwidth 1280
//#define scrheight 720

struct controlParam
{
	int	m_windowSize;
	float m_variance_pos;
	float m_variance_col;
	float m_color_saturation;
	float m_exposure_value;
	float m_whitepoint;
	float m_shadows;
	float m_midtones;
	float m_highlights;
};

struct enviParam
{
	int HDRwidth;
	int HDRheight;
};

// this structure need to be copied into GPU memory 
struct gpuData
{
	float4* cudaNodePtr;
    float4* cudaTriNormalPtr;
    float4* cudaTriDebugPtr;
    float4* cudaTriUVPtr;
    int*    cudaTriIndicesPtr;
	MaterialCUDA* cudaMaterialsPtr;
	TextureCUDA* cudaTexturePtr;
	float4* cudaTextureData;
	enviParam* cudaEnvi;
	Camera* cudaRendercam;
	Vec3f* accumulatebuffer; // image buffer storing accumulated pixel samples
	Vec3f* AOVdirectdiffuse; 
	float* AOVdiffusecount; 
	Vec3f* AOVspecular; 
	Vec3f* AOVindirectdiffuse; 
	Vec3f* AOVindirectspecular; 
    Vec3f* normalbuffer;	// stores ray intersect normal per pixel samples
    float* materialbuffer; // stores ray intersect material per pixel samples
	gpuData()
	{
		cudaNodePtr = NULL;
		cudaTriNormalPtr = NULL;
		cudaTriDebugPtr = NULL;
		cudaTriUVPtr = NULL;
		cudaTriIndicesPtr = NULL;
		cudaMaterialsPtr = NULL;
		cudaTexturePtr = NULL;
		cudaTextureData = NULL;
		cudaEnvi = NULL;
		cudaRendercam = NULL;
		accumulatebuffer = NULL; // image buffer storing accumulated pixel samples
		AOVdirectdiffuse = NULL;
		AOVdiffusecount = NULL;
		AOVspecular = NULL;
		AOVindirectdiffuse = NULL;
		AOVindirectspecular = NULL;
		normalbuffer = NULL;
		materialbuffer = NULL;
	}
	~gpuData()
	{
		cudaFree(cudaNodePtr);
		cudaFree(cudaTriNormalPtr);
		cudaFree(cudaTriDebugPtr);
		cudaFree(cudaTriUVPtr);
		cudaFree(cudaTriIndicesPtr);
		cudaFree(cudaMaterialsPtr);
		cudaFree(cudaTexturePtr);
		cudaFree(cudaTextureData);
		cudaFree(cudaEnvi);
		cudaFree(cudaRendercam);
		cudaFree(accumulatebuffer);
		cudaFree(AOVdirectdiffuse);
		cudaFree(AOVdiffusecount);
		cudaFree(AOVspecular);
		cudaFree(AOVindirectdiffuse);
		cudaFree(AOVindirectspecular);
		cudaFree(normalbuffer);
		cudaFree(materialbuffer);
	}
};
