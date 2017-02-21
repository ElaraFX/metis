#pragma once
#include <vector>
#include "linear_math.h"

#define MAX_PATH 260

struct MMaterial
{
	//float m_Emission;             // 自发光系数(发光颜色与其m_ColorReflect一致)(该值可以大于1)
	Vec3f m_ColorReflect;		// RGB三种颜色的反射率(物体颜色)(注意:一定保证取值范围在[0,1],不然可能会出错)
	Vec3f m_SpecColorReflect;	// RGB三种颜色的镜面反射率(物体颜色)(注意:一定保证取值范围在[0,1],不然可能会出错)
	float m_transparencyRate;		// 
	float m_glossiness;
	float m_ior;
	int m_index;
	char m_szNameMtl[MAX_PATH];	    // 材质名称

	// 成员函数
	MMaterial();
	~MMaterial();
};

// 以一个文件为单位
struct MATERIALCONTAINER
{
	char szPathName[MAX_PATH];						// 文件名
	std::vector<MMaterial*> arrayMaterial;
	MATERIALCONTAINER();
	~MATERIALCONTAINER();
	void AddMaterial(MMaterial* g);
	void LoadMaterialFromMtl(char* pName);
};

extern MATERIALCONTAINER g_MaterialContainer;