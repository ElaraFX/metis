#pragma once
#include <vector>
#include "linear_math.h"

#define MAX_PATH 260

struct MMaterial
{
	//float m_Emission;             // �Է���ϵ��(������ɫ����m_ColorReflectһ��)(��ֵ���Դ���1)
	Vec3f m_ColorReflect;		// RGB������ɫ�ķ�����(������ɫ)(ע��:һ����֤ȡֵ��Χ��[0,1],��Ȼ���ܻ����)
	Vec3f m_SpecColorReflect;	// RGB������ɫ�ľ��淴����(������ɫ)(ע��:һ����֤ȡֵ��Χ��[0,1],��Ȼ���ܻ����)
	float m_transparencyRate;		// 
	float m_glossiness;
	float m_ior;
	int m_index;
	char m_szNameMtl[MAX_PATH];	    // ��������

	// ��Ա����
	MMaterial();
	~MMaterial();
};

// ��һ���ļ�Ϊ��λ
struct MATERIALCONTAINER
{
	char szPathName[MAX_PATH];						// �ļ���
	std::vector<MMaterial*> arrayMaterial;
	MATERIALCONTAINER();
	~MATERIALCONTAINER();
	void AddMaterial(MMaterial* g);
	void LoadMaterialFromMtl(char* pName);
};

extern MATERIALCONTAINER g_MaterialContainer;