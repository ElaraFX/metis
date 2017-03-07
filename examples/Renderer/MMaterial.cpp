#include "MMaterial.h"



MATERIALCONTAINER g_MaterialContainer;

MMaterial::MMaterial()
{
	m_ColorReflect.x = 1;
	m_ColorReflect.y = 1;
	m_ColorReflect.z = 1;
	m_SpecColorReflect.x = 0;
	m_SpecColorReflect.y = 0;
	m_SpecColorReflect.z = 0;
	m_transparencyRate = 0;
	m_glossiness = 0;
	m_ior = 1.5;
	m_index = 0;
	m_pTexture = NULL;
}

MMaterial::~MMaterial()
{

}

MATERIALCONTAINER::MATERIALCONTAINER()
{
	arrayMaterial.clear();
}

MATERIALCONTAINER::~MATERIALCONTAINER()
{
	for (int i = 0; i < arrayMaterial.size(); ++i)
	{
		delete arrayMaterial[i];
	}
	arrayMaterial.clear();
}

void MATERIALCONTAINER::AddMaterial(MMaterial* g)
{
	arrayMaterial.push_back(g);
}