#include "MTexture.h"


TextureContainer g_TextureContainer;

Texture::~Texture()
{
	if (m_Bitmap)
		delete[] m_Bitmap;
}

void Texture::SetTexPath(char *pPath)
{
	strcpy(szTexPath, pPath);
}

bool Texture::CmpTexPath(char *pPath)
{
	if (!strcmp(szTexPath, pPath))
		return true;
	else
		return false;
}

void Texture::LoadTexture(char* a_File, IMAGETYPE type)
{
	strcpy(szTexPath, a_File);
	FILE* f = fopen(a_File, "rb");
	if (f)
	{
		m_ImageType = type;
		switch (type)
		{
		case BITMAP24:
			{
				fseek(f, 0x12, SEEK_SET);       
				fread(&m_Width, sizeof(int), 1, f);  
				fseek(f, 0x16, SEEK_SET); 
				fread(&m_Height, sizeof(int), 1, f);
				fseek(f, 0x36, SEEK_SET);
				m_Bitmap = new Vec4f[m_Width*m_Height];
				unsigned char tmp1;
				int offset = m_Width%4;

				// 开始加载数据
				unsigned int curfilepointer = 0x36;
				MemoryFile *mf = new MemoryFile(10000000);
				for (int k = 0; k < m_Height; ++k)
				{
					for(int m = 0; m < m_Width; ++m)
					{
						if (mf->CheckEnd())
						{
							if(!mf->ReadFromFile(f, curfilepointer))
							{
								delete mf;
								return;
							}
							curfilepointer += mf->size;
						}
						tmp1 = (unsigned char)mf->pBuffer[mf->pointer];
						++mf->pointer;
						m_Bitmap[k*m_Width+m].z = float(tmp1)/255;

						if (mf->CheckEnd())
						{
							if(!mf->ReadFromFile(f, curfilepointer))
							{
								delete mf;
								return;
							}
							curfilepointer += mf->size;
						}
						tmp1 = (unsigned char)mf->pBuffer[mf->pointer];
						++mf->pointer;
						m_Bitmap[k*m_Width+m].y = float(tmp1)/255;

						if (mf->CheckEnd())
						{
							if(!mf->ReadFromFile(f, curfilepointer))
							{
								delete mf;
								return;
							}
							curfilepointer += mf->size;
						}
						tmp1 = (unsigned char)mf->pBuffer[mf->pointer];
						++mf->pointer;
						m_Bitmap[k*m_Width+m].x = float(tmp1)/255;
					}
					mf->pointer += offset;
				}

				delete mf;
			}
			break;
		case TGA:                    // 从Jacco Bikker, a.k.a. "The Phantom"抄袭而来
			break;
		default:
			break;
		}
	}
}

Texture::Texture(char* a_File, IMAGETYPE type, int textype)
{
	LoadTexture(a_File, type);
	m_Textype = textype;
}


// 从Jacco Bikker, a.k.a. "The Phantom"的代码中抄袭而来
//void Texture::GetTexel(float a_U, float a_V, ColorRate4 *cr)
//{
//	// fetch a bilinearly filtered texel
//	a_U -= int(a_U);
//	a_V -= int(a_V);
//	if (a_U < 0) a_U += 1;
//	if (a_V < 0) a_V += 1;
//	float fu = a_U * (m_Width - 1);
//	float fv = a_V * (m_Height - 1);
//	int u1 = (int)fu;
//	int v1 = (int)fv;
//	int u2 = (u1 + 1) % m_Width;
//	int v2 = (v1 + 1) % m_Height;
//	// calculate fractional parts of u and v
//	float fracu = fu - floorf( fu );
//	float fracv = fv - floorf( fv );
//	// calculate weight factors
//	float w1 = (1 - fracu) * (1 - fracv);
//	float w2 = fracu * (1 - fracv);
//	float w3 = (1 - fracu) * fracv;
//	float w4 = fracu *  fracv;
//	// fetch four texels
//	ColorRate4 c1 = m_Bitmap[u1 + v1 * m_Width];
//	ColorRate4 c2 = m_Bitmap[u2 + v1 * m_Width];
//	ColorRate4 c3 = m_Bitmap[u1 + v2 * m_Width];
//	ColorRate4 c4 = m_Bitmap[u2 + v2 * m_Width];
//	// scale and sum the four colors
//	cr->r = c1.r * w1 + c2.r * w2 + c3.r * w3 + c4.r * w4;
//	cr->g = c1.g * w1 + c2.g * w2 + c3.g * w3 + c4.g * w4;
//	cr->b = c1.b * w1 + c2.b * w2 + c3.b * w3 + c4.b * w4;
//}


void TextureContainer::AddTexture(Texture *pt)
{
	arrayTexture.push_back(pt);
}

TextureContainer::TextureContainer()
{
	arrayTexture.clear();
}

TextureContainer::~TextureContainer()
{
	for (int i = 0; i < arrayTexture.size(); ++i)
	{
		delete arrayTexture[i];
	}
	arrayTexture.clear();
}

Texture* TextureContainer::FindTexByName(char *szPath)
{
	for (int i = 0; i < arrayTexture.size(); ++i)
	{
		if (arrayTexture[i]->CmpTexPath(szPath))
			return arrayTexture[i];
	}
	return NULL;
}