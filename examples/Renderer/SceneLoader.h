#ifndef __LOADER_H_
#define __LOADER_H_

#include "Geometry.h"
#include "MMaterial.h"
#include "MTexture.h"

struct TextureCUDA;

struct SceneInfo
{
	unsigned verticesNo;
	unsigned normalsNo;
	unsigned trianglesNo;
	unsigned materialNo;
	unsigned textureNo;
	unsigned textotalsize;
	unsigned uvNo;
	Vertex* vertices;   // vertex list
	Vec3f* normals;
	Vec3f* uvs;
	Triangle* triangles;  // triangle list
	MaterialCUDA* materials;
	TextureCUDA* textures;
	SceneInfo()
	{
		verticesNo = 0;
		normalsNo = 0;
		trianglesNo = 0;
		materialNo = 0;
		textureNo = 0;
		textotalsize = 0;
		uvNo = 0;
		vertices = NULL;   // vertex list
		normals = NULL;
		uvs = NULL;
		triangles = NULL;  // triangle list
		materials = NULL;
		textures = NULL;
	}
};

extern SceneInfo scene_info;

void panic(const char *fmt, ...);
void load_object(const char *filename);
void load_material(const char* strFileName);
float processgeo();

#endif
