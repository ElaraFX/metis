#ifndef __LOADER_H_
#define __LOADER_H_

#include "Geometry.h"
#include "MMaterial.h"
#include "MTexture.h"

struct TextureCUDA;

extern unsigned verticesNo;
extern unsigned normalsNo;
extern unsigned materialNo;
extern unsigned textureNo;
extern unsigned uvNo;
extern Vertex* vertices;
extern Vec3f* normals;
extern Vec3f* uvs;
extern unsigned int trianglesNo;
extern Triangle* triangles;
extern MaterialCUDA* materials;
extern TextureCUDA* textures;

void panic(const char *fmt, ...);
void load_object(const char *filename);
void load_material(const char* strFileName);
float processgeo();

#endif
