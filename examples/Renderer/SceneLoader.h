#ifndef __LOADER_H_
#define __LOADER_H_

#include "Geometry.h"

extern unsigned verticesNo;
extern unsigned normalsNo;
extern Vertex* vertices;
extern Vec3f* normals;
extern unsigned int trianglesNo;
extern Triangle* triangles;

void panic(const char *fmt, ...);
void load_object(const char *filename);
float processgeo();

#endif
