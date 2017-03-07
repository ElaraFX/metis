#pragma once
#include <stdio.h>
#include <math.h>

struct MemoryFile
{
	char *pBuffer;
	unsigned int pointer;
	unsigned int size;
	unsigned int maxsize;
	MemoryFile(unsigned int nsize);
	~MemoryFile();
	void IgnoreGap();
	char Readchar();
	char Peekchar();
	void ReadstrUntilGap(char *p);
	void IgnoreUntilchar(char c);
	float Readfloat();
	bool CheckEnd();
	unsigned int CutOffEndByChar(char c);                 // 由某一字符来截断文件尾(返回当前BUFFER SIZE)
	bool ReadFromFile(FILE *pf, unsigned int offset);
};


