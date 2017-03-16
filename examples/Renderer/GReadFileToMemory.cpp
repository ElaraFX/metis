#include <stdlib.h>
#include "GReadFileToMemory.h"



MemoryFile::MemoryFile(unsigned int nsize)
{
	pBuffer = new char[nsize];
	maxsize = nsize;
	pointer = 0;
	size = 0;
}

MemoryFile::~MemoryFile()
{
	delete[] pBuffer;
}

char MemoryFile::Readchar()
{
	return pBuffer[pointer++];
}

char MemoryFile::Peekchar()
{
	return pBuffer[pointer];
}

void MemoryFile::IgnoreGap()
{
	if (size == 0)
		return;
	while(pBuffer[pointer] == '\n' || pBuffer[pointer] == ' '|| pBuffer[pointer] == '\r')
	{
		++pointer;
	}
}

void MemoryFile::ReadstrUntilGap(char *p)
{
	int i = 0;
	IgnoreGap();
	while(pBuffer[pointer] != '\n' && pBuffer[pointer] != ' ' && pBuffer[pointer] != '\r')
	{
		p[i] = pBuffer[pointer];
		++pointer;
		++i;
	}
	p[i] = '\0';
	++pointer;
}

float MemoryFile::Readfloat()
{
	IgnoreGap();
	char szfloat[32] = {0};
	int i = 0;
	while(pBuffer[pointer] >= '-' && pBuffer[pointer] <= '9' && pBuffer[pointer] != '/')
	{
		szfloat[i] = pBuffer[pointer];
		++pointer;
		++i;
	}
	return float(atof(szfloat));
}

int MemoryFile::Readint()
{
	IgnoreGap();
	char szint[32] = {0};
	int i = 0;
	while(pBuffer[pointer] >= '-' && pBuffer[pointer] <= '9' && pBuffer[pointer] != '/')
	{
		szint[i] = pBuffer[pointer];
		++pointer;
		++i;
	}
	return atoi(szint);
}

void MemoryFile::IgnoreUntilchar(char c)
{
	while(pBuffer[pointer] != c)
	{
		++pointer;
	}
	++pointer;
}

bool MemoryFile::CheckEnd()
{
	if (pointer >= size)
		return true;
	return false;
}

unsigned int MemoryFile::CutOffEndByChar(char c)
{
	while (size > 0)
	{
		if (pBuffer[size - 1] == c)
			break;
		--size;
	}
	return size;
}

bool MemoryFile::ReadFromFile(FILE *pf, unsigned int offset)
{
	if (pf == NULL)
		return false;

	fseek(pf, offset, SEEK_SET);       
	size = fread(pBuffer, 1, maxsize, pf);  
	if (size == 0)
		return false;

	pointer = 0;
	return true;
}