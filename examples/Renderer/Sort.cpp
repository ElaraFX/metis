#include "Sort.h"

#define FW_ASSERT(X) ((void)0)
#define QSORT_STACK_SIZE    32

static inline void  InsertionSort(int start, int size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc);
static inline int   Median3(int low, int high, void* data, SortCompareFunc compareFunc);
static void         Qsort(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc);


void InsertionSort(int start, int size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
	FW_ASSERT(compareFunc && swapFunc);
	FW_ASSERT(size >= 0);

	for (int i = 1; i < size; i++)
	{
		int j = start + i - 1;
		while (j >= start && compareFunc(data, j, j + 1) > 0)
		{
			swapFunc(data, j, j + 1);
			j--;
		}
	}
}

//------------------------------------------------------------------------

int Median3(int low, int high, void* data, SortCompareFunc compareFunc)
{
	FW_ASSERT(compareFunc);
	FW_ASSERT(low >= 0 && high >= 2);

	int l = low;
	int c = (low + high) >> 1;
	int h = high - 2;

	if (compareFunc(data, l, h) > 0) swap(l, h);
	if (compareFunc(data, l, c) > 0) c = l;
	return (compareFunc(data, c, h) > 0) ? h : c;
}

//------------------------------------------------------------------------

void Qsort(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
	FW_ASSERT(compareFunc && swapFunc);
	FW_ASSERT(low <= high);

	int stack[QSORT_STACK_SIZE];
	int sp = 0;
	stack[sp++] = high;

	while (sp)
	{
		high = stack[--sp];
		FW_ASSERT(low <= high);

		// Use insertion sort for small values or if stack gets full.

		if (high - low <= 15 || sp + 2 > QSORT_STACK_SIZE)
		{
			InsertionSort(low, high - low, data, compareFunc, swapFunc);
			low = high + 1;
			continue;
		}

		// Select pivot using median-3, and hide it in the highest entry.

		swapFunc(data, Median3(low, high, data, compareFunc), high - 1);

		// Partition data.

		int i = low - 1;
		int j = high - 1;
		for (;;)
		{
			do
				i++;
			while (compareFunc(data, i, high - 1) < 0);
			do
				j--;
			while (compareFunc(data, j, high - 1) > 0);

			FW_ASSERT(i >= low && j >= low && i < high && j < high);
			if (i >= j)
				break;

			swapFunc(data, i, j);
		}

		// Restore pivot.

		swapFunc(data, i, high - 1);

		// Sort sub-partitions.

		FW_ASSERT(sp + 2 <= QSORT_STACK_SIZE);
		if (high - i > 2)
			stack[sp++] = high;
		if (i - low > 1)
			stack[sp++] = i;
		else
			low = i + 1;
	}
}

//------------------------------------------------------------------------

void Sort(int start, int end, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
	FW_ASSERT(start <= end);
	FW_ASSERT(compareFunc && swapFunc);

	if (start + 2 <= end)
		Qsort(start, end, data, compareFunc, swapFunc);
}

//------------------------------------------------------------------------

typedef signed int S32;
typedef float F32;

int CompareS32(void* data, int idxA, int idxB)
{
	S32 a = ((S32*)data)[idxA];
	S32 b = ((S32*)data)[idxB];
	return (a < b) ? -1 : (a > b) ? 1 : 0;
}

//------------------------------------------------------------------------

void SwapS32(void* data, int idxA, int idxB)
{
	swap(((S32*)data)[idxA], ((S32*)data)[idxB]);
}

//------------------------------------------------------------------------

int CompareF32(void* data, int idxA, int idxB)
{
	F32 a = ((F32*)data)[idxA];
	F32 b = ((F32*)data)[idxB];
	return (a < b) ? -1 : (a > b) ? 1 : 0;
}

//------------------------------------------------------------------------

void SwapF32(void* data, int idxA, int idxB)
{
	swap(((F32*)data)[idxA], ((F32*)data)[idxB]);
}

//------------------------------------------------------------------------
