#ifndef __SVM_KDTree_NODE__
#define __SVM_KDTree_NODE__
#define MAX_DIM 128

/* binary tree node definition */
typedef struct nodeStruct
{
	float des[MAX_DIM];
	int index;
	float x;
	float y;
	struct nodeStruct* left;
	struct nodeStruct* right;
} node;

#endif