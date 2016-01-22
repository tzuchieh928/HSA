#ifndef __SVM_KDTree_NODE__
#define __SVM_KDTree_NODE__
#define MAX_DIM 128

/* binary tree node definition */
struct hsaNode
{
	float des[MAX_DIM];
	int index;
	float x;
	float y;
	struct hsaNode* left;
	struct hsaNode* right;
} ;

#endif