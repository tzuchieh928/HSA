#ifndef _OCL2KDTree_H_
#define _OCL2KDTree_H_
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "CLUtil.hpp"
#include "SVMKDTreeNode.h"
using namespace appsdk;
using namespace std;
using namespace cv;

#define SAMPLE_VERSION      "AMD-APP-SDK-v3.0.130.2"
#define OCL_COMPILER_FLAGS  "SVMKDTreeSearch_OclFlags.txt"
class OCL2KDTree{
private:
	/* OpenCL runtime */
	cl_context            context;
	cl_device_id*         devices;
	cl_command_queue      commandQueue;
	cl_program            program;
	cl_kernel             sample_kernel;


	SDKDeviceInfo         deviceInfo;
	KernelWorkGroupInfo   kernelInfo;

	/* Timing information */
	cl_double             setupTime;
	cl_double             kernelTime;
	SDKTimer*             sampleTimer;
	/* svm buffer for binary tree */
	void*                 svmTreeBuf;

	/* svm buffer for search keys */
	void*                 svmSearchBuf;
	struct hsaNode *root;
	struct hsaNode *found;
	int retValue;
	FILE *fsvmkdmatch;
public:
	CLCommandArgs*       sampleArgs;
	
	OCL2KDTree()
	{
		sampleArgs = new CLCommandArgs();
		sampleTimer = new SDKTimer();
		sampleArgs->sampleVerStr = SAMPLE_VERSION;
		sampleArgs->flags = OCL_COMPILER_FLAGS;
		fsvmkdmatch = fopen("svmkdmatch.txt", "w");
	};


	~OCL2KDTree()
	{
		delete sampleArgs;
		delete sampleTimer;
		fclose(fsvmkdmatch);
	};

	int setupCL();
	int initialize();
	int genBinaryImage();
	int createTree(int len, int i, int dim);
	int dataMarshalling(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat descriptors1, Mat descriptors2);
	int findNearest(vector<KeyPoint> keypoints2, Mat descriptors2);

	void swap(struct hsaNode *x, struct hsaNode *y) {
		struct hsaNode tmp;
		memcpy(&tmp, x, sizeof(tmp));
		memcpy(x, y, sizeof(tmp));
		memcpy(y, &tmp, sizeof(tmp));
	}

	/* see quickselect method */
	struct hsaNode* find_median(struct hsaNode *start, struct hsaNode *end, int idx)
	{
		if (end <= start) return NULL;
		if (end == start + 1)
			return start;

		struct hsaNode *p, *store, *md = start + (end - start) / 2;
		float pivot;
		while (1) {
			pivot = md->des[idx];

			swap(md, end - 1);
			for (store = p = start; p < end; p++) {
				if (p->des[idx] < pivot) {
					if (p != store)
						swap(p, store);
					store++;
				}
			}
			swap(store, end - 1);

			/* median has duplicate values */
			if (store->des[idx] == md->des[idx])
				return md;

			if (store > md) end = store;
			else        start = store;
		}
	}

	struct hsaNode* make_tree(struct hsaNode *t, int len, int i, int dim)
	{
		struct hsaNode *n;

		if (!len) return 0;

		if ((n = find_median(t, t + len, i))) {
			i = (i + 1) % dim;
			n->left = make_tree(t, n - t, i, dim);
			n->right = make_tree(n + 1, t + len - (n + 1), i, dim);
		}
		return n;
	}

	double dist(struct hsaNode *a, struct hsaNode *b, int dim)
	{
		float t, d = 0;
		while (dim--) {
			t = a->des[dim] - b->des[dim];
			d += t * t;
		}
		return d;
	}

	void nearest(struct hsaNode *root, struct hsaNode *nd, int i, int dim,
	struct hsaNode **best, double *best_dist)
	{
		float d, dx, dx2;

		if (!root) return;
		d = dist(root, nd, dim);
		dx = root->des[i] - nd->des[i];
		dx2 = dx * dx;

		//visited++;

		if (!*best || d < *best_dist) {
			*best_dist = d;
			*best = root;
		}

		/* if chance of exact match is high */
		if (!*best_dist) return;

		if (++i >= dim) i = 0;

		nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
		if (dx2 >= *best_dist) return;
		nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);
	}



};

#endif