/* binary tree node definition */
#define MAX_DIM 128
typedef struct nodeStruct
{
	float des[MAX_DIM];
	int index;
	float x;
	float y;
  __global struct nodeStruct* left;
  __global struct nodeStruct* right;
} node;

double dist(node *a, node *b, int dim)
{
		float t, d = 0;
		while (dim--) {
			t = a->des[dim] - b->des[dim];
			d += t * t;
		}
		return d;
}

/***
 * sample_kernel:
 ***/
__kernel void nearest_kernel(__global void *root, __global void *nd, int i, int dim,
	__global int *best, __global double *best_dist)
{
		float d, dx, dx2;
		node *kdtreeroot = (node *)root;
		node *testNode = (node *)nd;
		int index = get_global_id(0);
		//best[index] = index;

		if (!kdtreeroot) return;
		d = dist(kdtreeroot, &testNode[index], dim);
		dx = kdtreeroot->des[i] - testNode[index].des[i];
		dx2 = dx * dx;

		//visited++;

		if (best[index] == -1 || d < best_dist[index]) {
			best_dist[index] = d;
			best[index] = kdtreeroot->index;
		}

		// if chance of exact match is high 
		if (!best_dist[index]) return;

		if (++i >= dim) i = 0;

		nearest_kernel(dx > 0 ? kdtreeroot->left : kdtreeroot->right, nd, i, dim, best, best_dist);
		if (dx2 >= best_dist[index]) return;
	//	nearest_kernel(dx > 0 ? kdtreeroot->right : kdtreeroot->left, nd, i, dim, best, best_dist);
}