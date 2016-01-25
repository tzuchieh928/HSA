/* binary tree node definition */
#define MAX_DIM 128
#define MAXDEPTH 128
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
		int index = get_global_id(0);
		node *testNode = (node *)nd;
		node *nodeStack[MAXDEPTH];
		int iStack[MAXDEPTH];
		float dx2OfParentStack[MAXDEPTH];
		int top = 0;

		iStack[top] = 0;
		dx2OfParentStack[top] = 0;
		nodeStack[top++] = (node *)root;//push
		while (top>0)
		{
			
			--top;
			node *n = nodeStack[top];//pop
			int iTop = iStack[top];
			float dx2ofParent = dx2OfParentStack[top];
			//check it
			if ( best_dist[index]!=-1 && dx2ofParent >= best_dist[index])
				continue;

			if (!n) continue;

			float d, dx, dx2;

			d = dist(n, &testNode[index], dim);
			dx = n->des[iTop] - testNode[index].des[iTop];
			dx2 = dx*dx;

			if (best[index] == -1 || d < best_dist[index])
			{
				best_dist[index] = d;
				best[index] = n->index;
			}

			if ((best_dist[index]) == 0)
			{
				//store solution
				best[index] = n->index;
				break;
			}

			if (++iTop >= dim) iTop = 0;

			//if pass check
			//push childs
			iStack[top] = iTop;
			dx2OfParentStack[top] = dx2;
			nodeStack[top] = n->left;
			top++;
			

			iStack[top] = iTop;
			dx2OfParentStack[top] = dx2;
			nodeStack[top] = n->right;
			top++;
		}
}
