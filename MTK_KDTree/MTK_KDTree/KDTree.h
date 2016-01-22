#include <iostream>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define MAX_DIM 128
//#define N 1000000
//#define rand1() (rand() / (double)RAND_MAX)
//#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); v.x[2] = rand1(); }

struct kd_node_t{
	float des[MAX_DIM];
	int index;
	float x;
	float y;
	struct kd_node_t *left, *right;
};
/* global variable, so sue me */
int visited;
class KDTree{

public:
	double dist(struct kd_node_t *a, struct kd_node_t *b, int dim)
	{
		float t, d = 0;
		while (dim--) {
			t = a->des[dim] - b->des[dim];
			d += t * t;
		}
		return d;
	}

	/*void swap(struct kd_node_t *x, struct kd_node_t *y) {
		float tmp[MAX_DIM];
		memcpy(tmp, x->des, sizeof(tmp));
		memcpy(x->des, y->des, sizeof(tmp));
		memcpy(y->des, tmp, sizeof(tmp));
	}
	*/
	void swap(struct kd_node_t *x, struct kd_node_t *y) {
		struct kd_node_t tmp;
		memcpy(&tmp, x, sizeof(tmp));
		memcpy(x, y, sizeof(tmp));
		memcpy(y, &tmp, sizeof(tmp));
	}

	/* see quickselect method */
	struct kd_node_t* find_median(struct kd_node_t *start, struct kd_node_t *end, int idx)
	{
		if (end <= start) return NULL;
		if (end == start + 1)
			return start;

		struct kd_node_t *p, *store, *md = start + (end - start) / 2;
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
	
	struct kd_node_t* make_tree(struct kd_node_t *t, int len, int i, int dim)
	{
		struct kd_node_t *n;

		if (!len) return 0;

		if ((n = find_median(t, t + len, i))) {
			i = (i + 1) % dim;
			n->left = make_tree(t, n - t, i, dim);
			n->right = make_tree(n + 1, t + len - (n + 1), i, dim);
		}
		return n;
	}

	

	void nearest(struct kd_node_t *root, struct kd_node_t *nd, int i, int dim,
	struct kd_node_t **best, double *best_dist)
	{
		float d, dx, dx2;

		if (!root) return;
		d = dist(root, nd, dim);
		dx = root->des[i] - nd->des[i];
		dx2 = dx * dx;

		visited++;

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