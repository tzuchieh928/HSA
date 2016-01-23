
#include "OCL2KDTree.h"
#include "KDtree.h"


int main(int argc, const char* argv[])
{
	const Mat img1 = imread("5.jpg", 0); //Load as grayscale
	const Mat img2 = imread("6.jpg", 0);

	FILE *fkp1, *fkp2, *fbfmatch,*fkdmatch;
	fkp1 = fopen("keypoint1.txt", "w");
	fkp2 = fopen("keypoint2.txt", "w");
	fbfmatch = fopen("bfmatch.txt", "w");
	fkdmatch = fopen("kdmatch.txt", "w");

	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	vector<KeyPoint> keypoints1, keypoints2;
	f2d->detect(img1, keypoints1);
	f2d->detect(img2, keypoints2);


	//-- Step 2: Calculate descriptors (feature vectors)    
	
	
	//Mat descriptors1(keypoints1.size(), 2, CV_32F), descriptors2(keypoints2.size(), 2, CV_32F);
	Mat descriptors1, descriptors2;
	f2d->compute(img1, keypoints1, descriptors1);
	f2d->compute(img2, keypoints2, descriptors2);

	/*for (int i = 0; i < keypoints1.size(); i++){
		descriptors1.at<float>(i, 0) = keypoints1[i].pt.x;
		descriptors1.at<float>(i, 1) = keypoints1[i].pt.y;
	}

	for (int i = 0; i < keypoints2.size(); i++){
		descriptors2.at<float>(i, 0) = keypoints2[i].pt.x;
		descriptors2.at<float>(i, 1) = keypoints2[i].pt.y;
	}*/

	cout << "# of keypoints1: " << keypoints1.size() << endl;
	cout << "# of keypoints2: " << keypoints2.size() << endl;
	// Add results to image and save.
	
	for (int i = 0; i < keypoints1.size(); i++){
		fprintf(fkp1, "%.2f\t%.2f\n", keypoints1[i].pt.x, keypoints1[i].pt.y);
	}

	for (int i = 0; i < keypoints2.size(); i++){
		fprintf(fkp2, "%.2f\t%.2f\n", keypoints2[i].pt.x, keypoints2[i].pt.y);
	}

	cout << "des1 cols: " << descriptors1.cols << endl;
	cout << "des1 rows: " << descriptors1.rows << endl;
	cout << "des2 cols: " << descriptors2.cols << endl;
	cout << "des2 rows: " << descriptors2.rows << endl;
	
	

	
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	BFMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors1, descriptors2, matches);
	Mat output;
	//drawKeypoints(img1, keypoints1, output);
	drawMatches(img1, keypoints1, img2, keypoints2, matches, output);

	cout << "# of matches: " << matches.size() << endl;

	for (int i = 0; i < keypoints2.size(); i++) {
		fprintf(fbfmatch, "%d\t%d\t%.5f\n", matches[i].trainIdx, matches[i].queryIdx, matches[i].distance);
		/*fprintf(fbfmatch, "%d\t%d\t%.5f\t\t%.2f\t%.2f\t%.2f\t%.2f\n", matches[i].trainIdx, matches[i].queryIdx, matches[i].distance, keypoints1[matches[i].trainIdx].pt.x, keypoints1[matches[i].trainIdx].pt.y, 
																											keypoints2[matches[i].queryIdx].pt.x, keypoints2[matches[i].queryIdx].pt.y);*/
	/*	for (int j = 0; j < descriptors1.cols; j++)
			fprintf(fbfmatch, "%.2f\t", descriptors1.at<float>(matches[i].trainIdx, j));
		fprintf(fbfmatch, "\n");
		for (int j = 0; j < descriptors1.cols; j++)
			fprintf(fbfmatch, "%.2f\t", descriptors2.at<float>(matches[i].queryIdx, j));
		fprintf(fbfmatch, "\n");*/
	}
		/*
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	*/
	imwrite("sift_result.bmp", output);
	//waitKey(0);

	KDTree kt;

	
	struct kd_node_t *testNode, *featureTree;
	struct kd_node_t *root, *found;
	double best_dist;

	featureTree = (struct kd_node_t*) calloc(keypoints1.size(), sizeof(struct kd_node_t));
	testNode = (struct kd_node_t*) calloc(keypoints2.size(), sizeof(struct kd_node_t));

	//srand(time(0));
	for (int i = 0; i < descriptors1.rows; i++)
	{
		featureTree[i].index = i;
		for (int j = 0; j < descriptors1.cols; j++)
		{
			featureTree[i].des[j] = descriptors1.at<float>(i, j);
			//cout << featureTree[i].des[j] << endl;
		}
		featureTree[i].x = keypoints1[i].pt.x;
		//cout << featureTree[i].x << endl;
		featureTree[i].y = keypoints1[i].pt.y;
		//cout << featureTree[i].y << endl;

	}
	for (int i = 0; i < descriptors2.rows; i++)
	{
		testNode[i].index = i;
		for (int j = 0; j < descriptors2.cols; j++)
		{
			testNode[i].des[j] = descriptors2.at<float>(i, j);
		}
		testNode[i].x = keypoints2[i].pt.x;
		testNode[i].y = keypoints2[i].pt.y;
	}
	root = kt.make_tree(featureTree, descriptors1.rows, 0, descriptors1.cols);

	int sum = 0;
	for (int i = 0; i < keypoints2.size(); i++) {
		found = 0;
		visited = 0;
		kt.nearest(root, &testNode[i], 0, descriptors2.cols, &found, &best_dist);
		fprintf(fkdmatch, "%d\t%d\t%.5f\n", found->index, testNode[i].index, best_dist);
		//fprintf(fkdmatch, "%d\t%d\t%.5f\t\t%.2f\t%.2f\t%.2f\t%.2f\n", found->index, testNode[i].index, best_dist, featureTree[found->index].x, featureTree[found->index].y, testNode[i].x, testNode[i].y);
	/*	for (int j = 0; j < descriptors1.cols; j++)
			fprintf(fkdmatch, "%.2f\t", featureTree[found->index].des[j]);
		fprintf(fkdmatch, "\n");
		for (int j = 0; j < descriptors2.cols; j++)
			fprintf(fkdmatch, "%.2f\t", testNode[i].des[j]);
		fprintf(fkdmatch, "\n");
		*/
		//cout << "1:	" << found << endl;
		//cout << "2:	" << &featureTree[found->index] << endl;
		sum += visited;
	}
	printf(	"visited %d nodes for %d random findings (%f per lookup)\n",
		sum, keypoints2.size(), sum / (double)keypoints2.size());


	free(featureTree);
	free(testNode);
	fclose(fkp1);
	fclose(fkp2);
	fclose(fkdmatch);
	fclose(fbfmatch);


	OCL2KDTree ocl2kdtree;
	// Initialize
	if (ocl2kdtree.initialize() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	/*
	if (ocl2kdtree.sampleArgs->isDumpBinaryEnabled())
	{
		//GenBinaryImage
		return ocl2kdtree.genBinaryImage();
	}
	*/
	int status = ocl2kdtree.setupCL();
	if (status != SDK_SUCCESS)
	{
		return status;
	}
	

	ocl2kdtree.dataMarshalling(keypoints1, keypoints2, descriptors1, descriptors2);
	ocl2kdtree.createTree(descriptors1.rows, 0, descriptors1.cols);
	ocl2kdtree.findNearest(keypoints2, descriptors2);























	
	system("PAUSE");
	return 0;
}