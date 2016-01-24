#include "OCL2KDTree.h"
int OCL2KDTree::initialize()
{
	// Call base class Initialize to get default configuration
	if (sampleArgs->initialize() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}
	return SDK_SUCCESS;
}

int OCL2KDTree::createTree(int len, int i, int dim)
{
	cl_int   status;
	/* reserve svm space for CPU update */
	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_WRITE_INVALIDATE_REGION,
		svmTreeBuf,
		len * sizeof(hsaNode),
		0,
		NULL,
		NULL);

	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmTreeBuf) failed.");

	struct hsaNode *t = (hsaNode *)svmTreeBuf;
	root = make_tree(t, len, i, dim);
	status = clEnqueueSVMUnmap(commandQueue,
		svmTreeBuf,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmTreeBuf) failed.");

	return SDK_SUCCESS;
}

int OCL2KDTree::gpuFindNearest(vector<KeyPoint> keypoints2, Mat descriptors2){
	
	size_t globalThreads = keypoints2.size();

	// Set appropriate arguments to the kernel
	int status = clSetKernelArgSVMPointer(sample_kernel,
		0,
		(void *)(svmTreeBuf));
	CHECK_OPENCL_ERROR(status, "clSetKernelArgSVMPointer(svmTreeBuf) failed.");

	status = clSetKernelArgSVMPointer(sample_kernel,
		1,
		(void *)(svmSearchBuf));
	CHECK_OPENCL_ERROR(status, "clSetKernelArgSVMPointer(svmSearchBuf) failed.");
	int i = 0;
	status = clSetKernelArg(sample_kernel,
		2,
		sizeof(int), &i);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed.");

	status = clSetKernelArg(sample_kernel,
		3,
		sizeof(int), &descriptors2.cols);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed.");

	status = clSetKernelArgSVMPointer(sample_kernel,
		4,
		(void *)(svmFound));
	CHECK_OPENCL_ERROR(status, "clSetKernelArgSVMPointer(svmFound) failed.");

/*	status = clSetKernelArg(sample_kernel,
		4,
		sizeof(cl_mem), &clmFound);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed.");*/

	status = clSetKernelArgSVMPointer(sample_kernel,
		5,
		(double *)(svmBestDist));
	CHECK_OPENCL_ERROR(status, "clSetKernelArgSVMPointer(svmBestDist) failed.");


	// Enqueue a kernel run call
	cl_event ndrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		sample_kernel,
		1,
		NULL,
		&globalThreads,
		NULL,
		0,
		NULL,
		&ndrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");




	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_READ,
		svmFound,
		keypoints2.size() * sizeof(int),
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmFound) failed.");

	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_READ,
		svmBestDist,
		keypoints2.size() * sizeof(double),
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmSearchBuf) failed.");

	int *found = (int *)svmFound;
	double *bestDist = (double *)svmBestDist;
	for (int i = 0; i < keypoints2.size(); i++) {
		fprintf(fsvmkdmatch, "%d\t%d\t%.5f\n", found[i], i, bestDist[i]);
	}
	status = clEnqueueSVMUnmap(commandQueue,
		svmFound,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmTreeBuf) failed.");


	status = clEnqueueSVMUnmap(commandQueue,
		svmBestDist,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmBestDist) failed.");




	status = clFlush(commandQueue);
	CHECK_OPENCL_ERROR(status, "clFlush failed.(commandQueue)");

	status = waitForEventAndRelease(&ndrEvt);
	CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(ndrEvt) Failed");

	return SDK_SUCCESS;
}


int OCL2KDTree::cpuFindNearest(vector<KeyPoint> keypoints2, Mat descriptors2)
{
	int        status = SDK_SUCCESS;

	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_WRITE_INVALIDATE_REGION,
		svmSearchBuf,
		keypoints2.size() * sizeof(hsaNode),
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmSearchBuf) failed.");
	
	struct hsaNode *found;
	double best_dist;
	hsaNode* testNode = (hsaNode *)svmSearchBuf;
	//testNode[0].x = 5.21f;
	for (int i = 0; i < keypoints2.size(); i++) {
		found = 0;
		nearest(root, &testNode[i], 0, descriptors2.cols, &found, &best_dist);
		fprintf(fsvmkdmatch, "%d\t%d\t%.5f\n", found->index, testNode[i].index, best_dist);
	}
	status = clEnqueueSVMUnmap(commandQueue,
		svmSearchBuf,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmTreeBuf) failed.");

	return SDK_SUCCESS;
}


int OCL2KDTree::dataMarshalling(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat descriptors1, Mat descriptors2)
{
	cl_int   status;
	// initialize any device/SVM memory here.
	svmTreeBuf = clSVMAlloc(context,
		CL_MEM_READ_WRITE,
		keypoints1.size() * sizeof(hsaNode),
		0);

	if (NULL == svmTreeBuf)
		retValue = SDK_FAILURE;

	CHECK_ERROR(retValue, SDK_SUCCESS, "clSVMAlloc(svmTreeBuf) failed.");

	svmSearchBuf = clSVMAlloc(context,
		CL_MEM_READ_WRITE,
		keypoints2.size() * sizeof(hsaNode),
		0);

	if (NULL == svmSearchBuf)
		retValue = SDK_FAILURE;

	CHECK_ERROR(retValue, SDK_SUCCESS, "clSVMAlloc(svmSearchBuf) failed.");



	svmFound = clSVMAlloc(context,
		CL_MEM_READ_WRITE,
		keypoints2.size() * sizeof(int),
		0);

	if (NULL == svmFound)
		retValue = SDK_FAILURE;

	CHECK_ERROR(retValue, SDK_SUCCESS, "clSVMAlloc(svmFound) failed.");

	//svmFound = malloc(sizeof(int)* keypoints2.size());
	//clmFound = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)* keypoints2.size(), svmFound, NULL);


	svmBestDist = clSVMAlloc(context,
		CL_MEM_READ_WRITE,
		keypoints2.size() * sizeof(double),
		0);

	if (NULL == svmBestDist)
		retValue = SDK_FAILURE;

	CHECK_ERROR(retValue, SDK_SUCCESS, "clSVMAlloc(svmFound) failed.");


	/* reserve svm space for CPU update */
	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_WRITE_INVALIDATE_REGION,
		svmTreeBuf,
		keypoints1.size() * sizeof(hsaNode),
		0,
		NULL,
		NULL);

	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmTreeBuf) failed.");

	hsaNode* hsaKdTree = (hsaNode *)svmTreeBuf;

	//srand(time(0));
	for (int i = 0; i < descriptors1.rows; i++)
	{
		hsaKdTree[i].index = i;
		for (int j = 0; j < descriptors1.cols; j++)
		{
			hsaKdTree[i].des[j] = descriptors1.at<float>(i, j);
			//cout << featureTree[i].des[j] << endl;
		}
		hsaKdTree[i].x = keypoints1[i].pt.x;
		//cout << hsaKdTree[i].x << endl;
		hsaKdTree[i].y = keypoints1[i].pt.y;
		//cout << featureTree[i].y << endl;

	}
	status = clEnqueueSVMUnmap(commandQueue,
		svmTreeBuf,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmTreeBuf) failed.");


	/* reserve svm space for CPU update */
	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_WRITE_INVALIDATE_REGION,
		svmSearchBuf,
		keypoints2.size() * sizeof(hsaNode),
		0,
		NULL,
		NULL);

	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmTreeBuf) failed.");

	hsaNode* hsaTestNode = (hsaNode *)svmSearchBuf;
	for (int i = 0; i < descriptors2.rows; i++)
	{
		hsaTestNode[i].index = i;
		for (int j = 0; j < descriptors2.cols; j++)
		{
			hsaTestNode[i].des[j] = descriptors2.at<float>(i, j);
			//cout << hsaTestNode[i].des[j] << endl;
		}
		hsaTestNode[i].x = keypoints2[i].pt.x;
		hsaTestNode[i].y = keypoints2[i].pt.y;
		
	}

	status = clEnqueueSVMUnmap(commandQueue,
		svmSearchBuf,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmTreeBuf) failed.");


	/* reserve svm space for CPU update */
	status = clEnqueueSVMMap(commandQueue,
		CL_TRUE, //blocking call
		CL_MAP_WRITE_INVALIDATE_REGION,
		svmFound,
		keypoints2.size() * sizeof(int),
		0,
		NULL,
		NULL);

	CHECK_OPENCL_ERROR(status, "clEnqueueSVMMap(svmFound) failed.");
	int* found = (int *)svmFound;
	for (int i = 0; i < keypoints2.size(); i++)
		found[i] = -1;
	status = clEnqueueSVMUnmap(commandQueue,
		svmFound,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmFound) failed.");



	return SDK_SUCCESS;
}


int OCL2KDTree::setupCL(){
	cl_int status = 0;
	cl_device_type dType = CL_DEVICE_TYPE_GPU;

	// Get platform
	cl_platform_id platform = NULL;
	retValue = getPlatform(platform, sampleArgs->platformId,
		sampleArgs->isPlatformEnabled());
	CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

	// Display available devices.
	retValue = displayDevices(platform, dType);
	CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");

	// If we could find our platform, use it. Otherwise use just available 
	// platform.
	cl_context_properties cps[3] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};

	context = clCreateContextFromType(
		cps,
		dType,
		NULL,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateContextFromType failed.");

	status = getDevices(context, &devices, sampleArgs->deviceId,
		sampleArgs->isDeviceIdEnabled());
	CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

	//Set device info of given cl_device_id
	status = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
	CHECK_ERROR(status, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");


	int majorRev, minorRev;
	if (sscanf(deviceInfo.deviceVersion, "OpenCL %d.%d", &majorRev, &minorRev) == 2)
	{
		if (majorRev < 2) {
			OPENCL_EXPECTED_ERROR("Unsupported device! Required CL_DEVICE_OPENCL_C_VERSION 2.0 or higher");
		}
	}

	// Create command queue
	cl_queue_properties prop[] = { 0 };
	commandQueue = clCreateCommandQueueWithProperties(
		context,
		devices[sampleArgs->deviceId],
		prop,
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");

	// create a CL program using the kernel source
	buildProgramData buildData;
	buildData.kernelName = std::string("SVMKDTreeSearch_Kernels.cl");
	buildData.devices = devices;
	buildData.deviceId = sampleArgs->deviceId;
	buildData.flagsStr = std::string("");

	if (sampleArgs->isLoadBinaryEnabled())
	{
		buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
	}

	if (sampleArgs->isComplierFlagsSpecified())
	{
		buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
	}

	retValue = buildOpenCLProgram(program, context, buildData);
	CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

	// get a kernel object handle for a kernel with the given name
	sample_kernel = clCreateKernel(program, "nearest_kernel", &status);
	CHECK_OPENCL_ERROR(status, "clCreateKernel::sample_kernel failed.");



	return SDK_SUCCESS;
}