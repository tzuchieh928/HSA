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

int OCL2KDTree::genBinaryImage()
{
	bifData binaryData;
	binaryData.kernelName = std::string("SVMKDTreeSearch_Kernels.cl");
	binaryData.flagsStr = std::string("");
	if (sampleArgs->isComplierFlagsSpecified())
	{
		binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
	}
	binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
	int status = generateBinaryImage(binaryData);
	return status;
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
		cout << keypoints1[i].pt.x << endl;
		hsaKdTree[i].index = i;
		for (int j = 0; j < descriptors1.cols; j++)
		{
			hsaKdTree[i].des[j] = descriptors1.at<float>(i, j);
			//cout << featureTree[i].des[j] << endl;
		}
		hsaKdTree[i].x = keypoints1[i].pt.x;
		cout << hsaKdTree[i].x << endl;
		hsaKdTree[i].y = keypoints1[i].pt.y;
		//cout << featureTree[i].y << endl;

	}
	status = clEnqueueSVMUnmap(commandQueue,
		svmTreeBuf,
		0,
		NULL,
		NULL);
	CHECK_OPENCL_ERROR(status, "clEnqueueSVMUnmap(svmTreeBuf) failed.");
	return SDK_SUCCESS;
}


int OCL2KDTree::setupCL(){
	cl_int status = 0;
	cl_device_type dType = CL_DEVICE_TYPE_GPU;

/*	if (sampleArgs->deviceType.compare("cpu") == 0)
	{
		dType = CL_DEVICE_TYPE_CPU;
	}
	else //deviceType = "gpu"
	{
		dType = CL_DEVICE_TYPE_GPU;
		if (sampleArgs->isThereGPU() == false)
		{
			std::cout << "GPU not found. Falling back to CPU device" << std::endl;
			dType = CL_DEVICE_TYPE_CPU;
		}
	}
	*/
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