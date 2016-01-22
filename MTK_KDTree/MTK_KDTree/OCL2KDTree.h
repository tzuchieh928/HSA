#ifndef _OCL2KDTree_H_
#define _OCL2KDTree_H_
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"

using namespace appsdk;

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
public:
	CLCommandArgs*       sampleArgs;

	OCL2KDTree()
	{
		sampleArgs = new CLCommandArgs();
		sampleTimer = new SDKTimer();
		sampleArgs->sampleVerStr = SAMPLE_VERSION;
		sampleArgs->flags = OCL_COMPILER_FLAGS;
	};


	~OCL2KDTree()
	{
		delete sampleArgs;
		delete sampleTimer;
	};

	int setupCL();

};

#endif