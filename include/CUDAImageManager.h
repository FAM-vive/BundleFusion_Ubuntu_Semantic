#pragma once
#include "RGBDSensor.h"
#include "CUDAImageUtil.h"
#include "CUDAImageCalibrator.h"
#include "GlobalBundlingState.h"
#include "TimingLog.h"
#include <cuda_runtime.h>
#include "maskrcnn.h"
#include "gSLICrTools.h"


class CUDAImageManager {
public:
	unsigned char* maskrcnnMask;
	float* scores;
	void setmaskrcnnMask(std::vector<std::pair<cv::Mat,BBoxInfo>> &results)
	{
		short resultMasks_Num_cpu = (short)results.size();
		maskrcnnMask =  (unsigned char*)malloc(resultMasks_Num_cpu*sizeof(unsigned char)*320*240);
		 scores = (float*)malloc(resultMasks_Num_cpu*sizeof(float));
		for(int i =0 ;i<resultMasks_Num_cpu;i++)
		{
			memcpy(maskrcnnMask+i*320*240, results[i].first.data,sizeof(unsigned char)*320*240);
			memcpy(scores+i, &(results[i].second.prob), sizeof(float));
		}
	}

	class ManagedRGBDInputFrame {
	public:
		friend class CUDAImageManager;

		static void globalInit(unsigned int width, unsigned int height, bool isOnGPU)
		{
			globalFree();

			s_width = width;
			s_height = height;
			s_bIsOnGPU = isOnGPU;

			if (!s_bIsOnGPU) {
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_depthIntegrationGlobal, sizeof(float)*width*height));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_colorIntegrationGlobal, sizeof(uchar4)*width*height));
			}
			else {
				s_depthIntegrationGlobal = new float[width*height];
				s_colorIntegrationGlobal = new uchar4[width*height];
			}
		}
		static void globalFree()
		{
			if (!s_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(s_depthIntegrationGlobal);
				MLIB_CUDA_SAFE_FREE(s_colorIntegrationGlobal);
			}
			else {
				SAFE_DELETE_ARRAY(s_depthIntegrationGlobal);
				SAFE_DELETE_ARRAY(s_colorIntegrationGlobal);
			}
		}


		void alloc() {
			if (s_bIsOnGPU) {
				printf("something\n");
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_depthIntegration, sizeof(float)*s_width*s_height));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_colorIntegration, sizeof(uchar4)*s_width*s_height));
			}
			else {
				m_depthIntegration = new float[s_width*s_height];
				m_colorIntegration = new uchar4[s_width*s_height];
			}
		}


		void free() {
			if (s_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(m_depthIntegration);
				MLIB_CUDA_SAFE_FREE(m_colorIntegration);


			}
			else {
				SAFE_DELETE_ARRAY(m_depthIntegration);
				SAFE_DELETE_ARRAY(m_colorIntegration);
			}

			// SAFE_DELETE_ARRAY(resultMasks_cpu);
			// SAFE_DELETE_ARRAY(resultMasks_Num_cpu);
			// SAFE_DELETE_ARRAY(scores_cpu);
			MLIB_CUDA_SAFE_FREE(resultMasks_Num_gpu);
			MLIB_CUDA_SAFE_FREE(resultMasks_gpu);
			MLIB_CUDA_SAFE_FREE(scores_gpu);
		}


		const float* getDepthFrameGPU() {	//be aware that only one depth frame is globally valid at a time
			if (s_bIsOnGPU) {
				return m_depthIntegration;
			}
			else {
				if (this != s_activeDepthGPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*s_width*s_height, cudaMemcpyHostToDevice));
					s_activeDepthGPU = this;
				}
				return s_depthIntegrationGlobal;
			}
		}
		const uchar4* getColorFrameGPU() {	//be aware that only one depth frame is globally valid at a time
			if (s_bIsOnGPU) {
				return m_colorIntegration;
			}
			else {
				if (this != s_activeColorGPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*s_width*s_height, cudaMemcpyHostToDevice));
					s_activeColorGPU = this;
				}
				return s_colorIntegrationGlobal;
			}
		}

		const float* getDepthFrameCPU() {
			if (s_bIsOnGPU) {
				if (this != s_activeDepthCPU) {
					//std::cout<<"getDepthFrameCPU s_activeDepthGPU Model. \n\n"<<std::endl;
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*s_width*s_height, cudaMemcpyDeviceToHost));
					s_activeColorCPU = this;
				}
				//std::cout<<"getDepthFrameCPU s_activeDepthCPU Model. \n\n\n"<<std::endl;
				return s_depthIntegrationGlobal;
			}
			else {
			  //std::cout<<"getDepthFrameCPU Not s_bIsOnGPU Model. \n\n\n"<<std::endl;
				return m_depthIntegration;
			}
		}
		const uchar4* getColorFrameCPU() {
			if (s_bIsOnGPU) {
				if (this != s_activeColorCPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*s_width*s_height, cudaMemcpyDeviceToHost));
					s_activeDepthCPU = this;
				}
				return s_colorIntegrationGlobal;
			}
			else {
				return m_colorIntegration;
			}
		}

		//-----------------YJ-----------------------------
		// 获取当前帧RGB在GPU上全部的mask
		const unsigned char* getMaskList(){
			return resultMasks_gpu;
		}

		const short* getMasks_Num(){
			return resultMasks_Num_gpu;
		}

		const float* getScores(){
			return scores_gpu;
		}

		// void setScores(const float* scores_){
		// 	scores = scores_;
		// }

		// void setmask(std::vector<std::pair<cv::Mat,BBoxInfo>> &results)
		// {
		// 	short resultMasks_Num_cpu = (short)results.size();
		// 	unsigned char* resultMasks_cpu =  (unsigned char*)malloc(resultMasks_Num_cpu*sizeof(unsigned char)*320*240);
		// 	float* scores_cpu = (float*)malloc(resultMasks_Num_cpu*sizeof(float));

		// 	for(int i =0 ;i<resultMasks_Num_cpu;i++)
		// 	{
		// 		memcpy(resultMasks_cpu+i*320*240, results[i].first.data,sizeof(unsigned char)*320*240);
		// 		memcpy(scores_cpu+i, &(results[i].second.prob), sizeof(float));
		// 	}
		// 	cudaMalloc(&resultMasks_Num_gpu, sizeof(short));
		// 	cudaMemcpy(resultMasks_Num_gpu, &resultMasks_Num_cpu,sizeof(short), cudaMemcpyHostToDevice);
		
		// 	if(resultMasks_Num_cpu!=0)
		// 	{
		// 		cudaMalloc(&resultMasks_gpu, sizeof(unsigned char)*320*240*resultMasks_Num_cpu);
		// 		cudaMemcpy(resultMasks_gpu,resultMasks_cpu,sizeof(unsigned char)*320*240*resultMasks_Num_cpu,cudaMemcpyHostToDevice);

		// 		cudaMalloc(&scores_gpu, sizeof(unsigned char)*320*240*resultMasks_Num_cpu);
		// 		cudaMemcpy(scores_gpu,scores_cpu,sizeof(float)*resultMasks_Num_cpu,cudaMemcpyHostToDevice);
		// 	}
			
		// }

		void setmasknew(short resultMasks_Num_cpu,  unsigned char* resultMasks_cpu, float* scores_cpu)
		{
			cudaMalloc(&resultMasks_Num_gpu, sizeof(short));
			cudaMemcpy(resultMasks_Num_gpu, &resultMasks_Num_cpu,sizeof(short), cudaMemcpyHostToDevice);

			cudaMalloc(&resultMasks_gpu, sizeof(unsigned char)*320*240*resultMasks_Num_cpu);
			cudaMemcpy(resultMasks_gpu,resultMasks_cpu,sizeof(unsigned char)*320*240*resultMasks_Num_cpu,cudaMemcpyHostToDevice);

			cudaMalloc(&scores_gpu, sizeof(unsigned char)*320*240*resultMasks_Num_cpu);
			cudaMemcpy(scores_gpu,scores_cpu,sizeof(float)*resultMasks_Num_cpu,cudaMemcpyHostToDevice);
		}


	private:
		float*	m_depthIntegration;	//either on the GPU or CPU
		uchar4*	m_colorIntegration;	//either on the GPU or CPU

		unsigned char* resultMasks_gpu;  //当前帧rgb的在GPU上的全部mask
		short*  resultMasks_Num_gpu; //一帧rgb中实例mask的个数
		float* scores_gpu;  //当前rgb的每个mask的概率

		// unsigned char* resultMasks_gpu;  //当前帧rgb的在GPU上的全部mask
		// short*  resultMasks_Num_gpu; //一帧rgb中实例mask的个数
		// float* scores_gpu;  //当前rgb的每个mask的概率

		static bool			s_bIsOnGPU;
		static unsigned int s_width;
		static unsigned int s_height;

		static float*		s_depthIntegrationGlobal;
		static uchar4*		s_colorIntegrationGlobal;
		static ManagedRGBDInputFrame*	s_activeColorGPU;
		static ManagedRGBDInputFrame*	s_activeDepthGPU;

		static float*		s_depthIntegrationGlobalCPU;
		static uchar4*		s_colorIntegrationGlobalCPU;
		static ManagedRGBDInputFrame*	s_activeColorCPU;
		static ManagedRGBDInputFrame*	s_activeDepthCPU;
	};

	CUDAImageManager(unsigned int widthIntegration, unsigned int heightIntegration, unsigned int widthSIFT, unsigned int heightSIFT, RGBDSensor* sensor, bool storeFramesOnGPU = false) {
		m_RGBDSensor = sensor;

		m_widthSIFTdepth = sensor->getDepthWidth();
		m_heightSIFTdepth = sensor->getDepthHeight();
		m_SIFTdepthIntrinsics = sensor->getDepthIntrinsics();
		m_widthIntegration = widthIntegration;
		m_heightIntegration = heightIntegration;

		const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
		const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputRaw, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputFiltered, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorInput, sizeof(uchar4)*bufferDimColorInput));

		m_currFrame = 0;

		const unsigned int rgbdSensorWidthDepth = m_RGBDSensor->getDepthWidth();
		const unsigned int rgbdSensorHeightDepth = m_RGBDSensor->getDepthHeight();

		// adapt intrinsics
		m_depthIntrinsics = m_RGBDSensor->getDepthIntrinsics();
		m_depthIntrinsics._m00 *= (float)m_widthIntegration / (float)rgbdSensorWidthDepth;  //focal 
		m_depthIntrinsics._m11 *= (float)m_heightIntegration/ (float)rgbdSensorHeightDepth; 
		m_depthIntrinsics._m02 *= (float)(m_widthIntegration-1) / (float)(rgbdSensorWidthDepth-1);	//principal point
		m_depthIntrinsics._m12 *= (float)(m_heightIntegration-1) / (float)(rgbdSensorHeightDepth-1);
		m_depthIntrinsicsInv = m_depthIntrinsics.getInverse();

		const unsigned int rgbdSensorWidthColor = m_RGBDSensor->getColorWidth();
		const unsigned int rgbdSensorHeightColor = m_RGBDSensor->getColorHeight();

		m_colorIntrinsics = m_RGBDSensor->getColorIntrinsics();
		m_colorIntrinsics._m00 *= (float)m_widthIntegration / (float)rgbdSensorWidthColor;  //focal 
		m_colorIntrinsics._m11 *= (float)m_heightIntegration/ (float)rgbdSensorHeightColor; 
		m_colorIntrinsics._m02 *=  (float)(m_widthIntegration-1) / (float)(rgbdSensorWidthColor-1);	//principal point
		m_colorIntrinsics._m12 *= (float)(m_heightIntegration-1) / (float)(rgbdSensorHeightColor-1);
		m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();

		// adapt extrinsics
		m_depthExtrinsics = m_RGBDSensor->getDepthExtrinsics();
		m_depthExtrinsicsInv = m_RGBDSensor->getDepthExtrinsicsInv();

		if (GlobalAppState::get().s_bUseCameraCalibration) {
			m_SIFTdepthIntrinsics = m_RGBDSensor->getColorIntrinsics();
			m_SIFTdepthIntrinsics._m00 *= (float)m_widthSIFTdepth / (float)rgbdSensorWidthColor;  
			m_SIFTdepthIntrinsics._m11 *= (float)m_heightSIFTdepth / (float)rgbdSensorHeightColor; 
			m_SIFTdepthIntrinsics._m02 *= (float)(m_widthSIFTdepth-1) / (float)(m_RGBDSensor->getColorWidth()-1);
			m_SIFTdepthIntrinsics._m12 *= (float)(m_heightSIFTdepth-1) / (float)(m_RGBDSensor->getColorHeight()-1);
		}

		ManagedRGBDInputFrame::globalInit(getIntegrationWidth(), getIntegrationHeight(), storeFramesOnGPU);
		m_bHasBundlingFrameRdy = false;

		maskrcnnParams.dataDirs.push_back("data/maskrcnn");
    	maskrcnnParams.inputTensorNames.push_back(MaskRCNNConfig::MODEL_INPUT);
    	maskrcnnParams.batchSize = 1;
    	maskrcnnParams.outputTensorNames.push_back(MaskRCNNConfig::MODEL_OUTPUTS[0]);
    	maskrcnnParams.outputTensorNames.push_back(MaskRCNNConfig::MODEL_OUTPUTS[1]);
    	maskrcnnParams.dlaCore = -1;
    	maskrcnnParams.int8 = false;
    	maskrcnnParams.fp16 = true;
    	maskrcnnParams.uffFileName = MaskRCNNConfig::MODEL_NAME;
    	maskrcnnParams.maskThreshold = MaskRCNNConfig::MASK_THRESHOLD;

		//初始化maskrcnn
		maskrcnn.init(maskrcnnParams);
		maskrcnn.build(engine_path,true);
		//maskrcnn.build(engine_path,false);

		slicTools.initCameraIntrinsics(m_depthIntrinsics(0, 2),m_depthIntrinsics(1, 2),m_depthIntrinsics(0, 0),m_depthIntrinsics(1, 1));

	}

	~CUDAImageManager() {
		reset();

		MLIB_CUDA_SAFE_FREE(d_depthInputRaw);
		MLIB_CUDA_SAFE_FREE(d_depthInputFiltered);
		MLIB_CUDA_SAFE_FREE(d_colorInput);
		ManagedRGBDInputFrame::globalFree();
		maskrcnn.teardown();
	}

	void reset() {
		for (auto& f : m_data) {
			f.free();
		}
		m_data.clear();
	}

	bool process(cv::Mat& rgb, cv::Mat& depth);

	void copyToBundling(float* d_depthRaw, float* d_depthFilt, uchar4* d_color) const {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthRaw, d_depthInputRaw, sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthFilt, d_depthInputFiltered, sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_color, d_colorInput, sizeof(uchar4)*m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight(), cudaMemcpyDeviceToDevice));
	}

	void maskSuperPixelFilter_OverSeg(int spNum, int *finalSPixel, int resultMasks_Num);
	//TODO not const because direct assignment in SiftGPU
	//float* getIntensityImageSIFT() {
	//	return d_intensitySIFT;
	//}

	ManagedRGBDInputFrame& getLastIntegrateFrame() {
		return m_data.back();
	}

	ManagedRGBDInputFrame& getIntegrateFrame(unsigned int frame) {
		return m_data[frame];
	}

	// called after process
	unsigned int getCurrFrameNumber() const {
		MLIB_ASSERT(m_currFrame > 0);
		return m_currFrame - 1;
	}

	unsigned int getIntegrationWidth() const {
		return m_widthIntegration;
	}
	unsigned int getIntegrationHeight() const {
		return m_heightIntegration;
	}



	const mat4f& getDepthIntrinsics() const	{
		return m_depthIntrinsics;
	}

	const mat4f& getDepthIntrinsicsInv() const {
		return m_depthIntrinsicsInv;
	}

	//const mat4f& getColorIntrinsics() const	{
	//	return m_colorIntrinsics;
	//}

	//const mat4f& getColorIntrinsicsInv() const {
	//	return m_colorIntrinsicsInv;
	//}

	const mat4f& getDepthExtrinsics() const	{
		return m_depthExtrinsics;
	}

	const mat4f& getDepthExtrinsicsInv() const {
		return m_depthExtrinsicsInv;
	}

	const unsigned int getSIFTDepthWidth() const {
		return m_widthSIFTdepth;
	}
	const unsigned int getSIFTDepthHeight() const {
		return m_heightSIFTdepth;
	}
	const mat4f& getSIFTDepthIntrinsics() const	{
		return m_SIFTdepthIntrinsics;
	}


	bool hasBundlingFrameRdy() const {
		return m_bHasBundlingFrameRdy;
	}

	//! must be called by depth sensing to signal bundling that a frame is ready
	void setBundlingFrameRdy() {
		m_bHasBundlingFrameRdy = true;
	}

	//! must be called by bundling to signal depth sensing it can read it a new frame
	void confirmRdyBundlingFrame() {
		m_bHasBundlingFrameRdy = false;
	}



private:
	bool m_bHasBundlingFrameRdy;

	RGBDSensor* m_RGBDSensor;
	CUDAImageCalibrator m_imageCalibrator;

	mat4f m_colorIntrinsics;
	mat4f m_colorIntrinsicsInv;
	mat4f m_depthIntrinsics;
	mat4f m_depthIntrinsicsInv;
	mat4f m_depthExtrinsics;
	mat4f m_depthExtrinsicsInv;

	//! resolution for integration both depth and color data
	unsigned int m_widthIntegration;
	unsigned int m_heightIntegration;
	mat4f m_SIFTdepthIntrinsics;

	//! temporary GPU storage for inputting the current frame
	float*	d_depthInputRaw;
	uchar4*	d_colorInput;
	float*	d_depthInputFiltered;
	unsigned int m_widthSIFTdepth;
	unsigned int m_heightSIFTdepth;
	//! all image data on the GPU
	std::vector<ManagedRGBDInputFrame> m_data;
	unsigned int m_currFrame;
	static Timer s_timer;

	SampleMaskRCNNParams maskrcnnParams;
	SampleMaskRCNN maskrcnn;
	std::string engine_path = "./weight/maskrcnn_fp16_512.bin";	
	gSLICrTools slicTools;
	int finalSPixel[320 * 240];		// 存最终RGBD超像素分割结果
	
};

