
#include "stdafx.h"

#include "CUDAImageManager.h"

bool		CUDAImageManager::ManagedRGBDInputFrame::s_bIsOnGPU = false;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_width = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_height = 0;

float*		CUDAImageManager::ManagedRGBDInputFrame::s_depthIntegrationGlobal = NULL;
uchar4*		CUDAImageManager::ManagedRGBDInputFrame::s_colorIntegrationGlobal = NULL;

CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorGPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthGPU = NULL;

CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorCPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthCPU = NULL;

Timer CUDAImageManager::s_timer;

bool CUDAImageManager::process ( cv::Mat& rgb, cv::Mat& depth )
{
    if ( !m_RGBDSensor->receiveDepthAndColor ( rgb,depth ) )
    {
        return false;
    }
    std::cout<<rgb.size()<<std::endl;
    //semantic start
    //mask 保存在maskrcnn.results
    maskrcnn.infer(rgb);
    // maskrcnnMask = maskrcnn.results;
    // slicTools.
    setmaskrcnnMask(maskrcnn.results);
    if(maskrcnn.getMaskNum()>0)
    {
			cv::Mat img1(240, 320, CV_8UC1);
			memcpy ( img1.data,maskrcnnMask , sizeof (uint8_t ) *320*240 );
            cv::imwrite("before.png",img1*200);
    }
    //slicTools.rgbdSuperPixelSeg(rgb,depth,finalSPixel);
    //maskSuperPixelFilter_OverSeg(1200, finalSPixel, maskrcnn.getMaskNum());
    if(maskrcnn.getMaskNum()>0)
    {
			cv::Mat img1(240, 320, CV_8UC1);
			memcpy ( img1.data,maskrcnnMask , sizeof (uint8_t ) *320*240 );
            cv::imwrite("after.png",img1*200);
    }
    //semantic end

    if ( m_currFrame + 1 > GlobalBundlingState::get().s_maxNumImages * GlobalBundlingState::get().s_submapSize )
    {
        std::cout << "WARNING: reached max #images, truncating sequence  num:" << m_currFrame << std::endl;
        return false;
    }

    if ( GlobalBundlingState::get().s_enableGlobalTimings )
    {
        TimingLog::addLocalFrameTiming();
        cudaDeviceSynchronize();
        s_timer.start();
    }

    m_data.push_back ( ManagedRGBDInputFrame() );
    ManagedRGBDInputFrame& frame = m_data.back();
    frame.alloc();
    //添加mask
    // frame.setmask(maskrcnn.results);
    frame.setmasknew(maskrcnn.getMaskNum(),maskrcnnMask,scores);


    const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth() *m_RGBDSensor->getColorHeight();
    MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( d_colorInput, m_RGBDSensor->getColorRGBX(), sizeof ( uchar4 ) *bufferDimColorInput, cudaMemcpyHostToDevice ) );

    if ( ( m_RGBDSensor->getColorWidth() == m_widthIntegration ) && ( m_RGBDSensor->getColorHeight() == m_heightIntegration ) )
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::copy<uchar4> ( frame.m_colorIntegration, d_colorInput, m_widthIntegration, m_heightIntegration );
            //std::swap(frame.m_colorIntegration, d_colorInput);
        }
        else
        {
            memcpy ( frame.m_colorIntegration, m_RGBDSensor->getColorRGBX(), sizeof ( uchar4 ) *bufferDimColorInput );
        }
    }
    else
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::resampleUCHAR4 ( frame.m_colorIntegration, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight() );
        }
        else
        {
            CUDAImageUtil::resampleUCHAR4 ( frame.s_colorIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight() );
            MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_colorIntegration, frame.s_colorIntegrationGlobal, sizeof ( uchar4 ) *m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost ) );
            frame.s_activeColorGPU = &frame;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // Process Depth
    ////////////////////////////////////////////////////////////////////////////////////

    const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth() *m_RGBDSensor->getDepthHeight();
    MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( d_depthInputRaw, m_RGBDSensor->getDepthFloat(), sizeof ( float ) *m_RGBDSensor->getDepthWidth() * m_RGBDSensor->getDepthHeight(), cudaMemcpyHostToDevice ) );


    if ( GlobalBundlingState::get().s_erodeSIFTdepth )
    {
        unsigned int numIter = 2;
        numIter = 2 * ( ( numIter + 1 ) / 2 );
        for ( unsigned int i = 0; i < numIter; i++ )
        {
            if ( i % 2 == 0 )
            {
                CUDAImageUtil::erodeDepthMap ( d_depthInputFiltered, d_depthInputRaw, 3,
                                               m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), 0.05f, 0.3f );
            }
            else
            {
                CUDAImageUtil::erodeDepthMap ( d_depthInputRaw, d_depthInputFiltered, 3,
                                               m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), 0.05f, 0.3f );
            }
        }
    }
    if ( GlobalBundlingState::get().s_depthFilter ) //smooth
    {
        CUDAImageUtil::gaussFilterDepthMap ( d_depthInputFiltered, d_depthInputRaw, GlobalBundlingState::get().s_depthSigmaD, GlobalBundlingState::get().s_depthSigmaR,
                                             m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight() );
    }
    else
    {
        CUDAImageUtil::copy<float> ( d_depthInputFiltered, d_depthInputRaw, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight() );
    }

    if ( ( m_RGBDSensor->getDepthWidth() == m_widthIntegration ) && ( m_RGBDSensor->getDepthHeight() == m_heightIntegration ) )
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::copy<float> ( frame.m_depthIntegration, d_depthInputFiltered, m_widthIntegration, m_heightIntegration );
            //std::swap(frame.m_depthIntegration, d_depthInput);
        }
        else
        {
            if ( GlobalBundlingState::get().s_erodeSIFTdepth )
            {
                MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_depthIntegration, d_depthInputFiltered, sizeof ( float ) *bufferDimDepthInput, cudaMemcpyDeviceToHost ) );
            }
            else
            {
                memcpy ( frame.m_depthIntegration, m_RGBDSensor->getDepthFloat(), sizeof ( float ) *bufferDimDepthInput );
            }
        }
    }
    else
    {
        if ( ManagedRGBDInputFrame::s_bIsOnGPU )
        {
            CUDAImageUtil::resampleFloat ( frame.m_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInputFiltered, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight() );
        }
        else
        {
            CUDAImageUtil::resampleFloat ( frame.s_depthIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_depthInputFiltered, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight() );
            MLIB_CUDA_SAFE_CALL ( cudaMemcpy ( frame.m_depthIntegration, frame.s_depthIntegrationGlobal, sizeof ( float ) *m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost ) );
            frame.s_activeDepthGPU = &frame;
        }
    }

    if ( GlobalBundlingState::get().s_enableGlobalTimings )
    {
        cudaDeviceSynchronize();
        s_timer.stop();
        TimingLog::getFrameTiming ( true ).timeSensorProcess = s_timer.getElapsedTimeMS();
    }

    m_currFrame++;
    return true;
}


// 用rgbd超像素分割的结果去优化原始的maskrcnn的mask
void CUDAImageManager::maskSuperPixelFilter_OverSeg(int spNum, int *finalSPixel, int resultMasks_Num)
{
	//Step 1 count pixel num。NumMatrix就是用来存每个mask和每个超像素重叠的像素数量，最后的spNum个元素存的是每个超像素标号的像素数
	int* PixelValueOfEachMask = (int*)malloc(resultMasks_Num * sizeof(int)); // 用于存每个mask的标签值
	int* NumMatrix = (int*)malloc((resultMasks_Num + 1)*spNum*sizeof(int));  // (mask的数量+1) * 超像素数量，多个1是用来存每个超像素的像素数量
	memset(NumMatrix, 0, (resultMasks_Num + 1)*spNum*sizeof(int));
	for (int x = 0; x<320; x++)  // 遍历超像素上的每一个像素点
	{
		for (int y = 0; y<240; y++)
		{
			int id = finalSPixel[y*320 + x];  // 像素坐标为(x,y)的超像素标号
			if (id >= spNum || id<0) 
                continue;
			//last row(pointNum of each superPixel) 统计每个超像素的像素数量
			NumMatrix[resultMasks_Num*spNum + id] ++;

			for (int i = 0; i<resultMasks_Num; i++)  // 遍历masks上每一个对应位置上的像素点
			{
				if (maskrcnnMask[i*320*240 + y*320 + x])  // 如果mask对应像素有值
				{
					PixelValueOfEachMask[i] = maskrcnnMask[i*320*240 + y*320 + x];
					//pointNum of each Instance
					NumMatrix[i*spNum + id] ++;  // 第i个mask和第id个超像素的重叠数加1
				}
			}
		}
	}

	// Step 2
	for (int x = 0; x<320; x++)  // 遍历超像素上的每一个像素点
	{
		for (int y = 0; y<240; y++)
		{
			int id = finalSPixel[y*320 + x];
			if (id >= spNum || id<0)  // 若像素点的超像素标号处于正常范围之外，该点的所有mask都直接置为0
			{
                // std::cout<<"error id:"<<id<<std::endl;
				// for (int i = 0; i<resultMasks_Num; i++) 
                //     maskrcnnMask[i*320*240 + y*320 + x] = 0;
				continue;
			}

			for (int i = 0; i<resultMasks_Num; i++)
			{
				int n = NumMatrix[resultMasks_Num*spNum + id];  // 超像素标号为id的超像素的像素个数
				int m = NumMatrix[i*spNum + id];  // 第i个mask和第id个超像素的像素重叠数
				//if(n<filterNumThreshold) continue;

				float test = m*1.0f / n;
				if (test>0.75)//&&n>filterNumThreshold 超像素内某种标签的mask占了75%以上，就把mask上该超像素内所有像素都置为mask标签值
				{
					maskrcnnMask[i*320*240 + y*320 + x] = PixelValueOfEachMask[i];
				}
				else
				{
					maskrcnnMask[i*320*240 + y*320 + x] = 0;
				}
			}
		}
	}


	free(NumMatrix);
	free(PixelValueOfEachMask);
}