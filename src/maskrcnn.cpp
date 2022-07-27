#include "maskrcnn.h"

class Logger : public ILogger,public std::ostream
{
    void log(Severity severity, const char* msg) override
    {
        //不提示INFO信息，只显示警告和错误
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
}gLogger;

//true :  从bin
//false：从uff
bool SampleMaskRCNN::build(std::string serializedEngine,bool isload)
{
    //initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    initLibNvInferPlugins(&gLogger, "");
    //bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    if(!isload)
    {
        //auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder)
        {
            return false;
        }

        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
        if (!network)
        {
            return false;
        }

        auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
        if (!parser)
        {
            return false;
        }

        parser->registerInput(mParams.inputTensorNames[0].c_str(), MaskRCNNConfig::IMAGE_SHAPE, nvuffparser::UffInputOrder::kNCHW);
        for (size_t i = 0; i < mParams.outputTensorNames.size(); i++)
            parser->registerOutput(mParams.outputTensorNames[i].c_str());

        auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
        if (!parsed)
        {
            return false;
        }

        builder->setMaxBatchSize(mParams.batchSize);
        builder->setMaxWorkspaceSize(2_GiB);
        builder->setFp16Mode(mParams.fp16);

        // Only for speed test
        if (mParams.int8)
        {
           setAllTensorScales(network.get());
            builder->setInt8Mode(true);
        }

        //保存engine
        std::cout<<"start build eigen"<<std::endl;
        auto tStart = std::chrono::high_resolution_clock::now();    
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), InferDeleter());
        auto tEnd = std::chrono::high_resolution_clock::now();
        float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        gLogger << "build eigen time is " << (totalHost)<<"ms"<< std::endl;
        //std::cout << "build eigen time is " << (totalHost)<<"ms"<< std::endl;
        std::cout<<"finish build eigen"<<std::endl;

        std::ofstream serialize_output_stream;

        // 将序列化的模型结果拷贝至serialize_str字符串
        IHostMemory* gie_model_stream = mEngine->serialize();
        std::string serialize_str;
        serialize_str.resize( (gie_model_stream)->size() );
        memcpy((void*)serialize_str.data(), gie_model_stream->data(), gie_model_stream->size());

        // 将serialize_str字符串的内容输出至cached_model.bin文件
        serialize_output_stream.open(serializedEngine);
        //serialize_output_stream.open("./weight/maskrcnn_fp16_512.bin");
        serialize_output_stream << serialize_str;
        serialize_output_stream.close();
        
        assert(network->getNbInputs() == 1);
        mInputDims = network->getInput(0)->getDimensions();
        assert(mInputDims.nbDims == 3);
        assert(network->getNbOutputs() == 2);

    }
    else
    {
        //读取engine
        std::cout<<"start build eigen"<<std::endl;
        auto tStart = std::chrono::high_resolution_clock::now();
        
        //std::string cached_path = "./maskrcnn_fp16.bin";
        std::ifstream fin(serializedEngine);
        // 将文件中的内容读取至cached_engine字符串
        std::string cached_engine = "";
        while (fin.peek() != EOF){ // 使用fin.peek()防止文件读取时无限循环
            std::stringstream buffer;
            buffer << fin.rdbuf();
            cached_engine.append(buffer.str());
        }
        fin.close();

        // 将序列化得到的结果进行反序列化，以执行后续的inference
        //IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
        IRuntime* runtime = createInferRuntime(gLogger);
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr), InferDeleter());
        auto tEnd = std::chrono::high_resolution_clock::now();
        float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        std::cout << "build eigen time is " << (totalHost) << "ms"<<std::endl;
    }

    if (!mEngine)
    {
        return false;
    }
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    return true;
}

bool SampleMaskRCNN::infer(cv::Mat  inputimg)
{
    //std::cout<<"sample maskrcnn"<<endl;
    // originalsize = inputimg.size();
    inputimg.copyTo(originalimg);
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    // if (!context)
    // {
    //     return false;
    // }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    std::cout<<"read success"<<endl;

    const int inputC = MaskRCNNConfig::IMAGE_SHAPE.d[0];
    const int inputH = MaskRCNNConfig::IMAGE_SHAPE.d[1];
    const int inputW = MaskRCNNConfig::IMAGE_SHAPE.d[2];

    preprocessImg(inputimg,resizedimg,inputH, inputW);
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float pixelMean[3]{123.7, 116.8, 103.9};    
    for (int c = 0; c < inputC; ++c)
        // The color image to input should be in RGB order
        for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            hostDataBuffer[c * volChl + j] = float(resizedimg.data[j * inputC + c]) - pixelMean[c];

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    bool status;
    auto tStart = std::chrono::high_resolution_clock::now();
    status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    auto tEnd = std::chrono::high_resolution_clock::now();
    float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    std::cout << "语义分割速度:" << totalHost<< " ms/frame" << std::endl;
    if (!status)
    {
        return false;
    }
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool SampleMaskRCNN::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

bool SampleMaskRCNN::sortMask(const std::pair<cv::Mat,BBoxInfo> &mask1, const std::pair<cv::Mat,BBoxInfo> &mask2)
{
    cv::Scalar mask1_sum = cv::sum(mask1.first);
    cv::Scalar mask2_sum = cv::sum(mask2.first);
    return mask1_sum(0) > mask2_sum(0);
}

std::vector<BBoxInfo> SampleMaskRCNN::decodeOutput(void* detectionsHost, void* masksHost)
{
    int input_dim_h = MaskRCNNConfig::IMAGE_SHAPE.d[1], input_dim_w = MaskRCNNConfig::IMAGE_SHAPE.d[2];
    assert(input_dim_h == input_dim_w);
    int image_height = originalimg.rows;
    int image_width = originalimg.cols;
    // resize the DsImage with scale
    const int image_dim = std::max(image_height, image_width);
    int resizeH = (int) image_height * input_dim_h / (float) image_dim;
    int resizeW = (int) image_width * input_dim_w / (float) image_dim;
    // keep accurary from (float) to (int), then to float
    float window_x = (1.0f - (float) resizeW / input_dim_w) / 2.0f;
    float window_y = (1.0f - (float) resizeH / input_dim_h) / 2.0f;
    float window_width = (float) resizeW / input_dim_w;
    float window_height = (float) resizeH / input_dim_h;

    float final_ratio_x = (float) image_width / window_width;
    float final_ratio_y = (float) image_height / window_height;
    std::vector<BBoxInfo> binfo;

    // int detectionOffset = MaskRCNNUtils::volume(MaskRCNNConfig::MODEL_DETECTION_SHAPE); // (100,6)
    // int maskOffset = MaskRCNNUtils::volume(MaskRCNNConfig::MODEL_MASK_SHAPE);           // (100, 81, 28, 28)

    RawDetection* detections = reinterpret_cast<RawDetection*>((float*) detectionsHost);
    Mask* masks = reinterpret_cast<Mask*>((float*) masksHost);
    for (int det_id = 0; det_id < MaskRCNNConfig::DETECTION_MAX_INSTANCES; det_id++)
    {
        RawDetection cur_det = detections[det_id];
        int label = (int) cur_det.class_id;
        if (label <= 0)
            continue;

        BBoxInfo det;
        det.label = label;
        det.prob = cur_det.score;

        det.box.x1 = std::min(std::max((cur_det.x1 - window_x) * final_ratio_x, 0.0f), (float) image_width);
        det.box.y1 = std::min(std::max((cur_det.y1 - window_y) * final_ratio_y, 0.0f), (float) image_height);
        det.box.x2 = std::min(std::max((cur_det.x2 - window_x) * final_ratio_x, 0.0f), (float) image_width);
        det.box.y2 = std::min(std::max((cur_det.y2 - window_y) * final_ratio_y, 0.0f), (float) image_height);

        if (det.box.x2 <= det.box.x1 || det.box.y2 <= det.box.y1)
            continue;

        det.mask = masks + det_id * MaskRCNNConfig::NUM_CLASSES + label;

        binfo.push_back(det);
    }

    return binfo;
}


bool SampleMaskRCNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    cv::Mat resized_mask;
    void* detectionsHost = buffers.getHostBuffer(mParams.outputTensorNames[0]);
    void* masksHost = buffers.getHostBuffer(mParams.outputTensorNames[1]);
    results.clear();
    std::vector<BBoxInfo> binfo = decodeOutput(detectionsHost, masksHost);
    std::cout<<"maskrcnn obj:"<<binfo.size()<<std::endl;

    for (size_t roi_id = 0; roi_id < binfo.size(); roi_id++)
    {
        if(binfo[roi_id].prob>0.8)
        {
            const auto _mask = resizeMask_mat(binfo[roi_id], mParams.maskThreshold,originalimg.size()); // mask threshold
            // MaskRCNNUtils::addBBoxImg(originalimg, binfo[roi_id], resized_mask);
            cv::resize(_mask,resized_mask,cv::Size(320,240),CV_INTER_NN);
            resized_mask = resized_mask*binfo[roi_id].label;
            results.push_back(std::pair<cv::Mat,BBoxInfo>(resized_mask,binfo[roi_id]));
            std::cout<<binfo[roi_id].prob<<"   "<<MaskRCNNConfig::CLASS_NAMES[binfo[roi_id].label]<<std::endl;
        }
    }


    std::sort(results.begin(),results.end(),sortMask);
    return true;
}


    inline void SampleMaskRCNN::setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f)
    {
        // Ensure that all layer inputs have a scale.
        for (int i = 0; i < network->getNbLayers(); i++)
        {
            auto layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbInputs(); j++)
            {
                ITensor* input{layer->getInput(j)};
                // Optional inputs are nullptr here and are from RNN layers.
                if (input != nullptr && !input->dynamicRangeIsSet())
                {
                    ASSERT(input->setDynamicRange(-inScales, inScales));
                }
            }
        }

        // Ensure that all layer outputs have a scale.
        // Tensors that are also inputs to layers are ingored here
        // since the previous loop nest assigned scales to them.
        for (int i = 0; i < network->getNbLayers(); i++)
        {
            auto layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbOutputs(); j++)
            {
                nvinfer1::ITensor* output{layer->getOutput(j)};
                // Optional outputs are nullptr here and are from RNN layers.
                if (output != nullptr && !output->dynamicRangeIsSet())
                {
                    // Pooling must have the same input and output scales.
                    if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                    {
                        ASSERT(output->setDynamicRange(-inScales, inScales));
                    }
                    else
                    {
                        ASSERT(output->setDynamicRange(-outScales, outScales));
                    }
                }
            }
        }
    }


    cv::Mat SampleMaskRCNN::resizeMask_mat(const BBoxInfo& box, const float mask_threshold,cv::Size size)
    {
        cv::Mat result;
        if (!box.mask)
        {
            assert(result.empty() == 1);
            return result;
        }

        const int h = box.box.y2 - box.box.y1;
        const int w = box.box.x2 - box.box.x1;

        cv::Mat raw_mask = cv::Mat(MaskRCNNConfig::MASK_POOL_SIZE * 2,MaskRCNNConfig::MASK_POOL_SIZE * 2, CV_32FC1, box.mask->raw);
        cv::Mat resized_mask;
        cv::resize(raw_mask,resized_mask,cv::Size(w,h),CV_BILATERAL);
        cv::threshold(resized_mask, result, mask_threshold, 1, CV_THRESH_BINARY);
        result.convertTo(result,CV_8UC1);

        cv::Mat background = cv::Mat::zeros(size,CV_8U);
        cv::Rect rect(box.box.x1,box.box.y1,result.cols,result.rows);
        result.copyTo(background(rect));
        
        return background;
    }


    void SampleMaskRCNN::resizeImg(cv::Mat& src,cv::Mat& dst,int target_heigh,int target_width)
    {
        cv::resize(src,dst,cv::Size(target_width,target_heigh),CV_BILATERAL);
    }

    void SampleMaskRCNN::padImg(cv::Mat& src,cv::Mat& dst,int top, int bottom, int left, int right)
    {
        cv::copyMakeBorder(src,dst,top,bottom,left,right,cv::BORDER_CONSTANT);
    }

    void SampleMaskRCNN::preprocessImg(cv::Mat& src, cv::Mat& dst, int target_h, int target_w)
    {
        assert(target_h == target_w);
        int input_dim = target_h;
        // padding the input img to model's input_size:
        const int image_dim = std::max(src.rows, src.cols);
        int resize_h = src.rows * input_dim / image_dim;
        int resize_w = src.cols * input_dim / image_dim;
        assert(resize_h == input_dim || resize_w == input_dim);

        int y_offset = (input_dim - resize_h) / 2;
        int x_offset = (input_dim - resize_w) / 2;
        cv::Mat resized_img;
        // resize
        resizeImg(src,resized_img,resize_h,resize_w);
        // pad
        padImg(resized_img,dst,y_offset, input_dim - resize_h - y_offset, x_offset, input_dim - resize_w - x_offset);
    }