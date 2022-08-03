#include "yololayer.h"

namespace Tn
{
    template<typename T> 
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> 
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    YoloLayerPlugin::~YoloLayerPlugin()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloLayerPlugin::initialize() TRT_NOEXCEPT
    {
        return 0;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        //output the result to channel
        int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

    const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT
    {
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int noElements,
        const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes;
        int info_len_kpt = KEY_POINTS_NUM * 3;
        const float* curInput = input + bnIdx * ((info_len_i + info_len_kpt) * total_grid * CHECK_COUNT);

        for (int k = 0; k < CHECK_COUNT; ++k) {
            float box_prob = Logist(curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0; //person class
            // float max_cls_prob = 0.0;
            // for (int i = 5; i < info_len_i; ++i) {
            //     float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
            //     if (p > max_cls_prob) {
            //         max_cls_prob = p;
            //         class_id = i - 5;
            //     }
            // }
            float max_cls_prob = Logist(curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + 5 * total_grid]);

            float *res_count = output + bnIdx * outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject) return;
            char *data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
            Detection *det = (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //------------bboxs------------
            //Location
            // pytorch:
            // y = x[i].sigmoid()
            // xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            // wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
            
            det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + 1 * total_grid])) * netheight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth
            // v5: https://github.com/ultralytics/yolov5/issues/471
            det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
            det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
            det->conf = box_prob * max_cls_prob;
            det->class_id = class_id;

            //------------keypoints------------
            // Location
            //pytorch:
            // x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
            // x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
            // x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()
            for (int kpt_idx = 0; kpt_idx < KEY_POINTS_NUM; ++kpt_idx)
            {
                det->kpts[kpt_idx].x = (col - 0.5f + 2.0f * (curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + (6 + kpt_idx*3) * total_grid]) ) * netwidth / yoloWidth;
                det->kpts[kpt_idx].y = (row - 0.5f + 2.0f * (curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + (7 + kpt_idx*3) * total_grid]) ) * netheight / yoloHeight;
                det->kpts[kpt_idx].kpt_conf = Logist(curInput[idx + k * (info_len_i + info_len_kpt) * total_grid + (8 + kpt_idx*3) * total_grid]);                
            }

        }
    }

    void YoloLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize)
    {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width * yolo.height * batchSize;
            if (numElem < mThreadCount) mThreadCount = numElem;

            //printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
            CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
                (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem);
        }
    }


    int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
    {
        // int class_count = 1;//p_netinfo[0];
        // int input_w = 960;//p_netinfo[1];
        // int input_h = 960;//p_netinfo[2];
        // int max_output_object_count = 1000;//p_netinfo[3];
        
        std::vector<Yolo::YoloKernel> kernels{Yolo::yolo4, Yolo::yolo3, Yolo::yolo2, Yolo::yolo1};
        
        YoloLayerPlugin* obj = new YoloLayerPlugin(CLASS_NUM, INPUT_W, INPUT_H, MAX_OUTPUT_BBOX_COUNT, kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}