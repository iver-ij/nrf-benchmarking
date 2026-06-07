#include "inference.h"
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "data.h"

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

static constexpr int kTensorArenaSize = 50 * 1024;
static constexpr int kTargetClassIndex = 2;
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;
static int8_t last_quantized_input[SPECTROGRAM_SIZE];
static int8_t last_quantized_output[16];
static uint32_t last_quantized_input_bytes = 0;
static uint32_t last_quantized_output_bytes = 0;

static int8_t quantize_to_int8(float value, float scale, int zero_point) {
    if (scale <= 0.0f) {
        return 0;
    }

    float scaled = value / scale + (float)zero_point;
    int32_t rounded = (int32_t)lrintf(scaled);
    if (rounded > 127) {
        rounded = 127;
    } else if (rounded < -128) {
        rounded = -128;
    }

    return (int8_t)rounded;
}

static int copy_tensor_bytes(int8_t *out, size_t out_len, const int8_t *src, uint32_t src_len) {
    if (out == nullptr || src == nullptr || src_len == 0U || out_len < (size_t)src_len) {
        return -1;
    }
    memcpy(out, src, (size_t)src_len);
    return (int)src_len;
}

extern "C" int inference_setup(void) {
    if (model_input != nullptr && model_output != nullptr && interpreter != nullptr) {
        return 0;
    }

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printk("ERROR: Model schema version mismatch!\n");
        printk("  Model version: %u, Expected: %d\n", (unsigned int)model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static tflite::MicroMutableOpResolver<12> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printk("ERROR: AllocateTensors() failed\n");
        printk("Arena size %d bytes may be too small for model\n", kTensorArenaSize);
        return -1;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    double arena_percent = 100.0 * (double)used_bytes / (double)kTensorArenaSize;
    printk("Tensor arena used: %u / %d bytes (%.1f%%)\n",
           (unsigned int)used_bytes, kTensorArenaSize,
           arena_percent);

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    if (model_input == nullptr || model_output == nullptr) {
        printk("ERROR: Input or output tensors are NULL\n");
        return -1;
    }

    if (model_input->type != kTfLiteInt8 || model_output->type != kTfLiteInt8) {
        printk("ERROR: Expected int8 model tensors (input=%d output=%d)\n",
               model_input->type, model_output->type);
        return -1;
    }
    return 0;
}

extern "C" int run_inference(const float* spectrogram_data, float* output_probability) {
    if (model_input == nullptr || model_output == nullptr ||
        spectrogram_data == nullptr || output_probability == nullptr) {
        return -1;
    }

    int num_elements = model_input->bytes / sizeof(int8_t);
    if (num_elements > SPECTROGRAM_SIZE) {
        printk("ERROR: Input tensor elements (%d) exceed spectrogram size (%d)\n",
               num_elements, SPECTROGRAM_SIZE);
        return -1;
    }

    for (int i = 0; i < num_elements; i++) {
        model_input->data.int8[i] = quantize_to_int8(
            spectrogram_data[i],
            model_input->params.scale,
            model_input->params.zero_point
        );
    }
    memcpy(last_quantized_input, model_input->data.int8, (size_t)num_elements);
    last_quantized_input_bytes = (uint32_t)num_elements;

    if (interpreter->Invoke() != kTfLiteOk) {
        printk("ERROR: Inference failed\n");
        return -1;
    }

    int output_bytes = model_output->bytes;
    if (output_bytes > (int)sizeof(last_quantized_output)) {
        output_bytes = (int)sizeof(last_quantized_output);
    }
    memcpy(last_quantized_output, model_output->data.int8, (size_t)output_bytes);
    last_quantized_output_bytes = (uint32_t)output_bytes;

    if (model_output->bytes <= kTargetClassIndex) {
        printk("ERROR: Output tensor too small for target class index %d\n", kTargetClassIndex);
        return -1;
    }

    *output_probability = (model_output->data.int8[kTargetClassIndex] - model_output->params.zero_point)
                        * model_output->params.scale;

    return 0;
}

extern "C" uint32_t inference_tensor_arena_size(void) {
    return (uint32_t)kTensorArenaSize;
}

extern "C" int inference_copy_last_quantized_input(int8_t *out, size_t out_len) {
    return copy_tensor_bytes(out, out_len, last_quantized_input, last_quantized_input_bytes);
}

extern "C" int inference_copy_last_quantized_output(int8_t *out, size_t out_len) {
    return copy_tensor_bytes(out, out_len, last_quantized_output, last_quantized_output_bytes);
}
