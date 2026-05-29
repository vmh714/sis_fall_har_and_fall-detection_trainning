idf_component_register(SRCS "sis_fall_firmware_inference.c" "tflite_wrapper.cpp" "model_data.cc"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_timer esp-tflite-micro)
