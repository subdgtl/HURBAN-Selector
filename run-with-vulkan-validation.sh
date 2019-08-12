#!/bin/bash

# VK_LAYER_PATH may differ on your system
VK_LAYER_PATH=/c/VulkanSDK/1.1.114.0/Bin VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_VALIDATION cargo run --features vulkan
