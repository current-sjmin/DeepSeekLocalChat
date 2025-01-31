from huggingface_hub import list_models


'''
unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
unsloth/DeepSeek-R1-Distill-Qwen-1.5B
unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit
unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit
emredeveloper/DeepSeek-R1-Distill-Qwen-1.5B-4bit
XelotX/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
hdnh2006/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
osllmai-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
osllmai-community/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit
osllmai-community/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit
'''


if __name__ == "__main__":
    models = list_models(filter="deepseek")
    for model in models:
        if "1.5B" in model.modelId:
            print(model.modelId)