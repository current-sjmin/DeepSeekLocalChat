# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import time


# if __name__ == "__main__":
#     # 모델명 설정
#     model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"

#     # 토크나이저 및 모델 로드
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu") # cpu모델에서는 fp32로 일단 해야하는듯 값이 nan이 나옴(fp16)
#     (model_name.split("/"))[-1]
#     print("Hello! I'm", (model_name.split("/"))[-1])
#     print("Enter the prompt.")
#     while True:
#         input_text = input(">> ")
#         inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

#         # 모델 실행 (텍스트 생성)
#         output = model.generate(**inputs,
#                                 max_new_tokens=256,
#                                 eos_token_id=tokenizer.eos_token_id, # 종료 토큰 추가
#                                 temperature=0.7,                     # 적당한 창의성 유지
#                                 top_p=0.9,                           # 반복 방지)
#                                 repetition_penalty=1.2)              # 확률 높은 단어 위주로 선택)

#         # 결과 출력
#         print(tokenizer.decode(output[0], skip_special_tokens=True))
#         time.sleep(1)


##############
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import sys

MAX_TOKENS = 2048
TEMPERATURE = 0.7

if __name__ == "__main__":
    # 모델명 설정
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"

    # 토크나이저 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")  # CPU에서는 fp32 사용

    print("Hello! I'm", (model_name.split("/"))[-1])
    print("Enter the prompt.")

    while True:
        input_text = input(">> ")
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # 모델 실행 (토큰을 하나씩 생성하면서 출력)
        output_ids = inputs["input_ids"]

        with torch.no_grad():
            for _ in range(MAX_TOKENS):  # 최대 256개 토큰 생성
                outputs = model(output_ids)
                logits = outputs.logits[:, -1, :]  # 마지막 토큰의 로짓값 가져오기

                # 온도 및 top-p 샘플링 적용
                probs = torch.nn.functional.softmax(logits / TEMPERATURE, dim=-1)  # temperature=0.7 적용
                next_token_id = torch.multinomial(probs, num_samples=1)  # 확률적으로 샘플링

                # 종료 토큰이면 멈추기
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

                # 입력 업데이트
                output_ids = torch.cat([output_ids, next_token_id], dim=-1)

                # 생성된 토큰 즉시 출력
                generated_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                print(generated_text, end="", flush=True)  # 실시간 출력
                # time.sleep(0.05)  # 부드러운 출력 효과

        print("\n")  # 줄 바꿈
