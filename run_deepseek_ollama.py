import ollama
import argparse

chat_history = []
MODEL = "deepseek-r1:1.5b"

def send_message_to_deepseek(user_message):
    global chat_history
    chat_history.append({"role":"user", "content":user_message})
    
    ## 출력된 결과를 스트리밍으로 바로 print하기 위해 변경되었음. --> 긴 글인 경우 오랫동안 기다림을 방지
    # response = ollama.chat(model=MODEL,  messages=chat_history)
    # assistant_reply = response["message"]["content"]
    # chat_history.append({"role":"assistant","content":assistant_reply})

    response = ollama.chat(model=MODEL, messages=chat_history, stream=True)

    assistant_reply = ""
    print("🤖: ", end="", flush=True)

    for chunk in response:
        chunk_text = chunk["message"]["content"]
        assistant_reply += chunk_text
        print(chunk_text, end="", flush=True)
    print("\n")
    chat_history.append({"role":"assistant", "content":assistant_reply})

    return assistant_reply

def update_to_system_prompt(system_prompt):
    global chat_history
    chat_history.append({"role":"system", "content":system_prompt})

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--update_system",
        action="store_true",
        help="Update to system prompt"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse()
    if args.update_system:
        system_prompt = input("Input System Prompt >> ")
        update_to_system_prompt(system_prompt)

    while True:
        user_message = input(">> ")
        if user_message.lower() == "exit":
            print("Bye!👋")
            break
        response = send_message_to_deepseek(user_message)
        print(f"🤖: {response}\n")