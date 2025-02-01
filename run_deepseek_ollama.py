import ollama
import argparse
import sys

chat_history = []
MODEL = ""


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


def update_model_name(args):
    global MODEL
    if "1.5b" in args.model or "1.5" in args.model:
        MODEL = "deepseek-r1:1.5b"
    if "7b" in args.model or "7" in args.model:
        MODEL = "deepseek-r1:7b"
    if "8b" in args.model or "8" in args.model:
        MODEL = "deepseek-r1:8b"
    if "14b" in args.model or "14" in args.model:
        MODEL = "deepseek-r1:14b"
    if "32b" in args.model or "32" in args.model:
        MODEL = "deepseek-r1:32b"
    if "70b" in args.model or "70" in args.model:
        MODEL = "deepseek-r1:70b"
    if "671b" in args.model or "671" in args.model:
        MODEL = "deepseek-r1:671b"


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--update_system",
        action="store_true",
        help="Update to system prompt"
    )

    parser.add_argument(
        "-m", "--model",
        dest="model",
        type=str,
        # default="deepseek-r1:1.5b",
        default="",
        help="set model name"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    update_model_name(args)
    if MODEL == "":
        print("Invalid Model Name!")
        print("The available models are as follows.")
        print("deepseek-r1:1.5b\ndeepseek-r1:7b\ndeepseek-r1:8b\ndeepseek-r1:14b\ndeepseek-r1:32b\ndeepseek-r1:70b\ndeepseek-r1:671b")
        sys.exit()

    print(f"Hello! I'm {MODEL}")
    print("How can I help you today?\n")

    if args.update_system:
        system_prompt = input("Input System Prompt >> ")
        update_to_system_prompt(system_prompt)

    while True:
        user_message = input(">> ")
        if user_message == "": continue
        if user_message.lower() == "exit":
            print("Bye!👋")
            break
        response = send_message_to_deepseek(user_message)
        # print(f"🤖: {response}\n")