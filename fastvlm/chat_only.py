import argparse
import torch
import time
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates

def chat(args):
    # 1. Check GPU status
    print(f"\n--- GPU Check ---")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM before load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("WARNING: Running on CPU! This will be very slow.")

    # 2. Load Model
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    
    print(f"\nLoading model: {model_name}...")
    start_time = time.time()
    
    # Force loading on CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        device=device
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    if torch.cuda.is_available():
        print(f"VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Model device: {model.device}")

    # 3. Text-only Loop
    print("\nModel ready! Type 'exit' to quit.\n")

    # Debug: Print stop token info
    stop_token_ids = [tokenizer.eos_token_id]
    if "<|im_end|>" in tokenizer.get_vocab():
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop_token_ids.append(im_end_id)
        print(f"Added <|im_end|> (ID: {im_end_id}) to stop tokens.")
    
    while True:
        try:
            query = input("User: ")
        except EOFError:
            break
            
        if query.lower() in ["exit", "quit"]:
            break
        
        if not query.strip():
            continue

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Generate
        start_gen = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
                repetition_penalty=1.2
            )
        
        gen_time = time.time() - start_gen
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Calculate tokens per second
        num_tokens = len(output_ids[0])
        print(f"Assistant: {outputs}")
        print(f"(Generated {num_tokens} tokens in {gen_time:.2f}s - {num_tokens/gen_time:.2f} t/s)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    chat(args)
