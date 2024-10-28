# orig_embeds_params = model.get_input_embeddings().weight.data.clone()
import glob
import os

import torch

from llava.eval.my_llava import *
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    #--- Model related
    parser.add_argument("--model_path", type=str, default="./llava_ckpts/llava-v1.6-internal-vicuna-13b-336px")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)

    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--epoch", type=str, default='2')
    parser.add_argument("--data_root", type=str, default='./yollava-data/test')
    parser.add_argument("--sks_name", type=str, default='bo')

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prefix_token", type=int, default=4)
    #--- Log related
    parser.add_argument("--exp_name", type=str, default='multi-token')
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--stage", type=str, default='s2')
    parser.add_argument("--suffix_prompt", type=str, default=None)


    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    prompts = [
        f"Describe <{args.sks_name}> in details.",
    ]

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path)
    )

    model.get_input_embeddings().weight.requires_grad = False
    model.lm_head.weight.requires_grad = False
    
    # placeholder_tokens = [f"<{args.sks_name}>", '<adj1>', '<adj2>', '<noun>']
    prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
    # placeholder_tokens = [f'<{args.sks_name}>']
    placeholder_tokens = [f'<{args.sks_name}>']
    placeholder_tokens.extend(prefix_tokens)
    # sks_prompt = f'<{args.sks_name}> is <adj1> <adj2> <noun>.'
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    model.resize_token_embeddings(len(tokenizer))
    sks_token = torch.load(f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-token.pt').detach()
    lm_head = torch.load(f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-lmhead.pt').detach()
    
    model.get_input_embeddings().weight[placeholder_token_ids] = sks_token.to(dtype=model.get_input_embeddings().weight.dtype)
    model.lm_head.weight[placeholder_token_ids] = lm_head.detach().to(dtype=model.lm_head.weight.dtype, device=model.lm_head.weight.device)

    # model.resize_token_embeddings(len(tokenizer))
    if args.prefix_token > 0:
        prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
        placeholder_tokens = [f'<{args.sks_name}>']
        placeholder_tokens.extend(prefix_tokens)
        if args.suffix_prompt is not None:
            sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}"
            sks_prompt = sks_prompt.replace('<sks>',f'<{args.sks_name}>')
        else:
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}."
        print('system prompt will add:', sks_prompt)
    else:
        placeholder_tokens = [f'sks']
        sks_prompt = None
        print('system prompt will add:', sks_prompt)
        
    print('Learned prompt: ', sks_prompt)

    for input_prompt in prompts:
        # print('====')
        prompt = get_query(args, sks_prompt + ' ' + input_prompt, model=model, sks_system_prompt=None).conv.get_prompt().replace('<image>\n', '')
        input_ids = [tokenizer.encode(prompt)]
        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
        output = model.generate(input_ids)

        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print('Q: ', input_prompt)
        print('A: ', answer)
        print('====================================')
        # breakpoint()
