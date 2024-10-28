# orig_embeds_params = model.get_input_embeddings().weight.data.clone()
import argparse
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
    parser.add_argument("--data_root", type=str, default='./yollava-data')
    parser.add_argument("--sks_name", type=str, default='shiba-yellow')

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prefix_token", type=int, default=3)
    #--- Log related
    parser.add_argument("--exp_name", type=str, default='multi-token')
    parser.add_argument("--save_json", action='store_true', default=False)
    parser.add_argument("--suffix_prompt", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    # model_path = 'liuhaotian/llava-v1.5-13b'
    args = get_args()
    prompt = f"Write a caption for this photo of <{args.sks_name}>."
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path)
    )

    model.get_input_embeddings().weight.requires_grad = False
    model.lm_head.weight.requires_grad = False

    # --- Create sks token
    prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
    placeholder_tokens = [f'<{args.sks_name}>']
    placeholder_tokens.extend(prefix_tokens)
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    model.resize_token_embeddings(len(tokenizer))
    sks_token = torch.load(f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-token.pt').detach()
    lm_head = torch.load(f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-lmhead.pt', map_location=model.lm_head.weight.device).detach()

    model.resize_token_embeddings(len(tokenizer))
    model.get_input_embeddings().weight[placeholder_token_ids] = sks_token.to(dtype=model.get_input_embeddings().weight.dtype)
    model.lm_head.weight[placeholder_token_ids] = lm_head.detach().to(dtype=model.lm_head.weight.dtype, device=model.lm_head.weight.device)
    print('Trained tokens are loaded in: ', placeholder_token_ids)
    # args = get_query(args, f"<{args.sks_name}> is <adj1> <adj2> <noun>. " + prompt, model=model)

    # sks_prompt = f"{placeholder_tokens[0]} is {' '.join(placeholder_tokens[1:])}."
    if args.prefix_token > 0:
        prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
        placeholder_tokens = [f'<{args.sks_name}>']
        placeholder_tokens.extend(prefix_tokens)
        if args.suffix_prompt is not None:
            # breakpoint()
            sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}."
        else:
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}"
        print('system prompt will add:', sks_prompt)
    else:
        placeholder_tokens = [f'sks']
        sks_prompt = None
        print('system prompt will add:', sks_prompt)
    args = get_query(args, sks_prompt + ' '+ prompt, model=model, sks_system_prompt=None)

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(glob.glob(os.path.join(args.data_root, ext)))
    # image_files = [x for x in image_files if 'henry' in x]
    print(image_files)
    save_dict = {}
    for image_file in image_files:
        print(image_file)
        images_tensor, image_sizes = get_image_tensor(args, [image_file], model, image_processor)
        output, pred_ids = eval_model(args,
                            model=model,
                            images_tensor=images_tensor,#images_tensor,
                            image_sizes=image_sizes,
                            image_processor=image_processor,
                            tokenizer=tokenizer,
                            return_ids=True)
        print(output)
        save_dict[image_file] = output
    if args.save_json:
        os.makedirs(f'./qualitative/{args.sks_name}/{args.exp_name}', exist_ok=True)
        with open(f'./qualitative/{args.sks_name}/{args.exp_name}/{args.epoch}-output.json', 'w') as f:
            json.dump(save_dict, f)
        print('Saved to: ', f'./qualitative/{args.sks_name}/{args.exp_name}/{args.epoch}-output.json')
