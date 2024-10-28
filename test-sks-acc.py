# orig_embeds_params = model.get_input_embeddings().weight.data.clone()
import argparse
import glob
import os

import torch
from llava.eval.my_llava import *
from llava.mm_utils import (get_model_name_from_path, tokenizer_image_token,
                            tokenizer_image_token_batch)
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
    parser.add_argument("--data_root", type=str, default='./yollava-data/test/')
    parser.add_argument("--sks_name", type=str, default='shiba-yellow')
    parser.add_argument("--stage", type=str, default='s2')

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prefix_token", type=int, default=4)
    #--- Log related
    parser.add_argument("--exp_name", type=str, default='multi-token')
    parser.add_argument("--save_txt", action='store_true', default=False)
    parser.add_argument("--system_prompt", default=False, action='store_true')
    parser.add_argument("--suffix_prompt", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path)
    )
    
    prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
    placeholder_tokens = [f'<{args.sks_name}>']
    placeholder_tokens.extend(prefix_tokens)
    
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Load the token and lm_head embeddings
    sks_token = torch.load(f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-token.pt').detach()
    lm_head = torch.load(f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-lmhead.pt').detach()
    model.get_input_embeddings().weight.requires_grad = False
    model.lm_head.weight.requires_grad = False
    model.get_input_embeddings().weight[placeholder_token_ids] = sks_token.to(model.device, dtype=model.dtype)
    model.lm_head.weight[placeholder_token_ids] = lm_head.detach().to(model.lm_head.weight.device, dtype=model.dtype)
    print('New tokens are loaded into: ', placeholder_token_ids)

    # sks_prompt = f"{placeholder_tokens[0]} is {' '.join(placeholder_tokens[1:])}."
    if args.prefix_token > 0:
        prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
        placeholder_tokens = [f'<{args.sks_name}>']
        placeholder_tokens.extend(prefix_tokens)
        if args.suffix_prompt is not None:
            # breakpoint()
            sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}"
        else:
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}"
        print('system prompt will add:', sks_prompt)
    else:
        placeholder_tokens = [f'<{args.sks_name}>']
        sks_prompt = placeholder_tokens[0]
        print('system prompt will add:', sks_prompt)
        
    print('Learned prompt: ', sks_prompt)
    if args.system_prompt:
        args = get_query(args, f"Is <{args.sks_name}> in this photo? Answer with a single word or phrase.", model=model, sks_system_prompt=sks_prompt)

    else:
        args = get_query(args, sks_prompt + f" Can you see <{args.sks_name}> in this photo? Answer with a single word or phrase.", model=model, sks_system_prompt=None)
    
    categories = os.listdir(args.data_root)
    if 'cc12m_images' in args.data_root:
        categories = [args.sks_name]
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')
        
    os.makedirs(f"./quantitative/{args.sks_name}", exist_ok=True)
    print('Categories: ')
    if args.save_txt:
        for category in categories:
            with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
                f.write(f'{category}\n')

    
    if args.save_txt:
        with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
            f.write(f'Results for {args.sks_name} with epoch {args.epoch} and setting {args.exp_name}\n')
        print('Results will be saved in: ', f"./quantitative/{args.sks_name}/acc.txt")
    print('✦ . 　⁺ 　 . ✦ . 　⁺ 　 . ✦ Accuracy by category: ')
    for category in categories:
        list_imgs =[]
        for ext in ['jpg', 'jpeg', 'png', "JPG", "JPEG", "PNG"]:
            list_imgs.extend(glob.glob(os.path.join(args.data_root, category, f'*.{ext}')))
            # if len(list_imgs)>0:
            #     break
        # list_imgs = glob.glob(os.path.join(args.data_root, category, '*.*'))
        pred = []
        list_incorrect = []
        for image_file in list_imgs:
            try:
                images_tensor, image_sizes = get_image_tensor(args, [image_file], model, image_processor)
                output, pred_ids = eval_model(args,
                                    model=model,
                                    images_tensor=images_tensor, #images_tensor,
                                    image_sizes=image_sizes,
                                    image_processor=image_processor,
                                    tokenizer=tokenizer,
                                    return_ids=True)
                # print(output)
                assert output in ['Yes', 'No']
                pred.append(output)
            except Exception as e:
                print(e)
                # list_incorrect.append(image_file)
                pass
        if category == args.sks_name:
            if 'laion' in args.data_root:
                gt = ['No']*len(pred)
            else:
                gt = ['Yes']*len(pred)
        else:
            gt = ['No']*len(pred)
        true_pos = np.array(pred)==np.array(gt)
        acc = true_pos.sum()/len(gt)
        # print(category)
        print(f'GT: {gt}; Pred: {pred}')
        print(f'{category}: {acc}')
        print(acc)
        if args.save_txt:
            with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
                f.write(f'{acc}\n')
