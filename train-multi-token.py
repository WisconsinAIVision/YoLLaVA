import json

import torch
import torch.nn as nn
from llava.eval.my_llava import *
from llava.mm_utils import (get_model_name_from_path, tokenizer_image_token,
                            tokenizer_image_token_batch)
from llava.model.builder import load_pretrained_model
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

IMAGE_TOKEN_INDEX = -200

def get_train_args():
    parser = argparse.ArgumentParser()
    #--- Model related
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    #--- Dataset related
    parser.add_argument("--data_root", type=str, default='/nobackup/thao-data/dataset/stuffed-animals')
    parser.add_argument("--sks_name", type=str, default='shiba-yellow')
    parser.add_argument("--prefix_token", type=int, default=4)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--train_lm_head", default=False, action='store_true')
    parser.add_argument("--user_prompt", default=False, action='store_true')
    parser.add_argument("--extreme_negative", default=False, action='store_true')
    parser.add_argument("--recog_only", default=False, action='store_true')
    parser.add_argument("--random_image", default=False, action='store_true')
    parser.add_argument("--text_only", default=False, action='store_true')
    parser.add_argument("--suffix_prompt", default=None, type=str)
    
    #--- Log related
    parser.add_argument("--tensorboard_path", type=str, default='./runs/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/')
    parser.add_argument("--exp_name", type=str, default='./debug/')
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=20)
    train_args = parser.parse_args()
    return train_args

if __name__ == "__main__":

    args = get_train_args()
    
    writer = SummaryWriter(os.path.join(args.tensorboard_path, args.sks_name, args.exp_name))
    save_location = os.path.join(args.checkpoint_path, args.sks_name, args.exp_name)
    os.makedirs(save_location, exist_ok=True)
    args.model_name = get_model_name_from_path(args.model_path)

    # Get models
    tokenizer, model, image_processor, context_len = get_model(args)
    # model = model.to(torch.float32)

    train_dataset = PersonalizedDataset_Mixture(
        data_root=args.data_root,
        sks_name = args.sks_name,
        tokenizer=tokenizer,
        config=model.config,
        image_processor=image_processor,
        device=model.device,
        flip_p= args.flip_p,
        train_lm_head = args.train_lm_head,
        extreme_negative = args.extreme_negative,
        recog_only = args.recog_only,
        random_image=args.random_image,
        text_only=args.text_only,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    # breakpoint()

    test_dataset = PersonalizedDataset(
        data_root=args.data_root,
        sks_name = args.sks_name,
        train_image_paths = train_dataset.images_path,
        tokenizer=tokenizer,
        config=model.config,
        image_processor=image_processor,
        device=model.device,
        set='test',
        # placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    print('sks is: ', args.sks_name)
    print('Number of training samples:', len(train_dataset))

    # --- Add <sks>
    if args.prefix_token > 0:
        prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
        placeholder_tokens = [f'<{args.sks_name}>']
        placeholder_tokens.extend(prefix_tokens)
        if args.suffix_prompt is not None:
            # breakpoint()
            sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}"
            sks_prompt = sks_prompt.replace('<sks>', f'<{args.sks_name}>')
        else:
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}."
        print('system prompt will add:', sks_prompt)
    else:
        placeholder_tokens = [f'<{args.sks_name}>']
        sks_prompt = f"{placeholder_tokens[0]}"
        print('system prompt will add:', sks_prompt)
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = model.get_input_embeddings().weight.data
    orig_embeds_params = model.get_input_embeddings().weight.data.clone()
    orig_lm_params = model.lm_head.weight.data.clone()
    
    trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]
    # trainable_params.append(model.lm_head.())
    optimizer = torch.optim.AdamW(
        trainable_params, # for optimize the embeddings and the head
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # if args.train_lm_head:
    model.train()
    model.model.requires_grad_(False)
    # else:
    #     model.requires_grad_(False)
    
    model.model.embed_tokens.weight.requires_grad_(True)
    # model.get_input_embeddings().weight = model.get_input_embeddings().weight.to(torch.float32)
    # model.get_input_embeddings().weight.to(torch.float32)
    best_acc = 0
    for epoch in tqdm(range(0, args.epoch)):
        for names, p in model.named_parameters():
            if p.requires_grad:
                print(names, "requires_grad")

        for step, batch in enumerate(tqdm(train_dataloader)):
            #--- Ground Truth Answer
            optimizer.zero_grad()
            if args.user_prompt: # sks_description is in USER PROMPT
                prompt = [get_query(args, sks_prompt + ' '+ x, model=model, sks_system_prompt = None).conv.get_prompt() for x in batch['query']]
            else:
                prompt = [get_query(args, x, model=model, sks_system_prompt = sks_prompt).conv.get_prompt() for x in batch['query']]
            
            prompt = [x + ' '+ y for x, y in zip(prompt, batch['answer'])]
            # print(prompt)
            #--- Train with text only
            if not batch['has_image']:
                prompt = [x.replace('<image>\n', '') for x in prompt]
            input_ids, labels = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            #--- Train with text-only
            # if not batch['has_image']:
            #     outputs = model(input_ids, labels=labels)
            # else:
            #     batch['images'] = batch['images'].to(model.dtype)
            #     outputs = model(input_ids, images=batch['images'][0], labels=labels, image_sizes=batch['image_sizes'])
            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                if not batch['has_image']:
                    outputs = model(input_ids, labels=labels)
                else:
                    outputs = model(input_ids, images=batch['images'][0], labels=labels, image_sizes=batch['image_sizes'])
            loss = outputs.loss
            # --- With AMP
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # --- Without AMP
            loss.backward()
            optimizer.step()
            
            # breakpoint()
            #---- Do not update the embedding matrix except the place holder
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[placeholder_token_ids] = False
            #--- Optional: Update lm_head for sks token only
            # index_no_updates_lmhead = torch.ones((len(tokenizer),), dtype=torch.bool)
            # index_no_updates_lmhead[placeholder_token_ids[:1]] = False
            with torch.no_grad():
                model.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]
                # if args.train_lm_head:
                # model.lm_head.weight[index_no_updates_lmhead] = orig_lm_params[index_no_updates_lmhead]
                model.lm_head.weight[index_no_updates] = orig_lm_params[index_no_updates]
            # torch.cuda.empty_cache()
            
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/Token-Norm', model.get_input_embeddings().weight[placeholder_token_ids].norm().item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/index_no_updates-Norm', model.get_input_embeddings().weight[index_no_updates].norm().item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/lm-head-norm', model.lm_head.weight[placeholder_token_ids].norm().item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/index_no_updates-lm-head', model.lm_head.weight[index_no_updates].norm().item(), epoch * len(train_dataloader) + step)
        
        if epoch % args.log_every == 0:
            print('Save model at: ', save_location)
            save_path_token = os.path.join(save_location, f'{epoch}-token.pt')
            save_path_lmhead = os.path.join(save_location, f'{epoch}-lmhead.pt')
            torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids], save_path_token)
            torch.save(model.lm_head.weight.data[placeholder_token_ids], save_path_lmhead)

        with torch.no_grad():
            print('Test')
            list_pred = []
            list_gt = []
            for j, batch in enumerate(tqdm(test_dataloader)):
                #--- Ground Truth Answer
                if args.user_prompt: # sks_description is in USER PROMPT
                    prompt = [get_query(args, sks_prompt + ' '+ x, model=model, sks_system_prompt = None).conv.get_prompt() for x in batch['query']]
                else:
                    prompt = [get_query(args, x, model=model, sks_system_prompt = sks_prompt).conv.get_prompt() for x in batch['query']]
                input_ids, labels = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
                outputs = model.generate(input_ids.cuda(), images=batch['images'][0].cuda(), image_sizes=batch['image_sizes'])
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                list_pred.append(answer)
                list_gt.append(batch['answer'][0])
            list_pred = np.array(list_pred)
            list_gt = np.array(list_gt)

            index_yes = np.where(np.array(list_gt)=='Yes')[0] # where the image is sks
            index_no = np.where(np.array(list_gt)=='No')[0] # where the image is not sks
            pred_yes =(list_pred[index_yes] =='Yes').sum()/len(index_yes) # accuracy of predicting sks
            pred_no = (list_pred[index_no] =='No').sum()/len(index_no)
            writer.add_scalar('Accuracy/sks', pred_yes, epoch)
            writer.add_scalar('Accuracy/no-sks', pred_no, epoch)
            current_acc = (pred_yes + pred_no)/2
            writer.add_scalar('Accuracy/ave', current_acc, epoch)
            if (current_acc >= best_acc) and (epoch >4):
                print('Best accuracy: ', current_acc)
                save_path_token = os.path.join(save_location, 'best-token.pt')
                save_path_lmhead = os.path.join(save_location, 'best-lmhead.pt')
                torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids], save_path_token)
                torch.save(model.lm_head.weight.data[placeholder_token_ids], save_path_lmhead)
                best_acc = current_acc
            # writer.add_text('Test/Prediction', str(list_pred), epoch)
            # writer.add_text('Test/GT', str(list_gt), epoch)
