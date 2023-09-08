import os
from omegaconf import OmegaConf
from perfusion_pytorch.embedding import EmbeddingWrapper
from perfusion_pytorch import Rank1EditModule
from perfusion_pytorch.ldm.util import instantiate_from_config
import torch
import argparse
from tqdm.auto import tqdm
from utils import count_params, count_trainable_params

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=r"D:\\PycharmProjects\\perfusion0907\\configs\\perfusion_teddy.yaml")
parser.add_argument('--accumulator', type=int, default=4)
parser.add_argument('--eval_fre', type=int, default=25)
args = parser.parse_args()

def exists(val):
    return val is not None

def set_trainable_parameters(text_image_model):
    for param in text_image_model.parameters():
        param.requires_grad = False

    embed_params = None
    key_value_params = []

    for module in text_image_model.modules():
        if isinstance(module, EmbeddingWrapper):
            assert not exists(embed_params), 'there should only be one wrapped EmbeddingWrapper'
            module.concepts.requires_grad = True
            embed_params = module.concepts.data

        elif isinstance(module, Rank1EditModule):
            # module.ema_concept_text_encs.requires_grad = True
            if not module.is_key_proj:
                module.concept_outputs.requires_grad = True
                key_value_params.append([
                    # module.ema_concept_text_encs.data,
                    module.concept_outputs.data
                ])
            # else:
            #     key_value_params.append([
            #         module.ema_concept_text_encs.data])
    pkg = dict(
        embed_params = embed_params,
        key_value_params = key_value_params,
    )

    torch.save(pkg, 'tmp.pth')

    return text_image_model, embed_params, key_value_params

def build_optimizers(embed_params, key_value_params, cfg):
    new_key_value_params = []
    for x in key_value_params:
        if isinstance(x, list):
            new_key_value_params += x
        else:
            new_key_value_params.append(x)
    opt = torch.optim.AdamW(
        [{"params": embed_params, "lr": cfg.training.embed_lr}, {"params": new_key_value_params}],
        lr=cfg.training.key_value_lr,
    )
    return opt

def get_saved_params(text_image_model):
    embed_params = None
    key_value_params = []
    C_inv = None

    for module in text_image_model.modules():
        if isinstance(module, EmbeddingWrapper):
            assert not exists(embed_params), 'there should only be one wrapped EmbeddingWrapper'
            embed_params = module.concepts.data

        elif isinstance(module, Rank1EditModule):
            key_value_params.append([
                module.ema_concept_text_encs.data,
                module.concept_outputs.data
            ])

            C_inv = module.C_inv.data

    assert exists(C_inv), 'Rank1EditModule not found. you likely did not wire up the text to image model correctly'
    pkg = dict(
        embed_params=embed_params,
        key_value_params=key_value_params,
        C_inv=C_inv
    )

    return pkg

if __name__ == '__main__':
    torch_dtype = torch.float32
    config = OmegaConf.load(args.cfg)
    text_image_model = instantiate_from_config(config.model)
    text_image_model, embed_params, key_value_params = set_trainable_parameters(text_image_model)
    text_image_model = text_image_model.to(torch.device('cuda'))

    count_trainable_params(text_image_model, verbose=True)
    count_params(text_image_model, verbose=True)
    dataset = instantiate_from_config(config.data)
    optimizer = build_optimizers(embed_params, key_value_params, config)


    def collate_fn(examples):
        images = []
        captions = []
        for example in examples:
            images.append(torch.from_numpy(example['image']))
            captions.append(example['caption'])

        images = torch.stack(images, 0)

        dicts = {}
        dicts['images'] = images
        dicts['captions'] = captions
        return dicts

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=config.training.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler()

    first_epoch = 0
    global_step = 0
    max_train_steps = config.training.epochs * len(train_dataloader) // args.accumulator
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, config.training.epochs):
        text_image_model.train()
        for step, batch in enumerate(train_dataloader):
            # with torch.cuda.amp.autocast():
            loss = text_image_model.run(batch, torch_dtype)
            loss = loss / args.accumulator
            loss.backward()
            # scaler.scale(loss).backward()
            global_step += 1

            if global_step > 0 and global_step % args.accumulator == 0:
                optimizer.step()
                optimizer.zero_grad()
                # scaler.step(optimizer)
                # scaler.update()
                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)


        if global_step > 0 and global_step % (args.eval_fre*args.accumulator) == 0:
            text_image_model.eval()
            # pkg = get_saved_params(text_image_model)
            # torch.save(pkg, f'ckpts\\{global_step}.ckpt')
            os.makedirs(f'ckpts\\{int(global_step // (args.eval_fre*args.accumulator))}', exist_ok=True)
            images = text_image_model.predict(text_image_model, 'A teddy is playing with a ball in the water', num_imgs=1)
            for index, img in enumerate(images):
                images[index].save(os.path.join(f'ckpts\\{int(global_step // (args.eval_fre*args.accumulator))}', f'{int(global_step // (args.eval_fre*args.accumulator))}.jpg'))
            text_image_model.train()
