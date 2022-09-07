import wandb

def init_wandb(args):
    try:
        wandb.init(
            project=args['wandb_project'],
            entity=args['wandb_entity'],
            sync_tensorboard=True,
            config=args,  # dict type
            name=args['wandb_name'],
            monitor_gym=True,
            save_code=True,
        )
    except:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),  # namespace to dict type
            name=args.wandb_name,
            monitor_gym=True,
            save_code=True,
        )        


