from train import run

res = run(model_name="dimenet++", 
        # emb sizes
        emb_size=4, out_emb_size=4, 
        int_emb_size=4, basis_emb_size=4,
        # model size
        num_blocks=1,
        # rbf and sbf embs 
        num_radial=12, num_spherical=12, 
        output_init='GlorotOrthogonal',
        num_before_skip=1, num_after_skip=2, 
        num_dense_output=3,
        cutoff=2.5, 
        # distance - ppr or rdkit or both
        # dist={'type': 'ppr_rdkit_bounds', 'alpha': 0.15}, 
        # dist={'type': 'rdkit_bounds'}, 
        dist={'type': 'ppr', 'alpha': 0.15}, 
        dataset='datasets/qm9', 
        num_train=None, num_valid=None,
        data_seed=42, num_steps=3000000, 
        learning_rate=0.001, ema_decay=0.999,
        decay_steps=4000000, warmup_steps=None, 
        decay_rate=0.01, batch_size=32,
        evaluation_interval=1, save_interval=10000,
        restart=None, targets=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'],
        comment="DimeNet++", logdir='logs',
        quick_run=False,
        ablation=None)