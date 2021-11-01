import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from seml_scripts.graph_clsreg import run

result = run(
    dataset_name="ZINC",
    model="smp",
    # optim params
    lr_schedule=True,
    min_lr=1e-5,
    patience=10,
    max_epochs=3000,
    learning_rate=1e-3,
    l2_reg=1e-6,
    batch_size=64,
    lr_warmup=False,
    # random seed
    seed=0,
    # our new features
    add_ppr_dist=False,
    add_rdkit_dist=False,
    linegraph_dist=False,
    linegraph_angle=False,
    linegraph_angle_mode="center_both",
    # basis dimension
    dist_basis=16,
    dist_basis_type="gaussian",
    angle_basis=18,
    # multiply the distance/angle embedding with the message
    emb_basis_global=True,
    emb_basis_local=False,
    emb_bottleneck=None,
    # model params
    mlp_act="relu",
    num_layers=12,
    dropout=0.2,
    conv_encode_edge=True,
    hidden_channels=128,
    gcn_aggr="softmax",
    learn_t=True,
    t=0.1,
    learn_p=False,
    p=1,
    msg_norm=False,
    learn_msg_scale=False,
    norm="batch",
    mlp_layers=1,
    graph_pooling="mean",
    qm9_target_ndx=7,
    quick_run=True,
    max_hours=144,
    log=False,
)

print(result["best_val"], result["final_test"])
