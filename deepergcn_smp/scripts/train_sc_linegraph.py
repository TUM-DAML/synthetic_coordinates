from seml_scripts.graph_clsreg import run

result = run(
    dataset_name="ZINC",
    model="smp",
    # optim params
    lr_schedule=True,
    min_lr=1e-5,
    patience=10,
    max_epochs=3000,
    learning_rate=1e-4,
    l2_reg=0,
    batch_size=128,
    lr_warmup=False,
    # random seed
    seed=0,
    # our new features
    add_ppr_dist=True,
    add_rdkit_dist="bounds_matrix_both",
    linegraph_dist=True,
    linegraph_angle=True,
    # use the lowest possible angle/highest possible angle/center angle
    # only in combination with add_rdkist_dist=bounds_matrix_both
    linegraph_angle_mode="center_both",
    # basis dimension
    dist_basis=8,
    dist_basis_type="gaussian",
    angle_basis=9,
    # multiply the distance/angle embedding with the message
    emb_product=True,
    emb_use_both=True,
    emb_basis_global=True,
    emb_basis_local=True,
    emb_bottleneck=4,
    # model params
    mlp_act="relu",
    num_layers=12,
    dropout=0.2,
    block="res+",
    conv_encode_edge=True,
    add_virtual_node=False,
    hidden_channels=128,
    conv="gen",
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
    # run on small dataset
    quick_run=False,
    # use metric graph when distances are available
    metric_graph_cutoff=None,
    metric_graph_edgeattr=None,
    max_hours=144,
)

print(result["best_val"], result["final_test"])
