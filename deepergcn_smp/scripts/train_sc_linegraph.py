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
    add_rdkit_dist="bounds_matrix_both",
    linegraph_dist=True,
    linegraph_angle=True,
    linegraph_angle_mode="center_both",
    # basis dimension
    dist_basis=16,
    dist_basis_type="gaussian",
    angle_basis=18,
    # multiply the distance/angle embedding with the message
    emb_basis_global=True,
    emb_basis_local=True,
    emb_bottleneck=4,
    # model params
    num_layers=12,
    hidden_channels=128,
    # target to predict in QM9
    qm9_target_ndx=7,
    # run on a subset of data
    quick_run=False,
    # max time to run
    max_hours=144,
    # log metrics and losses to tensorboard
    log=False,
)

print(result["best_val"], result["final_test"])
