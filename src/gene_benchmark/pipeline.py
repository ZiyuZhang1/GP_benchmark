from __future__ import annotations

from gene_benchmark.config import Config, default_config
from gene_benchmark.pipelines.diffusion import run_diffusion
from gene_benchmark import main_gnn, main_mf, main_nn_non_para, main_occ


def run_all(config: Config | None = None) -> None:
    """
    Execute the full workflow sequentially with a shared config.
    """
    cfg = config or default_config()

    # Diffusion-based pipeline
    run_diffusion(cfg)

    # GNN pipeline
    main_gnn.main()

    # Matrix factorization
    main_mf.main()

    # Deep NN fusion
    main_nn_non_para.main()

    # OCC pipeline
    main_occ.main()


__all__ = ["run_all"]
