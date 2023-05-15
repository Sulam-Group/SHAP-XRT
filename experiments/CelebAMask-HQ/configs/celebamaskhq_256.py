import os
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.name = name = os.path.basename(__file__.split(".")[0])
    config.deepspeed_config = os.path.join("configs", "ds", f"{name}.json")
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 64
    training.n_iters = 2400001
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100

    training.snapshot_freq_for_preemption = 5000

    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.sde = "vesde"
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 50
    evaluate.end_ckpt = 96
    evaluate.batch_size = 512
    evaluate.enable_sampling = True
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "CelebAMask-HQ"
    data.image_size = 256
    data.centered = False
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = "ncsnpp"
    model.sigma_max = 348
    model.sigma_min = 0.01
    model.scale_by_sigma = True
    model.num_scales = 2000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "output_skip"
    model.progressive_input = "input_skip"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    return config
