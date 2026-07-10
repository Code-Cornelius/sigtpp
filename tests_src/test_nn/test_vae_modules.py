import torch

from src.nn.nn.vae_modules.vae_encoder import VAEEncoder

H, D, N, L = 16, 8, 4, 10


def test_vae_encoder_output_shapes():
    enc = VAEEncoder(hidden_size=H, latent_dim=D)
    tau_i = torch.randn(N, L, 1)
    h = torch.randn(N, L, H)
    mu, log_var = enc(tau_i, h)
    assert mu.shape == (N, L, D)
    assert log_var.shape == (N, L, D)


def test_vae_encoder_reparameterize_shape():
    enc = VAEEncoder(hidden_size=H, latent_dim=D)
    mu = torch.zeros(N, L, D)
    log_var = torch.zeros(N, L, D)
    z = VAEEncoder.reparameterize(mu, log_var)
    assert z.shape == (N, L, D)


def test_vae_encoder_reparameterize_stochastic():
    mu = torch.zeros(N, L, D)
    log_var = torch.zeros(N, L, D)
    z1 = VAEEncoder.reparameterize(mu, log_var)
    z2 = VAEEncoder.reparameterize(mu, log_var)
    assert not torch.allclose(z1, z2), "reparameterize must be stochastic"


from src.nn.nn.vae_modules.vae_decoder import VAEDecoder


def test_vae_decoder_output_shape():
    dec = VAEDecoder(latent_dim=D, hidden_size=H)
    z = torch.randn(N, L, D)
    h = torch.randn(N, L, H)
    tau_hat = dec(z, h)
    assert tau_hat.shape == (N, L, 1)


def test_vae_decoder_sample_shape():
    dec = VAEDecoder(latent_dim=D, hidden_size=H)
    h = torch.randn(N, L, H)
    tau_hat = dec.sample(h)
    assert tau_hat.shape == (N, L, 1)


def test_vae_decoder_sample_stochastic():
    dec = VAEDecoder(latent_dim=D, hidden_size=H)
    h = torch.randn(N, L, H)
    s1 = dec.sample(h)
    s2 = dec.sample(h)
    assert not torch.allclose(s1, s2), "sample() must be stochastic"
