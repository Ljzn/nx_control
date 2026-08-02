# NxControl

[![Hex.pm](https://img.shields.io/hexpm/v/nx_control.svg)](https://hex.pm/packages/nx_control)
[![Docs](https://img.shields.io/badge/docs-hexpm-blue.svg)](https://hexdocs.pm/nx_control)
[![License](https://img.shields.io/hexpm/l/nx_control.svg)](https://github.com/Ljzn/nx_control/blob/main/LICENSE)

**NxControl** is a numerical control library for Elixir, built on [Nx](https://github.com/elixir-nx/nx). It provides transfer function analysis, stability criteria, pole/zero computation, and frequency response evaluation — all using pure Nx tensor operations with no external native dependencies.

## Features

- **Transfer function** `G(s) = num(s) / den(s)` representation
- **Stability analysis** — Routh-Hurwitz criterion (characteristic polynomial coefficients)
- **Pole/zero computation** — DKA (Durand-Kerner-Aberth) polynomial root finder
- **Frequency response** — evaluate `G(s)` at any complex frequency `s`
- **No external dependencies** — pure Nx, works on all backends (BinaryBackend, EXLA, EMLX)

## Installation

Add `nx_control` to your `mix.exs`:

```elixir
def deps do
  [
    {:nx_control, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# Create a transfer function G(s) = 1 / (s^2 + 3s + 2)
tf = NxControl.TransferFunction.new([1], [1, 3, 2])

# Routh-Hurwitz stability check (works on coefficients alone)
NxControl.TransferFunction.stable?(tf)
# => true

# Pole computation (via DKA algorithm)
NxControl.TransferFunction.poles(tf)
# => #Nx.Tensor<c128[2] [-1.0+0.0i, -2.0+0.0i]>

# Pole-based stability check (all real parts < 0)
NxControl.TransferFunction.pole_stable?(tf)
# => true

# Unstable system
tf2 = NxControl.TransferFunction.new([1], [1, -1, -6])
NxControl.TransferFunction.pole_stable?(tf2)
# => false  (pole at +3)

# Evaluate G(s) at a complex frequency
s = Nx.tensor(Complex.new(0, 2), type: {:c, 128})
NxControl.TransferFunction.evalf(tf, s)
# => #Nx.Tensor<c128 -0.0769-0.3077i>
```

## Stability Analysis

NxControl provides two complementary methods for stability analysis:

| Method | Function | How it works | When to use |
|--------|----------|--------------|-------------|
| Routh-Hurwitz | `stable?/1` | Examines coefficient signs and builds the Routh array | Fast, no root computation needed; works for any polynomial |
| Pole-based | `pole_stable?/1` | Computes all poles via DKA and checks real parts | Gives exact pole locations for deeper analysis |

## Documentation

Full API documentation is available at [hexdocs.pm/nx_control](https://hexdocs.pm/nx_control).

## SVD-Painted Orange Cat (Example)

`examples/svd_cat.exs` demonstrates `Nx.Lapack.svd` (via the `nx_lapack` dependency) by drawing an orange tabby cat *procedurally* in Nx — no external images — decomposing each RGB channel, and reconstructing the image at different singular-value ranks. The full-rank reconstruction restores the original; low ranks show the classic SVD compression trade-off.

| Original | rank k = 8 | rank k = 32 |
|:---:|:---:|:---:|
| <img src="examples/images/svd_cat/orange_cat_original.png" width="160"> | <img src="examples/images/svd_cat/orange_cat_k8.png" width="160"> | <img src="examples/images/svd_cat/orange_cat_k32.png" width="160"> |

The cat is rendered with soft golden lighting, half-closed eyes and a faint smile to give it a warm, serene mood. The script also writes the results as dependency-free 24-bit BMP bitmaps into `examples/images/svd_cat/`:

```bash
mix run examples/svd_cat.exs
```

The images are also available in `examples/images/svd_cat/` as `.bmp` (raw output) and `.png` (for the web).

## License

NxControl is released under the Apache-2.0 License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/Ljzn/nx_control).
