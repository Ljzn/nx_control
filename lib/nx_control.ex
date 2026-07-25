defmodule NxControl do
  @moduledoc """
  NxControl — Numerical Control Library for Elixir.

  Built on Nx, provides transfer function analysis, Routh-Hurwitz
  stability criterion, pole/zero computation via the DKA algorithm,
  and frequency response evaluation.

  ## Example

      # Create G(s) = 1 / (s² + 3s + 2) and check stability
      tf = NxControl.TransferFunction.new([1], [1, 3, 2])
      NxControl.TransferFunction.stable?(tf)
      # => true

      NxControl.TransferFunction.poles(tf)
      # => #Nx.Tensor<c128[2] [-1.0+0.0i, -2.0+0.0i]>

  ## Core modules

      #{NxControl.TransferFunction}  — Transfer function representation and analysis
  """
end
