defmodule NxControl.Vision.Report do
  @moduledoc """
  A structured summary of a `NxControl.Vision` analysis.

  Every field is a plain Elixir term (lists/maps) so the report is easy to
  serialize, log, or turn into a text summary with `to_text/1`. See
  `NxControl.Vision.analyze/2` for how it is produced.
  """

  @enforce_keys [:shape, :colors, :hues, :grid, :edges, :svd, :texture]
  defstruct shape: nil,
            colors: nil,
            hues: [],
            grid: nil,
            edges: nil,
            svd: nil,
            texture: nil

  @type t :: %__MODULE__{
          shape: {non_neg_integer(), non_neg_integer()},
          colors: %{
            r: map(),
            g: map(),
            b: map(),
            luminance: float()
          },
          hues: [{String.t(), float()}],
          grid: Nx.Tensor.t(),
          edges: map(),
          svd: map(),
          texture: map()
        }
end
